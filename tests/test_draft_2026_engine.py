from __future__ import annotations

import pandas as pd
import pytest

from ffbayes.draft_2026.engine import (
    build_draft_board,
    calculate_replacement_levels,
    next_snake_pick,
    score_projections,
)
from ffbayes.draft_2026.league import LeagueProfile, LeagueProfileError
from ffbayes.draft_2026.sources import SemanticInputError


def _profile(**overrides) -> LeagueProfile:
    data = {
        'profile_id': 'test-league',
        'league_name': 'Test League',
        'season': 2026,
        'team_count': 10,
        'draft_format': 'snake',
        'draft_slot': 5,
        'scoring_label': 'Half PPR',
        'scoring_items': {
            '3': 0.04,
            '4': 4.0,
            '24': 0.1,
            '25': 6.0,
            '42': 0.1,
            '43': 6.0,
            '53': 0.5,
            '80': 1.0,
            '85': 1.0,
        },
        'scoring_overrides': {},
        'bonuses': [],
        'roster_slots': {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1,
            'DST': 1,
            'K': 1,
        },
        'bench_slots': 7,
        'ir_slots': 1,
        'flex_eligible': ['RB', 'WR', 'TE'],
        'waiver_type': 'unknown',
        'waiver_constraints': [],
        'settings_source': 'test fixture',
        'settings_verified_at': '2026-08-22T12:00:00Z',
    }
    data.update(overrides)
    return LeagueProfile.from_mapping(data)


def _players() -> pd.DataFrame:
    rows = []
    counts = {'QB': 30, 'RB': 80, 'WR': 100, 'TE': 40, 'DST': 20, 'K': 20}
    stat_ids = {
        'QB': {'3': 4000, '4': 30},
        'RB': {'24': 1000, '25': 10, '53': 40},
        'WR': {'42': 1000, '43': 8, '53': 80},
        'TE': {'42': 700, '43': 6, '53': 70},
        'DST': {'80': 10},
        'K': {'85': 10},
    }
    adp_sequences = {
        'QB': lambda index: 20.0 + index * 10.0,
        'RB': lambda index: 1.0 + index * 2.5,
        'WR': lambda index: 2.0 + index * 2.5,
        'TE': lambda index: 45.0 + index * 8.0,
        'DST': lambda index: 130.0 + index * 5.0,
        'K': lambda index: 140.0 + index * 5.0,
    }
    for position, count in counts.items():
        for index in range(count):
            factor = 1.0 - index / (count * 1.5)
            rows.append(
                {
                    'name': f'{position} {index + 1}',
                    'position': position,
                    'projection_stats': {
                        key: value * factor for key, value in stat_ids[position].items()
                    },
                    'adp': adp_sequences[position](index),
                    'adp_updated_at': pd.Timestamp('2026-08-22', tz='UTC'),
                    'projection_season': 2026,
                    'eligibility_status': 'current',
                }
            )
    return pd.DataFrame(rows)


def test_unresolved_local_league_settings_fail_closed() -> None:
    profile = _profile(draft_slot=None)
    assert profile.draft_slot is None
    profile.validate_runtime_slot(1)
    with pytest.raises(LeagueProfileError, match='draft_slot'):
        profile.validate_runtime_slot(0)
    with pytest.raises(LeagueProfileError, match='draft_slot'):
        profile.validate_runtime_slot(profile.team_count + 1)
    with pytest.raises(LeagueProfileError, match='scoring_items'):
        _profile(scoring_items={})
    with pytest.raises(LeagueProfileError, match='waiver_constraints'):
        _profile(waiver_constraints=None)


@pytest.mark.parametrize(
    ('team_count', 'draft_slot', 'current_pick', 'expected'),
    [(10, 10, 10, 11), (10, 10, 11, 30), (12, 12, 12, 13), (12, 12, 13, 36), (8, 8, 8, 9)],
)
def test_snake_pick_boundaries_keep_back_to_back_turns(
    team_count: int, draft_slot: int, current_pick: int, expected: int
) -> None:
    profile = _profile(team_count=team_count, draft_slot=draft_slot)

    assert next_snake_pick(current_pick, profile) == expected


def test_scoring_configuration_changes_player_values() -> None:
    players = _players()
    standard = _profile(
        scoring_label='Standard', scoring_items={**_profile().scoring_items, '53': 0.0}
    )
    ppr = _profile(
        scoring_label='PPR', scoring_items={**_profile().scoring_items, '53': 1.0}
    )

    standard_scored = score_projections(players, standard).set_index('name')
    ppr_scored = score_projections(players, ppr).set_index('name')

    assert (
        ppr_scored.loc['WR 1', 'projected_points']
        > standard_scored.loc['WR 1', 'projected_points']
    )
    assert (
        ppr_scored.loc['QB 1', 'projected_points']
        == standard_scored.loc['QB 1', 'projected_points']
    )


def test_position_scoring_overrides_are_applied() -> None:
    players = _players()
    base = score_projections(players, _profile()).set_index('name')
    overridden = score_projections(
        players, _profile(scoring_overrides={'DST': {'80': 5.0}})
    ).set_index('name')

    assert overridden.loc['DST 1', 'projected_points'] == 50.0
    assert (
        overridden.loc['DST 1', 'projected_points']
        > base.loc['DST 1', 'projected_points']
    )
    assert (
        overridden.loc['QB 1', 'projected_points']
        == base.loc['QB 1', 'projected_points']
    )


def test_weekly_bonus_configuration_changes_projection() -> None:
    players = _players()
    players['weekly_projection_stats'] = [
        [stats, stats] for stats in players['projection_stats']
    ]
    no_bonus = _profile()
    passing_bonus = _profile(
        bonuses=[{'stat_id': '3', 'threshold': 300.0, 'points': 3.0, 'scope': 'weekly'}]
    )

    base = score_projections(players, no_bonus).set_index('name')
    bonus = score_projections(players, passing_bonus).set_index('name')

    assert (
        bonus.loc['QB 1', 'projected_points']
        == base.loc['QB 1', 'projected_points'] + 6.0
    )
    assert bonus.loc['WR 1', 'projected_points'] == base.loc['WR 1', 'projected_points']


def test_flex_and_league_size_change_replacement_and_vor() -> None:
    players = _players()
    one_flex = _profile()
    two_flex = _profile(
        roster_slots={**one_flex.roster_slots, 'FLEX': 2}, bench_slots=6
    )
    twelve_team = _profile(team_count=12, draft_slot=6)

    board_one = build_draft_board(players, one_flex)
    board_two = build_draft_board(players, two_flex)
    board_twelve = build_draft_board(players, twelve_team)

    replacement_one = calculate_replacement_levels(
        score_projections(players, one_flex), one_flex
    )
    replacement_two = calculate_replacement_levels(
        score_projections(players, two_flex), two_flex
    )
    replacement_twelve = calculate_replacement_levels(
        score_projections(players, twelve_team), twelve_team
    )

    assert replacement_two['starter_demand'] != replacement_one['starter_demand']
    assert replacement_twelve['levels'] != replacement_one['levels']
    assert not board_two['scarcity'].equals(board_one['scarcity'])
    assert not board_two['value_signal'].equals(board_one['value_signal'])
    assert not board_twelve['vor'].equals(board_one['vor'])
    assert sum(replacement_one['bench_allocation'].values()) == (
        one_flex.bench_slots * one_flex.team_count
    )


def test_draft_slot_changes_next_pick_recommendation() -> None:
    players = _players()
    slot_one = build_draft_board(players, _profile(draft_slot=1), current_pick=1)
    slot_ten = build_draft_board(players, _profile(draft_slot=10), current_pick=10)

    assert slot_one.attrs['next_pick'] != slot_ten.attrs['next_pick']
    assert not slot_one['availability_next_pick'].equals(
        slot_ten['availability_next_pick']
    )
    assert (
        slot_one['recommendation'].value_counts().to_dict()
        != slot_ten['recommendation'].value_counts().to_dict()
    )


def test_slot_neutral_board_has_no_guessed_timing_recommendation() -> None:
    board = build_draft_board(_players(), _profile(draft_slot=None))

    assert board.attrs['next_pick'] is None
    assert board['availability_next_pick'].isna().all()
    assert set(board['recommendation']) == {'slot_required'}


def test_runtime_draft_state_uses_stable_ids_and_is_actionable() -> None:
    players = _players().copy()
    players['espn_id'] = range(1000, 1000 + len(players))
    first_id = int(players.iloc[0]['espn_id'])
    second_id = int(players.iloc[1]['espn_id'])

    board = build_draft_board(
        players,
        _profile(draft_slot=3),
        current_pick=3,
        taken_ids=[first_id],
        your_ids=[second_id],
    )

    statuses = board.set_index('espn_id')['roster_status']
    assert statuses.loc[first_id] == 'taken'
    assert statuses.loc[second_id] == 'mine'
    assert not bool(board.set_index('espn_id').loc[first_id, 'is_available'])
    assert not bool(board.set_index('espn_id').loc[second_id, 'is_available'])
    assert board.set_index('espn_id').loc[first_id, 'recommendation'] == 'taken'
    assert board.set_index('espn_id').loc[second_id, 'recommendation'] == 'mine'

    with pytest.raises(SemanticInputError, match='unknown player'):
        build_draft_board(
            players,
            _profile(draft_slot=3),
            current_pick=3,
            taken_ids=[999999],
        )


def test_missing_adp_is_rejected_instead_of_assigned_fifty_percent() -> None:
    players = _players()
    players.loc[0, 'adp'] = None

    with pytest.raises(SemanticInputError, match='missing ADP'):
        build_draft_board(players, _profile())


def test_zero_replacement_level_is_rejected() -> None:
    players = _players()
    players['projection_stats'] = players['projection_stats'].map(
        lambda _: {'999': 0.0}
    )

    with pytest.raises(SemanticInputError, match='no usable signal'):
        build_draft_board(players, _profile(scoring_items={'999': 1.0}))
