from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ffbayes.draft_2026.decision.lineup import optimize_lineup
from ffbayes.draft_2026.decision.roster_value import calculate_marginal_roster_value
from ffbayes.draft_2026.engine import build_draft_board
from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.pipeline import _sha256_json
from ffbayes.draft_2026.rules import (
    RosterRules,
    SlotRule,
    league_can_complete,
    roster_is_feasible,
    validate_complete_roster,
)
from ffbayes.evaluation.baselines import (
    EvaluationInputError,
    evaluate_baselines,
    main,
    simulate_snake_draft,
)
from ffbayes.evaluation.rolling import TemporalLeakageError, validate_as_of_inputs

ROOT = Path(__file__).resolve().parents[1]


def profile(**updates):
    data = json.loads((ROOT / 'config/leagues/example_2026.json').read_text())
    data.update(
        team_count=2,
        draft_slot=1,
        bench_slots=1,
        roster_slots=dict(QB=1, RB=1, WR=1, TE=0, FLEX=0, DST=0, K=0),
        scoring_items={'x': 1.0},
        scoring_overrides={},
        bonuses=[],
    )
    data.update(updates)
    return LeagueProfile.from_mapping(data)


def players():
    rows = []
    for pos, base in [
        ('QB', 300),
        ('RB', 220),
        ('WR', 210),
        ('TE', 180),
        ('K', 150),
        ('DST', 140),
    ]:
        for rank in range(10):
            points = float(base - rank * 10)
            rows.append(
                dict(
                    espn_id=len(rows) + 1,
                    name=f'{pos}{rank}',
                    position=pos,
                    projected_points=points,
                    projection_stats={'x': points},
                    adp=float(len(rows) + 1),
                    projection_season=2026,
                )
            )
    return pd.DataFrame(rows)


def test_legacy_example_digest_is_unchanged():
    p = LeagueProfile.from_mapping(
        json.loads((ROOT / 'config/leagues/example_2026.json').read_text())
    )
    assert (
        _sha256_json(p.to_dict())
        == 'ea71a6f326fa525c2d9a706ce4cc156c9c3f1d4a56359a766f5741e98c198371'
    )
    assert 'slot_eligibility' not in p.to_dict()


def test_explicit_slot_contract_roundtrips_and_changes_digest():
    p = profile()
    before = p.to_dict()
    explicit = {name: list(pos) for name, _, pos in p.slot_groups()}
    upgraded = profile(slot_eligibility=explicit)
    assert _sha256_json(upgraded.to_dict()) != _sha256_json(before)
    assert (
        LeagueProfile.from_mapping(upgraded.to_dict()).to_dict() == upgraded.to_dict()
    )


def test_exact_assignment_with_overlapping_slots():
    rules = RosterRules(
        2,
        (SlotRule('FLEX', 1, ('RB', 'WR')), SlotRule('SF', 1, ('QB', 'RB', 'WR'))),
        0,
        0,
    )
    frame = pd.DataFrame(
        [
            dict(espn_id=1, position='RB', projected_points=100.0),
            dict(espn_id=2, position='QB', projected_points=90.0),
            dict(espn_id=3, position='WR', projected_points=80.0),
        ]
    )
    result = optimize_lineup(frame, rules)
    assert result.total_points == 190
    assert len(set(result.used_player_ids)) == result.filled_slots == 2


def test_feasibility_counts_remaining_picks_and_shared_supply():
    rules = RosterRules(
        3, (SlotRule('QB', 1, ('QB',)), SlotRule('RB', 1, ('RB',))), 1, 0
    )
    assert not roster_is_feasible(['QB', 'QB', 'QB'], ['RB'] * 8, rules)
    assert not league_can_complete(
        [['RB'], ['RB'], ['QB', 'QB']], ['QB', 'RB', 'RB', 'RB'], rules
    )
    assert league_can_complete(
        [['RB'], ['RB'], ['QB']], ['QB', 'QB', 'RB', 'RB', 'RB', 'RB'], rules
    )


def test_marginal_utility_preserves_upgrade_and_telescopes():
    rules = RosterRules(2, (SlotRule('QB', 1, ('QB',)),), 2, 0)
    frame = pd.DataFrame(
        [
            dict(espn_id=i + 1, position='QB', projected_points=float(v))
            for i, v in enumerate([250, 300, 220, 200])
        ]
    )

    def values(mine):
        return calculate_marginal_roster_value(
            frame, rules, your_ids=mine, replacement_levels={'QB': 200.0}
        )

    assert values([]).loc[2, 'starter_slot_gain'] == 100
    assert values([1]).loc[2, 'starter_slot_gain'] == 50
    assert values([2]).loc[1, 'starter_slot_gain'] == 0
    assert values([2]).loc[1, 'bench_insurance_value'] == 2.5
    assert (
        values([]).loc[1, 'marginal_usable_points']
        + values([1]).loc[2, 'marginal_usable_points']
        == values([]).loc[2, 'marginal_usable_points']
        + values([2]).loc[1, 'marginal_usable_points']
    )


def test_roster_value_decreases_for_filled_qb_but_not_two_qb():
    frame = players()
    one = profile()
    two = profile(roster_slots=dict(QB=2, RB=1, WR=1, TE=0, FLEX=0, K=0, DST=0))
    empty = build_draft_board(frame, one).set_index('espn_id')
    owned = build_draft_board(frame, one, your_ids=[1]).set_index('espn_id')
    dual = build_draft_board(frame, two, your_ids=[1]).set_index('espn_id')
    assert (
        owned.loc[2, 'marginal_usable_points'] < empty.loc[2, 'marginal_usable_points']
    )
    assert (
        dual.loc[2, 'marginal_usable_points'] > owned.loc[2, 'marginal_usable_points']
    )
    assert owned.loc[2, 'board_rank'] > empty.loc[2, 'board_rank']
    assert not empty.loc[empty.position.isin(['K', 'DST']), 'is_draftable'].any()


def test_full_roster_has_no_draftable_candidate():
    board = build_draft_board(players(), profile(), your_ids=[1, 11, 21, 12])
    assert not board.is_draftable.any()


def test_flex_and_superflex_are_distinct_configuration_groups():
    p = profile(
        roster_slots=dict(QB=1, RB=1, WR=1, TE=0, FLEX=1, SUPERFLEX=1, K=0, DST=0),
        slot_eligibility={
            'QB': ['QB'],
            'RB': ['RB'],
            'WR': ['WR'],
            'FLEX': ['RB', 'WR', 'TE'],
            'SUPERFLEX': ['QB', 'RB', 'WR', 'TE'],
        },
    )
    board = build_draft_board(players(), p, your_ids=[1, 11, 21])
    assert board.is_draftable.any()
    assert 'SUPERFLEX' in dict(
        (name, eligible) for name, _, eligible in p.slot_groups()
    )


def test_all_baselines_use_same_fixed_opponents_and_legal_rosters():
    result = evaluate_baselines(players(), profile(), synthetic=True, seed=4)
    assert result.manifest['temporal_status'] == 'synthetic_unverified'
    assert result.manifest['metric_scope'] == 'projected_utility_only'
    for sim in result.simulations:
        all_ids = [p.player_id for p in sim.picks]
        assert len(all_ids) == len(set(all_ids)) == profile().total_draft_picks()
        assert all(p.policy == 'adp' for p in sim.picks if p.team_index == 1)
        for _, ids in sim.rosters:
            validate_complete_roster(
                players().set_index('espn_id').loc[list(ids), 'position'], profile()
            )
    assert (
        result.to_dict()
        == evaluate_baselines(players(), profile(), synthetic=True, seed=4).to_dict()
    )


def test_shared_wrapper_reserves_scarce_qb():
    p = profile(roster_slots=dict(QB=1, RB=1, WR=0, TE=0, FLEX=0, K=0, DST=0))
    frame = pd.DataFrame(
        [
            dict(
                espn_id=i + 1,
                position=pos,
                projected_points=float(100 - i),
                adp=float(i + 1),
            )
            for i, pos in enumerate(['RB', 'QB', 'QB', 'RB', 'RB', 'RB'])
        ]
    )
    sim = simulate_snake_draft(frame, p, policy='adp')
    for _, ids in sim.rosters:
        validate_complete_roster(
            frame.set_index('espn_id').loc[list(ids), 'position'], p
        )


@pytest.mark.parametrize('bad', [True, 1.5, float('inf'), None])
def test_baseline_rejects_invalid_ids(bad):
    frame = players()
    frame['espn_id'] = frame.espn_id.astype(object)
    frame.at[0, 'espn_id'] = bad
    with pytest.raises(EvaluationInputError):
        simulate_snake_draft(frame, profile())


def test_baseline_rejects_scoring_mismatch():
    frame = players()
    frame.loc[0, 'projected_points'] += 100
    with pytest.raises(EvaluationInputError, match='disagree'):
        simulate_snake_draft(frame, profile())


def test_temporal_validation_requires_provenance_and_rejects_future_features():
    with pytest.raises(EvaluationInputError, match='cutoff'):
        evaluate_baselines(players(), profile())
    with pytest.raises(TemporalLeakageError, match='projection_as_of'):
        validate_as_of_inputs([{}], '2026-08-22T00:00:00Z')
    with pytest.raises(TemporalLeakageError, match='after'):
        validate_as_of_inputs(
            [
                {
                    'projection_as_of': '2026-08-23T00:00:00Z',
                    'adp_as_of': '2026-08-20T00:00:00Z',
                }
            ],
            '2026-08-22T00:00:00Z',
        )


def test_benchmark_command_writes_reproducible_synthetic_report(tmp_path):
    source = tmp_path / 'input.json'
    output = tmp_path / 'result.json'
    source.write_text(
        json.dumps(
            {
                'profile': profile().to_dict(),
                'players': players().to_dict(orient='records'),
            }
        )
    )
    assert (
        main(
            [
                '--input',
                str(source),
                '--output',
                str(output),
                '--synthetic',
                '--policy',
                'adp',
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result['manifest']['temporal_status'] == 'synthetic_unverified'
    assert result['manifest']['code_sha256']


def test_pre_modeling_checkpoint_restores_saved_selections(tmp_path):
    from test_draft_2026_dashboard_service import _players

    from ffbayes.draft_2026.dashboard_app import DashboardService
    from ffbayes.draft_2026.pipeline import FreshInputs

    p = LeagueProfile.from_mapping(
        json.loads((ROOT / 'config/leagues/example_2026.json').read_text())
    )
    frame = _players()
    # Use positive stat projections under the unchanged public scoring profile;
    # the service fixture's simplified DST stats use different scoring IDs.
    stats = []
    for index, row in frame.iterrows():
        weights = {**p.scoring_items, **p.scoring_overrides.get(row.position, {})}
        stat, weight = next((key, value) for key, value in weights.items() if value > 0)
        stats.append({stat: (1000.0 - index) / weight})
    frame['projection_stats'] = stats
    ids = frame.espn_id.head(2).astype(int).tolist()
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    # This digest was captured from the code before optional slot rules existed.
    # Do not generate it with the new profile serializer in this test.
    legacy = {
        'schema_version': 'draft_2026_checkpoint_v1',
        'season': 2026,
        'profile_id': p.profile_id,
        'profile_sha256': 'ea71a6f326fa525c2d9a706ce4cc156c9c3f1d4a56359a766f5741e98c198371',
        'runtime_state': {
            'draft_slot': 2,
            'current_pick': 2,
            'actions': [{'pick': 1, 'player_id': ids[0], 'disposition': 'mine'}],
            'your_ids': [ids[0]],
            'taken_ids': [],
            'queue_ids': [ids[1]],
        },
    }
    (checkpoints / f'2026-{p.profile_id}.json').write_text(json.dumps(legacy))
    manifest = {
        'season': 2026,
        'sha256': 'fixture',
        'fetched_at': '2026-08-23T00:00:00Z',
    }
    inputs = FreshInputs(
        payload={},
        source_manifest=manifest,
        roster=pd.DataFrame(),
        roster_manifest=manifest,
        players=frame,
        coverage={'status': 'passed'},
    )
    service = DashboardService(
        inputs,
        [p],
        project_root=tmp_path,
        run_root=tmp_path / 'run',
        checkpoint_root=checkpoints,
        code_revision='fixture',
    )
    payload = service.handle_board({'profile_id': p.profile_id})
    assert payload['runtime_state']['your_ids'] == [ids[0]]
    assert payload['runtime_state']['queue_ids'] == [ids[1]]
    assert payload['runtime_state']['draft_slot'] == 2
    assert payload['current_pick'] == 2
    assert (
        service.status_payload()['persistence'][p.profile_id]['status'] == 'recovered'
    )


def test_asof_report_validates_declared_provenance_without_inventing_it():
    frame = players()
    frame['projection_as_of'] = '2026-08-20T00:00:00Z'
    frame['adp_as_of'] = '2026-08-20T00:00:00Z'
    source = dict(
        source='test_snapshot',
        season=2026,
        as_of='2026-08-20T00:00:00Z',
        published_at='2026-08-20T00:00:00Z',
        fetched_at='2026-08-20T00:00:00Z',
        sha256='a' * 64,
        schema_version='fixture_v1',
        row_count=len(frame),
    )
    result = evaluate_baselines(
        frame,
        profile(),
        policies=['adp'],
        cutoff='2026-08-22T00:00:00Z',
        source_manifests=[source],
    )
    assert result.manifest['temporal_status'] == 'declared_timestamps_validated'
    source['season'] = 2025
    with pytest.raises(TemporalLeakageError, match='season'):
        evaluate_baselines(
            frame,
            profile(),
            policies=['adp'],
            cutoff='2026-08-22T00:00:00Z',
            source_manifests=[source],
        )


def test_count_based_solver_matches_exhaustive_overlapping_assignments():
    import itertools
    import random

    rng = random.Random(71)
    rules = RosterRules(
        2,
        (
            SlotRule('QB', 1, ('QB',)),
            SlotRule('FLEX', 1, ('RB', 'WR')),
            SlotRule('SF', 1, ('QB', 'RB', 'WR')),
        ),
        1,
        0,
    )
    slots = rules.expanded_slots()
    for _ in range(25):
        rows = [
            dict(
                espn_id=i + 1,
                position=rng.choice(['QB', 'RB', 'WR']),
                projected_points=float(rng.randrange(100)),
            )
            for i in range(6)
        ]
        expected = (0, 0.0)
        for choice in itertools.product(range(-1, 6), repeat=len(slots)):
            used = [i for i in choice if i >= 0]
            if len(used) != len(set(used)) or any(
                i >= 0 and rows[i]['position'] not in slots[s][1]
                for s, i in enumerate(choice)
            ):
                continue
            expected = max(
                expected, (len(used), sum(rows[i]['projected_points'] for i in used))
            )
        actual = optimize_lineup(pd.DataFrame(rows), rules)
        assert (actual.filled_slots, actual.total_points) == expected


def test_unallocated_flex_position_uses_shared_replacement_boundary():
    p = profile(
        roster_slots=dict(QB=0, RB=0, WR=0, TE=0, FLEX=0, SUPERFLEX=1, K=0, DST=0),
        slot_eligibility={'SUPERFLEX': ['QB', 'RB', 'WR', 'TE']},
        bench_slots=0,
    )
    board = build_draft_board(players(), p)
    wr = board[board.position.eq('WR')]
    assert (wr.replacement_level > 0).all()
    assert not wr.is_draftable.empty
    assert board[board.is_draftable].iloc[0].position == 'QB'


def test_signed_espn_defenses_pass_model_and_baseline():
    frame = players()
    dst = frame.position.eq('DST')
    frame.loc[dst, 'espn_id'] = [-16001 - i for i in range(int(dst.sum()))]
    p = profile(roster_slots=dict(QB=1, RB=1, WR=1, TE=0, FLEX=0, K=0, DST=1))
    board = build_draft_board(frame, p, your_ids=[-16001])
    assert board.loc[board.espn_id.eq(-16001), 'roster_status'].iloc[0] == 'mine'
    result = evaluate_baselines(frame, p, synthetic=True)
    assert result.simulations
    for simulation in result.simulations:
        assert any(pick.player_id < 0 for pick in simulation.picks)
