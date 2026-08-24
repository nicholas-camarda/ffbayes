from __future__ import annotations

import json

import pandas as pd
import pytest

from ffbayes.draft_2026 import dashboard_app
from ffbayes.draft_2026.dashboard_app import (
    DashboardRequestError,
    DashboardService,
)
from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.pipeline import FreshInputs, validate_output_provenance


def _profile(profile_id: str, flex: int) -> LeagueProfile:
    return LeagueProfile.from_mapping(
        {
            'profile_id': profile_id,
            'league_name': profile_id,
            'season': 2026,
            'team_count': 10,
            'draft_format': 'snake',
            'draft_slot': None,
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
                'FLEX': flex,
                'DST': 1,
                'K': 1,
            },
            'bench_slots': 6 if flex == 2 else 7,
            'ir_slots': 1,
            'flex_eligible': ['RB', 'WR', 'TE'],
            'waiver_type': 'unknown',
            'waiver_constraints': [],
            'settings_source': 'fixture',
            'settings_verified_at': '2026-08-23T00:00:00Z',
        }
    )


def _players() -> pd.DataFrame:
    counts = {'QB': 30, 'RB': 80, 'WR': 100, 'TE': 40, 'DST': 20, 'K': 20}
    stat_ids = {
        'QB': {'3': 4000, '4': 30},
        'RB': {'24': 1000, '25': 10, '53': 40},
        'WR': {'42': 1000, '43': 8, '53': 80},
        'TE': {'42': 700, '43': 6, '53': 70},
        'DST': {'80': 10},
        'K': {'85': 10},
    }
    rows = []
    espn_id = 1000
    for position, count in counts.items():
        for index in range(count):
            factor = 1.0 - index / (count * 1.5)
            rows.append(
                {
                    'espn_id': espn_id,
                    'name': f'{position} {index + 1}',
                    'position': position,
                    'projection_stats': {
                        key: value * factor for key, value in stat_ids[position].items()
                    },
                    'adp': float(index + 1 + (0 if position in {'RB', 'WR'} else 40)),
                    'adp_updated_at': pd.Timestamp('2026-08-23', tz='UTC'),
                    'projection_season': 2026,
                    'eligibility_status': 'current',
                }
            )
            espn_id += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def service(tmp_path):
    players = _players()
    inputs = FreshInputs(
        payload={'players': []},
        source_manifest={
            'season': 2026,
            'sha256': 'espn-fixture',
            'fetched_at': '2026-08-23T00:00:00Z',
        },
        roster=pd.DataFrame(),
        roster_manifest={
            'season': 2026,
            'sha256': 'roster-fixture',
            'fetched_at': '2026-08-23T00:00:00Z',
        },
        players=players,
        coverage={'status': 'passed', 'rows': len(players)},
    )
    return DashboardService(
        inputs,
        [_profile('bill', 2), _profile('family', 1)],
        project_root=tmp_path,
        run_root=tmp_path / 'run',
        code_revision='fixture-revision',
    )


def test_service_lists_two_named_leagues_and_ready_status(service) -> None:
    assert service.status_payload()['status'] == 'ready'
    leagues = service.leagues_payload()['leagues']
    assert [league['profile_id'] for league in leagues] == ['bill', 'family']
    assert leagues[0]['roster_slots']['FLEX'] == 2
    assert leagues[1]['roster_slots']['FLEX'] == 1


def test_service_recalculates_board_and_isolates_league_state(service) -> None:
    bill = service.handle_board(
        {
            'profile_id': 'bill',
            'draft_slot': 2,
            'current_pick': 2,
            'taken_ids': [1000],
            'your_ids': [],
            'queue_ids': [1001],
        }
    )
    family = service.handle_board(
        {
            'profile_id': 'family',
            'draft_slot': 9,
            'current_pick': 9,
            'taken_ids': [],
            'your_ids': [1000],
            'queue_ids': [],
        }
    )

    assert bill['league_profile']['profile_id'] == 'bill'
    assert bill['next_pick'] != family['next_pick']
    bill_row = next(row for row in bill['decision_table'] if row['espn_id'] == 1000)
    family_row = next(row for row in family['decision_table'] if row['espn_id'] == 1000)
    assert bill_row['recommendation'] == 'taken'
    assert family_row['recommendation'] == 'mine'


def test_service_rejects_invalid_slots_and_unknown_players(service) -> None:
    with pytest.raises(DashboardRequestError, match='draft_slot'):
        service.handle_board({'profile_id': 'bill', 'draft_slot': 0, 'current_pick': 1})
    with pytest.raises(DashboardRequestError, match='unknown player'):
        service.handle_board(
            {
                'profile_id': 'bill',
                'draft_slot': 1,
                'current_pick': 1,
                'taken_ids': [999999],
            }
        )


def test_service_snapshot_is_provenance_bound_and_atomic(service, tmp_path) -> None:
    snapshot = service.write_snapshot(
        {
            'profile_id': 'bill',
            'draft_slot': 1,
            'current_pick': 1,
            'taken_ids': [],
            'your_ids': [],
            'queue_ids': [1000],
        }
    )

    assert snapshot.parent == tmp_path / 'run' / 'snapshots'
    payload = json.loads(snapshot.read_text(encoding='utf-8'))
    validate_output_provenance(payload)
    assert payload['runtime_state']['queue_ids'] == [1000]
    assert not list(snapshot.parent.glob('*.tmp'))


def test_blocked_status_preserves_external_source_failure_details(tmp_path) -> None:
    blocked = DashboardService(
        None,
        [_profile('bill', 2)],
        project_root=tmp_path,
        run_root=tmp_path / 'run',
        code_revision='fixture-revision',
        blocked_error='HTTP 503 from ESPN',
        blocked_details={
            'source': 'ESPN public current-season fantasy player feed',
            'failure_mode': 'HTTPError: 503',
            'transience': 'possibly transient',
            'pipeline_dependencies': ['ADP'],
            'alternate_source': 'none used; silent fallback is prohibited',
        },
    )

    status = blocked.status_payload()
    assert status['status'] == 'blocked'
    assert status['source'].startswith('ESPN public')
    assert status['failure_mode'] == 'HTTPError: 503'
    assert status['fallback'] is False


def test_main_classifies_incompatible_profile_scoring_as_local_configuration(
    monkeypatch, tmp_path
) -> None:
    players = _players()
    players.loc[players['position'].eq('DST'), 'projection_stats'] = [
        {'89': 10.0}
    ] * int(players['position'].eq('DST').sum())
    inputs = FreshInputs(
        payload={'players': []},
        source_manifest={
            'season': 2026,
            'sha256': 'espn-fixture',
            'fetched_at': '2026-08-23T00:00:00Z',
        },
        roster=pd.DataFrame(),
        roster_manifest={
            'season': 2026,
            'sha256': 'roster-fixture',
            'fetched_at': '2026-08-23T00:00:00Z',
        },
        players=players,
        coverage={'status': 'passed', 'rows': len(players)},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(dashboard_app, 'default_profile_paths', lambda: ())
    monkeypatch.setattr(
        dashboard_app, '_load_profiles', lambda _: ([_profile('bill', 2)], None)
    )
    monkeypatch.setattr(dashboard_app, 'load_fresh_inputs', lambda _: inputs)
    monkeypatch.setattr(
        dashboard_app,
        'serve_dashboard',
        lambda service, **_: captured.setdefault('status', service.status_payload())
        and 0,
    )

    assert dashboard_app.main(['--year', '2026', '--no-browser']) == 0
    status = captured['status']
    assert isinstance(status, dict)
    assert status['status'] == 'blocked'
    assert status['fallback'] is False
    assert 'Projection scoring has no usable signal for DST' in status['error']
    assert status['source'] == 'local league profile scoring configuration'
    assert status['external_dependency'] is False
