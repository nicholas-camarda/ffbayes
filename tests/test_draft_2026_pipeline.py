from __future__ import annotations

import json

import pandas as pd
import pytest

import ffbayes.draft_2026.pipeline as current_pipeline
from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.pipeline import (
    FreshInputs,
    OutputProvenanceError,
    build_dashboard_payload,
    render_dashboard_html,
    validate_output_provenance,
)


def _profile() -> LeagueProfile:
    return LeagueProfile.from_mapping(
        {
            'profile_id': 'league-a',
            'league_name': 'League A',
            'season': 2026,
            'team_count': 10,
            'draft_format': 'snake',
            'draft_slot': 4,
            'scoring_label': 'PPR',
            'scoring_items': {'53': 1.0},
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
            'waiver_type': 'FAAB',
            'waiver_constraints': ['weekly waivers'],
            'settings_source': 'fixture',
            'settings_verified_at': '2026-08-22T12:00:00Z',
        }
    )


def test_dashboard_payload_binds_sources_profile_code_and_board() -> None:
    board = pd.DataFrame(
        {
            'board_rank': [1],
            'espn_id': [101],
            'name': ['Player One'],
            'position': ['WR'],
            'projected_points': [250.0],
            'vor': [100.0],
            'adp': [5.0],
            'availability_next_pick': [0.1],
            'recommendation': ['draft_now'],
        }
    )
    board.attrs['replacement'] = {'levels': {'WR': 150.0}, 'demand': {'WR': 30}}
    board.attrs['current_pick'] = 4
    board.attrs['next_pick'] = 17
    source_manifest = {
        'season': 2026,
        'sha256': 'abc',
        'fetched_at': '2026-08-22T12:00:00Z',
    }

    payload = build_dashboard_payload(
        board,
        _profile(),
        source_manifest=source_manifest,
        roster_manifest={
            'season': 2026,
            'sha256': 'def',
            'fetched_at': '2026-08-22T12:01:00Z',
        },
        coverage_report={'status': 'passed'},
        code_revision='deadbeef',
    )

    validate_output_provenance(payload)
    assert payload['league_profile']['profile_id'] == 'league-a'
    assert payload['provenance']['code_revision'] == 'deadbeef'
    assert payload['decision_table'][0]['name'] == 'Player One'


def test_dashboard_payload_rejects_mismatched_or_stale_provenance() -> None:
    payload = {
        'schema_version': 'draft_2026_v1',
        'season': 2026,
        'generated_at': '2026-08-22T12:10:00+00:00',
        'league_profile': {'season': 2026, 'profile_id': 'x'},
        'provenance': {
            'code_revision': 'abc',
            'profile_sha256': 'bad',
            'source_manifests': [
                {'season': 2025, 'sha256': 'old', 'fetched_at': '2026-06-01T00:00:00Z'}
            ],
        },
        'coverage_report': {'status': 'passed'},
        'decision_table': [],
    }

    with pytest.raises(OutputProvenanceError, match='season'):
        validate_output_provenance(payload)


def test_dashboard_renderer_embeds_parseable_validated_payload() -> None:
    board = pd.DataFrame(
        {
            'board_rank': [1],
            'espn_id': [101],
            'name': ['Player One'],
            'position': ['WR'],
            'projected_points': [250.0],
            'replacement_level': [150.0],
            'vor': [100.0],
            'scarcity': [5.0],
            'adp': [5.0],
            'availability_next_pick': [0.1],
            'recommendation': ['draft_now'],
        }
    )
    board.attrs['replacement'] = {'levels': {'WR': 150.0}, 'demand': {'WR': 30}}
    board.attrs['current_pick'] = 4
    board.attrs['next_pick'] = 17
    manifest = {'season': 2026, 'sha256': 'abc', 'fetched_at': '2026-08-22T12:00:00Z'}
    payload = build_dashboard_payload(
        board,
        _profile(),
        source_manifest=manifest,
        roster_manifest={**manifest, 'sha256': 'def'},
        coverage_report={'status': 'passed'},
        code_revision='deadbeef',
    )

    rendered = render_dashboard_html(payload)
    embedded = rendered.split(
        '<script id="draft-2026-payload" type="application/json">', 1
    )[1].split('</script>', 1)[0]

    assert json.loads(embedded) == json.loads(json.dumps(payload))
    assert '<table id="draft-board">' in rendered
    assert '&quot;schema_version&quot;' not in embedded


def _payload_board() -> pd.DataFrame:
    board = pd.DataFrame(
        {
            'board_rank': [1],
            'espn_id': [101],
            'name': ['Player One'],
            'position': ['WR'],
            'projected_points': [250.0],
            'replacement_level': [150.0],
            'vor': [100.0],
            'scarcity': [5.0],
            'adp': [5.0],
            'availability_next_pick': [0.1],
            'recommendation': ['draft_now'],
            'roster_status': ['available'],
            'is_available': [True],
        }
    )
    board.attrs.update(
        {
            'replacement': {'levels': {'WR': 150.0}, 'demand': {'WR': 30}},
            'current_pick': 4,
            'next_pick': 17,
            'runtime_state': {
                'draft_slot': 4,
                'current_pick': 4,
                'taken_ids': [202],
                'your_ids': [303],
                'queue_ids': [101],
            },
        }
    )
    return board


def test_dashboard_payload_contains_runtime_state_and_stable_player_ids() -> None:
    manifest = {'season': 2026, 'sha256': 'abc', 'fetched_at': '2026-08-22T12:00:00Z'}
    payload = build_dashboard_payload(
        _payload_board(),
        _profile(),
        source_manifest=manifest,
        roster_manifest={**manifest, 'sha256': 'def'},
        coverage_report={'status': 'passed'},
        code_revision='deadbeef',
    )

    assert payload['decision_table'][0]['espn_id'] == 101
    assert payload['runtime_state']['taken_ids'] == [202]
    assert payload['provenance']['state_sha256']
    validate_output_provenance(payload)


def test_output_provenance_rejects_tampered_runtime_state() -> None:
    manifest = {'season': 2026, 'sha256': 'abc', 'fetched_at': '2026-08-22T12:00:00Z'}
    payload = build_dashboard_payload(
        _payload_board(),
        _profile(),
        source_manifest=manifest,
        roster_manifest={**manifest, 'sha256': 'def'},
        coverage_report={'status': 'passed'},
        code_revision='deadbeef',
    )
    payload['runtime_state']['current_pick'] = 5

    with pytest.raises(OutputProvenanceError, match='state digest'):
        validate_output_provenance(payload)


def test_slot_neutral_static_renderer_shows_explicit_requirement() -> None:
    board = _payload_board()
    board.attrs['current_pick'] = None
    board.attrs['next_pick'] = None
    board.attrs['runtime_state']['draft_slot'] = None
    board['availability_next_pick'] = [float('nan')]
    board['recommendation'] = ['slot_required']
    manifest = {'season': 2026, 'sha256': 'abc', 'fetched_at': '2026-08-22T12:00:00Z'}
    payload = build_dashboard_payload(
        board,
        LeagueProfile.from_mapping({**_profile().to_dict(), 'draft_slot': None}),
        source_manifest=manifest,
        roster_manifest={**manifest, 'sha256': 'def'},
        coverage_report={'status': 'passed'},
        code_revision='deadbeef',
    )

    rendered = render_dashboard_html(payload)

    assert 'Draft slot required' in rendered


def test_checked_in_profiles_have_explicit_stable_settings_and_runtime_slot() -> None:
    path = __import__('pathlib').Path('config/leagues') / 'example_2026.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    profile = LeagueProfile.from_mapping(data)
    assert profile.profile_id == 'example-2026'
    assert profile.league_name == 'Example League'
    assert profile.team_count == 12
    assert profile.draft_format == 'snake'
    assert profile.scoring_label == 'Half PPR'
    assert profile.draft_slot is None
    assert profile.flex_eligible == ('RB', 'WR', 'TE')


def test_default_profile_discovery_prefers_ignored_local_profiles(tmp_path, monkeypatch) -> None:
    from ffbayes.draft_2026 import pipeline

    example = tmp_path / 'example_2026.json'
    local_one = tmp_path / 'alpha.local.json'
    local_two = tmp_path / 'beta.local.json'
    example.write_text('{}', encoding='utf-8')
    local_one.write_text('{}', encoding='utf-8')
    local_two.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(pipeline, 'PROFILE_ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'EXAMPLE_PROFILE', example)

    assert pipeline.default_profile_paths() == (local_one, local_two)

    local_one.unlink()
    local_two.unlink()
    assert pipeline.default_profile_paths() == (example,)


def test_main_fetches_one_validated_snapshot_for_all_leagues(
    tmp_path, monkeypatch
) -> None:
    first = _profile().to_dict()
    second = {**first, 'profile_id': 'league-b', 'league_name': 'League B'}
    profile_paths = []
    for index, value in enumerate((first, second), start=1):
        path = tmp_path / f'profile-{index}.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        profile_paths.append(path)

    inputs = FreshInputs(
        payload={'players': []},
        source_manifest={'season': 2026},
        roster=pd.DataFrame(),
        roster_manifest={'season': 2026},
        players=pd.DataFrame({'name': ['fixture']}),
        coverage={'status': 'passed'},
    )
    fetch_calls = []
    preflight_calls = []
    run_calls = []

    def fake_load(season):
        fetch_calls.append(season)
        return inputs

    def fake_build(players, profile):
        preflight_calls.append((id(players), profile.profile_id))
        return pd.DataFrame()

    def fake_run(profile, output_dir, *, project_root, fresh_inputs):
        run_calls.append((profile.profile_id, id(fresh_inputs), output_dir))
        return {}

    monkeypatch.setattr(current_pipeline, 'load_fresh_inputs', fake_load)
    monkeypatch.setattr(current_pipeline, 'build_draft_board', fake_build)
    monkeypatch.setattr(current_pipeline, 'run_profile', fake_run)

    args = [
        '--profile',
        str(profile_paths[0]),
        '--profile',
        str(profile_paths[1]),
        '--output-root',
        str(tmp_path / 'outputs'),
    ]
    assert current_pipeline.main(args) == 0
    assert fetch_calls == [2026]
    assert [call[1] for call in preflight_calls] == ['league-a', 'league-b']
    assert {call[1] for call in run_calls} == {id(inputs)}


def test_main_reports_missing_local_configuration(tmp_path, capsys) -> None:
    incomplete = {
        'profile_id': 'incomplete',
        'league_name': 'Incomplete',
        'season': 2026,
        'team_count': None,
        'draft_format': None,
        'draft_slot': None,
        'scoring_label': None,
        'scoring_items': {},
        'bonuses': [],
        'roster_slots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1},
        'bench_slots': 6,
        'ir_slots': 1,
        'flex_eligible': ['RB', 'WR', 'TE'],
        'waiver_type': 'unknown',
        'waiver_constraints': [],
        'settings_source': 'fixture',
        'settings_verified_at': '2026-08-23T00:00:00Z',
    }
    profile_path = tmp_path / 'incomplete.json'
    profile_path.write_text(json.dumps(incomplete), encoding='utf-8')
    assert current_pipeline.main(
        ['--profile', str(profile_path), '--output-root', str(tmp_path / 'outputs')]
    ) == 2
    report = json.loads(capsys.readouterr().out)

    assert report['status'] == 'blocked'
    assert len(report['profiles']) == 1
    assert all(item['external_dependency'] is False for item in report['profiles'])
    assert all(
        item['failure_class'] == 'incomplete_local_configuration'
        for item in report['profiles']
    )


def test_2026_pipeline_has_no_email_login_or_private_league_dependency() -> None:
    trust_surface = [
        __import__('pathlib').Path('README.md'),
        __import__('pathlib').Path('docs/README.md'),
        __import__('pathlib').Path('docs/DASHBOARD_OPERATOR_GUIDE.md'),
        __import__('pathlib').Path('src/ffbayes/draft_2026/league.py'),
        __import__('pathlib').Path('src/ffbayes/draft_2026/pipeline.py'),
        *__import__('pathlib').Path('config/leagues').glob('*.json'),
    ]
    forbidden = (
        'email',
        'login',
        'authenticat',
    )

    violations = {
        str(path): [term for term in forbidden if term in path.read_text().lower()]
        for path in trust_surface
        if any(term in path.read_text().lower() for term in forbidden)
    }

    assert violations == {}
    allowed_profile_keys = {
        'profile_id',
        'league_name',
        'season',
        'team_count',
        'draft_format',
        'draft_slot',
        'scoring_label',
        'scoring_items',
        'scoring_overrides',
        'bonuses',
        'roster_slots',
        'bench_slots',
        'ir_slots',
        'flex_eligible',
        'waiver_type',
        'waiver_constraints',
        'settings_source',
        'settings_verified_at',
    }
    for path in __import__('pathlib').Path('config/leagues').glob('*.json'):
        assert set(json.loads(path.read_text())) <= allowed_profile_keys


def test_current_roster_fetch_disables_and_restores_nflreadpy_cache(
    monkeypatch,
) -> None:
    class Config:
        cache_mode = 'memory'

    class RosterResult:
        @staticmethod
        def to_pandas():
            return pd.DataFrame({'season': [2026]})

    config = Config()
    updates = []

    def fake_update_config(*, cache_mode):
        updates.append(cache_mode)
        config.cache_mode = cache_mode

    monkeypatch.setattr(current_pipeline, 'get_config', lambda: config)
    monkeypatch.setattr(current_pipeline, 'update_config', fake_update_config)
    monkeypatch.setattr(
        current_pipeline.nfl, 'load_rosters', lambda seasons: RosterResult()
    )

    roster = current_pipeline._load_fresh_roster(2026)

    assert roster['season'].tolist() == [2026]
    assert str(updates[0]) == 'CacheMode.OFF'
    assert updates[-1] == 'memory'
