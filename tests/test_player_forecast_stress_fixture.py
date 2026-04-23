from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from ffbayes.analysis.bayesian_player_model import (
    aggregate_season_player_table,
    build_posterior_projection_table,
)
from ffbayes.draft_strategy.draft_decision_system import (
    DraftContext,
    LeagueSettings,
    build_draft_decision_artifacts,
    save_draft_decision_artifacts,
)
from ffbayes.publish_pages import stage_pages_site
from ffbayes.utils.json_serialization import dumps_strict_json

REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_ASSIGNMENT_PREFIX = 'window.FFBAYES_DASHBOARD = '
PAYLOAD_ASSIGNMENT_SUFFIX = ';\n\n    (() => {'


def _reject_json_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def _loads_strict_json(text: str) -> dict:
    return json.loads(text, parse_constant=_reject_json_constant)


def _extract_embedded_payload(html_text: str) -> dict:
    start = html_text.find(PAYLOAD_ASSIGNMENT_PREFIX)
    if start == -1:
        raise AssertionError('Dashboard HTML is missing the inline payload assignment')
    payload_start = start + len(PAYLOAD_ASSIGNMENT_PREFIX)
    end = html_text.find(PAYLOAD_ASSIGNMENT_SUFFIX, payload_start)
    if end == -1:
        raise AssertionError('Dashboard HTML is missing the inline payload terminator')
    return _loads_strict_json(html_text[payload_start:end])


def _stress_history() -> pd.DataFrame:
    rows: list[dict] = []
    players = {
        'Stable QB': {
            'position': 'QB',
            'teams': {2022: 'KC', 2023: 'KC', 2024: 'KC', 2025: 'KC'},
            'weekly_points': {
                2022: [24, 22, 26],
                2023: [25, 23, 24],
                2024: [26, 25, 27],
                2025: [27, 26, 28],
            },
        },
        'Pocket QB': {
            'position': 'QB',
            'teams': {2022: 'LAR', 2023: 'LAR', 2024: 'LAR', 2025: 'LAR'},
            'weekly_points': {
                2022: [18, 19, 20],
                2023: [19, 18, 21],
                2024: [20, 21, 19],
                2025: [21, 20, 22],
            },
        },
        'Mover RB': {
            'position': 'RB',
            'teams': {2022: 'SF', 2023: 'SF', 2024: 'DAL', 2025: 'DAL'},
            'weekly_points': {
                2022: [17, 16, 18],
                2023: [18, 17, 19],
                2024: [16, 18, 17],
                2025: [19, 20, 18],
            },
        },
        'Committee RB': {
            'position': 'RB',
            'teams': {2022: 'NYG', 2023: 'NYG', 2024: 'NYG', 2025: 'NYG'},
            'weekly_points': {
                2022: [11, 9, 12],
                2023: [12, 10, 11],
                2024: [13, 11, 12],
                2025: [12, 13, 11],
            },
        },
        'Power RB': {
            'position': 'RB',
            'teams': {2022: 'DET', 2023: 'DET', 2024: 'DET', 2025: 'DET'},
            'weekly_points': {
                2022: [14, 15, 13],
                2023: [15, 16, 14],
                2024: [16, 15, 17],
                2025: [17, 16, 18],
            },
        },
        'Veteran WR': {
            'position': 'WR',
            'teams': {2022: 'CIN', 2023: 'CIN', 2024: 'CIN', 2025: 'CIN'},
            'weekly_points': {
                2022: [15, 14, 16],
                2023: [16, 17, 15],
                2024: [18, 16, 17],
                2025: [19, 18, 17],
            },
        },
        'Slot WR': {
            'position': 'WR',
            'teams': {2022: 'MIN', 2023: 'MIN', 2024: 'MIN', 2025: 'MIN'},
            'weekly_points': {
                2022: [13, 12, 14],
                2023: [14, 13, 15],
                2024: [15, 14, 16],
                2025: [16, 15, 17],
            },
        },
        'Field Stretcher WR': {
            'position': 'WR',
            'teams': {2022: 'MIA', 2023: 'MIA', 2024: 'MIA', 2025: 'MIA'},
            'weekly_points': {
                2022: [14, 18, 12],
                2023: [15, 19, 13],
                2024: [16, 20, 14],
                2025: [17, 21, 15],
            },
        },
        'Fragile TE': {
            'position': 'TE',
            'teams': {2022: 'BUF', 2023: 'BUF', 2024: 'BUF', 2025: 'BUF'},
            'weekly_points': {
                2022: [12, 0, 11],
                2023: [10, 0, 9],
                2024: [11, 0, 10],
                2025: [13, 0, 12],
            },
        },
        'Stable TE': {
            'position': 'TE',
            'teams': {2022: 'BAL', 2023: 'BAL', 2024: 'BAL', 2025: 'BAL'},
            'weekly_points': {
                2022: [10, 11, 10],
                2023: [11, 12, 11],
                2024: [12, 11, 13],
                2025: [13, 12, 14],
            },
        },
        'Rookie WR': {
            'position': 'WR',
            'teams': {2025: 'LAC'},
            'weekly_points': {
                2025: [11, 12, 13],
            },
        },
        'Reliable DST': {
            'position': 'DST',
            'teams': {2022: 'SF', 2023: 'SF', 2024: 'SF', 2025: 'SF'},
            'weekly_points': {
                2022: [8, 9, 7],
                2023: [9, 10, 8],
                2024: [10, 9, 11],
                2025: [11, 10, 12],
            },
        },
        'Reliable K': {
            'position': 'K',
            'teams': {2022: 'KC', 2023: 'KC', 2024: 'KC', 2025: 'KC'},
            'weekly_points': {
                2022: [9, 10, 8],
                2023: [10, 9, 11],
                2024: [11, 10, 12],
                2025: [12, 11, 13],
            },
        },
    }
    for name, profile in players.items():
        for season, points in profile['weekly_points'].items():
            for week, fant_pt in enumerate(points, start=1):
                rows.append(
                    {
                        'Season': season,
                        'Week': week,
                        'Name': name,
                        'Position': profile['position'],
                        'FantPt': float(fant_pt),
                        'FantPtPPR': float(fant_pt),
                        'Tm': profile['teams'][season],
                    }
                )
    return pd.DataFrame(rows)


def _stress_target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'Season': 2026,
                'Name': 'Stable QB',
                'Position': 'QB',
                'fantasy_points': 81.0,
                'current_team': 'KC',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Pocket QB',
                'Position': 'QB',
                'fantasy_points': 62.0,
                'current_team': 'LAR',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Mover RB',
                'Position': 'RB',
                'fantasy_points': 60.0,
                'current_team': 'NYJ',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Committee RB',
                'Position': 'RB',
                'fantasy_points': 41.0,
                'current_team': 'NYG',
                'depth_chart_rank': 2,
            },
            {
                'Season': 2026,
                'Name': 'Power RB',
                'Position': 'RB',
                'fantasy_points': 52.0,
                'current_team': 'DET',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Veteran WR',
                'Position': 'WR',
                'fantasy_points': 54.0,
                'current_team': 'CIN',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Slot WR',
                'Position': 'WR',
                'fantasy_points': 49.0,
                'current_team': 'MIN',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Field Stretcher WR',
                'Position': 'WR',
                'fantasy_points': 51.0,
                'current_team': 'MIA',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Fragile TE',
                'Position': 'TE',
                'fantasy_points': 30.0,
                'current_team': 'BUF',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Stable TE',
                'Position': 'TE',
                'fantasy_points': 39.0,
                'current_team': 'BAL',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Rookie WR',
                'Position': 'WR',
                'fantasy_points': 42.0,
                'current_team': 'LAC',
                'rookie_draft_round': 1,
                'rookie_draft_pick': 18,
                'rookie_combine_score': 0.82,
                'depth_chart_rank': 2,
            },
            {
                'Season': 2026,
                'Name': 'Reliable DST',
                'Position': 'DST',
                'fantasy_points': 33.0,
                'current_team': 'SF',
                'depth_chart_rank': 1,
            },
            {
                'Season': 2026,
                'Name': 'Reliable K',
                'Position': 'K',
                'fantasy_points': 36.0,
                'current_team': 'KC',
                'depth_chart_rank': 1,
            },
        ]
    )


def test_player_forecast_stress_fixture_exercises_forecast_and_dashboard(tmp_path):
    history = _stress_history()
    season_table = aggregate_season_player_table(history)
    projection_table = build_posterior_projection_table(
        train_history=season_table,
        target_frame=_stress_target_frame(),
        holdout_year=2026,
        min_history_seasons=0,
    )

    artifacts = build_draft_decision_artifacts(
        projection_table,
        league_settings=LeagueSettings(),
        context=DraftContext(current_pick_number=10),
        season_history=history,
    )

    assert {'QB', 'RB', 'WR', 'TE'}.issubset(set(artifacts.decision_table['position']))
    assert (artifacts.decision_table['player_name'] == 'Rookie WR').any()
    assert (
        artifacts.dashboard_payload['player_forecast_validation']['status']
        == 'available'
    )
    assert (
        artifacts.dashboard_payload['war_room_visuals']['supported_model'][
            'validation_status'
        ]
        == 'available'
    )
    assert (
        artifacts.dashboard_payload['decision_evidence']['supported_model'][
            'production_estimator'
        ]
        == 'hierarchical_empirical_bayes'
    )
    payload_text = dumps_strict_json(artifacts.dashboard_payload)
    assert 'hybrid' not in payload_text.lower()
    assert 'sampled_hierarchical_bayes' not in payload_text
    assert any(
        row['slice'] == 'rookie'
        for row in artifacts.dashboard_payload['player_forecast_validation'][
            'cohort_slices'
        ]
    )
    assert 'component_diagnostics' in artifacts.dashboard_payload[
        'player_forecast_validation'
    ]

    output_dir = tmp_path / 'seasons' / '2026' / 'draft_strategy'
    diagnostics_dir = tmp_path / 'seasons' / '2026' / 'diagnostics'
    saved = save_draft_decision_artifacts(
        artifacts,
        output_dir=output_dir,
        year=2026,
        dashboard_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
    )

    assert saved['player_forecast_validation_path'].exists()
    assert saved['validation_summary_path'].parent == diagnostics_dir / 'validation'
    assert saved['validation_summary_path'].name.startswith(
        'player_forecast_validation_summary_'
    )
    assert saved['validation_summary_path'].exists()
    assert 'Decision evidence' in saved['html_path'].read_text(encoding='utf-8')
    assert 'Forecast validation' in saved['html_path'].read_text(encoding='utf-8')
    _loads_strict_json(saved['payload_path'].read_text(encoding='utf-8'))
    _loads_strict_json(
        saved['player_forecast_validation_path'].read_text(encoding='utf-8')
    )
    _loads_strict_json(saved['validation_summary_path'].read_text(encoding='utf-8'))


def test_player_forecast_stress_fixture_preserves_artifact_lineage(
    tmp_path, monkeypatch
):
    history = _stress_history()
    season_table = aggregate_season_player_table(history)
    projection_table = build_posterior_projection_table(
        train_history=season_table,
        target_frame=_stress_target_frame(),
        holdout_year=2026,
        min_history_seasons=0,
    )

    artifacts = build_draft_decision_artifacts(
        projection_table,
        league_settings=LeagueSettings(),
        context=DraftContext(current_pick_number=10),
        season_history=history,
    )

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    output_dir = runtime_root / 'seasons' / '2026' / 'draft_strategy'
    diagnostics_dir = runtime_root / 'seasons' / '2026' / 'diagnostics'
    saved = save_draft_decision_artifacts(
        artifacts,
        output_dir=output_dir,
        year=2026,
        dashboard_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
    )
    staged = stage_pages_site(
        year=2026,
        source_html=saved['html_path'],
        source_payload=saved['payload_path'],
        output_dir=project_root / 'site',
    )

    canonical_payload = _loads_strict_json(
        saved['payload_path'].read_text(encoding='utf-8')
    )
    runtime_payload = _loads_strict_json(
        saved['runtime_dashboard_payload'].read_text(encoding='utf-8')
    )
    repo_payload = _loads_strict_json(
        saved['repo_dashboard_payload'].read_text(encoding='utf-8')
    )
    staged_payload = _loads_strict_json(
        staged['payload_path'].read_text(encoding='utf-8')
    )
    staged_provenance = _loads_strict_json(
        staged['provenance_path'].read_text(encoding='utf-8')
    )
    _loads_strict_json(saved['player_forecast_path'].read_text(encoding='utf-8'))
    _loads_strict_json(
        saved['player_forecast_diagnostics_path'].read_text(encoding='utf-8')
    )
    _loads_strict_json(
        saved['player_forecast_validation_path'].read_text(encoding='utf-8')
    )
    _loads_strict_json(saved['validation_summary_path'].read_text(encoding='utf-8'))

    assert runtime_payload == canonical_payload
    assert repo_payload == canonical_payload
    embedded_payload = _extract_embedded_payload(
        saved['html_path'].read_text(encoding='utf-8')
    )
    assert embedded_payload == canonical_payload
    staged_payload_without_publish = dict(staged_payload)
    staged_payload_without_publish.pop('publish_provenance', None)
    assert staged_payload_without_publish == canonical_payload
    for field in [
        'decision_evidence',
        'player_forecast_validation',
        'metric_glossary',
        'war_room_visuals',
    ]:
        assert runtime_payload[field] == canonical_payload[field]
        assert repo_payload[field] == canonical_payload[field]
        assert staged_payload[field] == canonical_payload[field]
    assert staged_payload['publish_provenance']['source_html'] == 'draft_board_2026.html'
    assert (
        staged_payload['publish_provenance']['source_payload']
        == 'dashboard_payload_2026.json'
    )
    assert staged_provenance['source_html'] == 'draft_board_2026.html'
    assert staged_provenance['source_payload'] == 'dashboard_payload_2026.json'
    assert sorted(path.name for path in (runtime_root / 'dashboard').iterdir()) == [
        'dashboard_payload.json',
        'index.html',
    ]
    assert sorted(path.name for path in (project_root / 'dashboard').iterdir()) == [
        'dashboard_payload.json',
        'index.html',
    ]
    assert sorted(path.name for path in staged['site_dir'].iterdir()) == [
        '.nojekyll',
        'dashboard_payload.json',
        'index.html',
        'publish_provenance.json',
    ]
    expected_output_entries = [
        'dashboard_payload_2026.json',
        'draft_board_2026.html',
        'draft_board_2026.xlsx',
        'model_outputs',
    ]
    if saved['backtest_path'].exists():
        expected_output_entries.append(saved['backtest_path'].name)
    output_entries = sorted(path.name for path in output_dir.iterdir())
    assert output_entries == sorted(expected_output_entries)
    model_output_entries = sorted(
        path.name for path in (output_dir / 'model_outputs').iterdir()
    )
    assert model_output_entries == [
        'current_year_model_comparison_2026.json',
        'player_forecast',
    ]
    player_forecast_entries = sorted(
        path.name for path in (output_dir / 'model_outputs' / 'player_forecast').iterdir()
    )
    assert player_forecast_entries == [
        'player_forecast_2026.json',
        'player_forecast_diagnostics_2026.json',
        saved['player_forecast_validation_path'].name,
    ]
    assert sorted(path.name for path in diagnostics_dir.iterdir()) == ['validation']
    assert sorted(path.name for path in (diagnostics_dir / 'validation').iterdir()) == [
        saved['validation_summary_path'].name
    ]


def test_player_forecast_fixture_drives_full_dashboard_smoke(
    tmp_path, monkeypatch
):
    history = _stress_history()
    season_table = aggregate_season_player_table(history)
    projection_table = build_posterior_projection_table(
        train_history=season_table,
        target_frame=_stress_target_frame(),
        holdout_year=2026,
        min_history_seasons=0,
    )

    artifacts = build_draft_decision_artifacts(
        projection_table,
        league_settings=LeagueSettings(),
        context=DraftContext(current_pick_number=10),
        season_history=history,
    )

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    output_dir = runtime_root / 'seasons' / '2026' / 'draft_strategy'
    diagnostics_dir = runtime_root / 'seasons' / '2026' / 'diagnostics'
    saved = save_draft_decision_artifacts(
        artifacts,
        output_dir=output_dir,
        year=2026,
        dashboard_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
    )
    staged = stage_pages_site(
        year=2026,
        source_html=saved['html_path'],
        source_payload=saved['payload_path'],
        output_dir=project_root / 'site',
    )

    env = os.environ.copy()
    env['FFBAYES_SMOKE_SITE_DIR'] = str(staged['site_dir'])
    completed = subprocess.run(
        ['node', 'tests/dashboard_smoke.mjs'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    smoke = json.loads(completed.stdout)
    assert smoke['title'] == 'FFBayes dashboard smoke test'
    assert smoke['rosterComplete'] is True
    assert len(smoke['finalizedFiles']) == 3
    assert any(name.endswith('.json') for name in smoke['finalizedFiles'])
    assert sum(name.endswith('.html') for name in smoke['finalizedFiles']) == 2
    assert len(smoke['draftedPlayers']) >= 5
    assert any(pill.startswith('Yours:') for pill in smoke['finalPills'])
