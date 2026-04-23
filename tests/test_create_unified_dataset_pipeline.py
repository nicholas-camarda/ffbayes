from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ffbayes.data_pipeline.create_unified_dataset as cud
import ffbayes.data_pipeline.preprocess_analysis_data as preprocess_analysis_data
import ffbayes.utils.path_constants as path_constants
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
from ffbayes.utils.vor_filename_generator import get_vor_csv_filename


def _make_combined_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'Name': 'Alpha Player',
                'Position': 'RB',
                'FantPt': 12.5,
                'FantPtPPR': 15.0,
                'Season': 2025,
                'G#': 1,
                'Opp': 'DAL',
                'Tm': 'NYG',
                'Date': '2025-09-01',
                'is_home': 1,
            },
            {
                'Name': 'Beta Player',
                'Position': 'WR',
                'FantPt': 8.0,
                'FantPtPPR': 9.5,
                'Season': 2025,
                'G#': 1,
                'Opp': 'NYG',
                'Tm': 'DAL',
                'Date': '2025-09-01',
                'is_home': 0,
            },
        ]
    )


def _write_vor_snapshot(path: Path) -> None:
    pd.DataFrame(
        [
            {
                'PLAYER': 'Alpha Player',
                'POS': 'RB',
                'AVG': 4.0,
                'FPTS': 240.0,
                'VOR': 55.0,
                'VALUERANK': 1,
            },
            {
                'PLAYER': 'Beta Player',
                'POS': 'WR',
                'AVG': 8.0,
                'FPTS': 220.0,
                'VOR': 35.0,
                'VALUERANK': 2,
            },
        ]
    ).to_csv(path, index=False)


def _make_raw_season_frame(season: int) -> pd.DataFrame:
    rows: list[dict] = []
    alpha_points = [10.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]
    beta_points = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]
    for week, (alpha, beta) in enumerate(zip(alpha_points, beta_points), start=1):
        rows.append(
            {
                'G#': week,
                'Date': f'{season}-09-{week:02d}',
                'Tm': 'NYG',
                'Away': 'DAL',
                'Opp': 'DAL',
                'FantPt': alpha,
                'FantPtPPR': alpha + 2.0,
                'Name': 'Alpha Player',
                'PlayerID': f'alpha-{season}',
                'Position': 'RB',
                'Season': season,
            }
        )
        rows.append(
            {
                'G#': week,
                'Date': f'{season}-09-{week:02d}',
                'Tm': 'DAL',
                'Away': 'DAL',
                'Opp': 'NYG',
                'FantPt': beta,
                'FantPtPPR': beta + 1.5,
                'Name': 'Beta Player',
                'PlayerID': f'beta-{season}',
                'Position': 'WR',
                'Season': season,
            }
        )
    return pd.DataFrame(rows)


def test_resolve_existing_vor_csv_prefers_runtime_path(tmp_path, monkeypatch):
    runtime_dir = tmp_path / 'runtime' / 'snake_draft'
    organized_dir = tmp_path / 'runtime' / 'organized'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', runtime_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )

    runtime_file = runtime_dir / get_vor_csv_filename(2026)
    organized_file = organized_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(runtime_file)
    _write_vor_snapshot(organized_file)

    resolved = cud._resolve_existing_vor_csv(2026)

    assert resolved == runtime_file


def test_resolve_existing_vor_csv_falls_back_to_organized_path(
    tmp_path, monkeypatch
):
    runtime_dir = tmp_path / 'runtime' / 'snake_draft'
    organized_dir = tmp_path / 'runtime' / 'organized'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', runtime_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )

    organized_file = organized_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(organized_file)

    resolved = cud._resolve_existing_vor_csv(2026)

    assert resolved == organized_file


def test_resolve_existing_vor_csv_returns_none_when_absent(tmp_path, monkeypatch):
    runtime_dir = tmp_path / 'runtime' / 'snake_draft'
    organized_dir = tmp_path / 'runtime' / 'organized'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', runtime_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )

    resolved = cud._resolve_existing_vor_csv(2026)

    assert resolved is None


def test_create_unified_dataset_reuses_existing_vor_and_writes_compact_outputs(
    tmp_path, monkeypatch
):
    runtime_dir = tmp_path / 'runtime'
    snake_dir = runtime_dir / 'inputs' / 'processed' / 'snake_draft_datasets'
    unified_dir = runtime_dir / 'inputs' / 'processed' / 'unified_dataset'
    organized_dir = runtime_dir / 'seasons' / '2026' / 'vor_strategy'
    snake_dir.mkdir(parents=True, exist_ok=True)
    unified_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    vor_file = snake_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(vor_file)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', snake_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants, 'UNIFIED_DATASET_DIR', unified_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_path',
        lambda: unified_dir / 'unified_dataset.json',
        raising=False,
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_csv_path',
        lambda: unified_dir / 'unified_dataset.csv',
        raising=False,
    )
    monkeypatch.setattr(
        cud,
        'load_config',
        lambda: {'ppr': 0.5, 'top_rank': 120},
    )
    monkeypatch.setattr(cud, 'load_combined_dataset', lambda data_directory=None: _make_combined_history())
    monkeypatch.setattr(cud, 'validate_data_quality', lambda data: None)
    monkeypatch.setattr(cud, 'clean_data_types', lambda data: data)
    monkeypatch.setattr(cud, 'calculate_basic_features', lambda data: data)
    monkeypatch.setattr(cud, 'add_predraft_composites', lambda data, risk_tolerance='medium': data)
    monkeypatch.setattr(
        cud,
        '_load_player_context_snapshot',
        lambda current_year: pd.DataFrame(
            [
                {
                    '__match_name': 'alpha player',
                    '__match_position': 'RB',
                    'current_team': 'NYG',
                    'rookie_draft_round': 1,
                    'rookie_draft_pick': 12,
                    'rookie_combine_score': 0.7,
                    'depth_chart_rank': 1,
                    'context_available': True,
                }
            ]
        ),
    )

    def _unexpected_call(*args, **kwargs):
        raise AssertionError('scraping should not run when a current VOR file exists')

    monkeypatch.setattr(cud, 'scrape_adp_data', _unexpected_call)
    monkeypatch.setattr(cud, 'scrape_projection_data', _unexpected_call)
    monkeypatch.setattr(cud, 'calculate_vor_rankings', _unexpected_call)
    monkeypatch.setattr(cud, 'save_vor_data', _unexpected_call)

    result = cud.create_unified_dataset()

    csv_path = unified_dir / 'unified_dataset.csv'
    json_path = unified_dir / 'unified_dataset.json'
    excel_path = unified_dir / 'unified_dataset.xlsx'

    assert csv_path.exists()
    assert json_path.exists()
    assert not excel_path.exists()
    assert 'current_vor_value' in result.columns
    assert result['current_vor_value'].notna().any()
    assert result['vor_global_rank'].notna().any()
    assert 'current_team' in result.columns
    assert 'rookie_draft_pick' in result.columns
    assert result.loc[result['Name'] == 'Alpha Player', 'current_team'].iloc[0] == 'NYG'
    assert result.loc[result['Name'] == 'Alpha Player', 'current_vor_rank'].iloc[0] == 1
    assert json.loads(json_path.read_text(encoding='utf-8'))[0]['Name'] == 'Alpha Player'


def test_small_pipeline_fixture_runs_from_unified_dataset_to_dashboard(
    tmp_path, monkeypatch
):
    runtime_dir = tmp_path / 'runtime'
    snake_dir = runtime_dir / 'inputs' / 'processed' / 'snake_draft_datasets'
    unified_dir = runtime_dir / 'inputs' / 'processed' / 'unified_dataset'
    organized_dir = runtime_dir / 'seasons' / '2026' / 'vor_strategy'
    snake_dir.mkdir(parents=True, exist_ok=True)
    unified_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    vor_file = snake_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(vor_file)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', snake_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants, 'UNIFIED_DATASET_DIR', unified_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_path',
        lambda: unified_dir / 'unified_dataset.json',
        raising=False,
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_csv_path',
        lambda: unified_dir / 'unified_dataset.csv',
        raising=False,
    )
    monkeypatch.setattr(cud, 'load_config', lambda: {'ppr': 0.5, 'top_rank': 120})
    monkeypatch.setattr(
        cud, 'load_combined_dataset', lambda data_directory=None: _make_combined_history()
    )
    monkeypatch.setattr(cud, 'validate_data_quality', lambda data: None)
    monkeypatch.setattr(cud, 'clean_data_types', lambda data: data)
    monkeypatch.setattr(cud, 'calculate_basic_features', lambda data: data)
    monkeypatch.setattr(
        cud, 'add_predraft_composites', lambda data, risk_tolerance='medium': data
    )
    monkeypatch.setattr(
        cud,
        '_load_player_context_snapshot',
        lambda current_year: pd.DataFrame(
            [
                {
                    '__match_name': 'alpha player',
                    '__match_position': 'RB',
                    'current_team': 'NYG',
                    'rookie_draft_round': 1,
                    'rookie_draft_pick': 12,
                    'rookie_combine_score': 0.7,
                    'depth_chart_rank': 1,
                    'context_available': True,
                },
                {
                    '__match_name': 'beta player',
                    '__match_position': 'WR',
                    'current_team': 'DAL',
                    'rookie_draft_round': None,
                    'rookie_draft_pick': None,
                    'rookie_combine_score': None,
                    'depth_chart_rank': 1,
                    'context_available': True,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        cud,
        'scrape_adp_data',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('scraping should not run when a current VOR file exists')
        ),
    )
    monkeypatch.setattr(cud, 'scrape_projection_data', cud.scrape_adp_data)
    monkeypatch.setattr(cud, 'calculate_vor_rankings', cud.scrape_adp_data)
    monkeypatch.setattr(cud, 'save_vor_data', cud.scrape_adp_data)

    unified = cud.create_unified_dataset()
    season_table = aggregate_season_player_table(unified)
    target_frame = (
        unified.sort_values(['Name', 'Season'])
        .groupby('Name', as_index=False)
        .tail(1)
        .assign(
            Season=2026,
            fantasy_points=lambda df: df['FantPtPPR'],
            current_team=lambda df: df['current_team'].fillna(df['Tm']),
            depth_chart_rank=lambda df: df['depth_chart_rank'].fillna(1),
        )[
            [
                'Season',
                'Name',
                'Position',
                'fantasy_points',
                'current_team',
                'rookie_draft_round',
                'rookie_draft_pick',
                'rookie_combine_score',
                'depth_chart_rank',
            ]
        ]
        .reset_index(drop=True)
    )

    projection_table = build_posterior_projection_table(
        train_history=season_table,
        target_frame=target_frame,
        holdout_year=2026,
        min_history_seasons=0,
    )
    artifacts = build_draft_decision_artifacts(
        projection_table,
        league_settings=LeagueSettings(),
        context=DraftContext(current_pick_number=10),
        season_history=unified,
    )
    output_dir = runtime_dir / 'seasons' / '2026' / 'draft_strategy'
    diagnostics_dir = runtime_dir / 'seasons' / '2026' / 'diagnostics'
    saved = save_draft_decision_artifacts(
        artifacts,
        output_dir=output_dir,
        year=2026,
        dashboard_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
    )

    assert not projection_table.empty
    assert saved['payload_path'].exists()
    assert saved['html_path'].exists()
    payload = json.loads(saved['payload_path'].read_text(encoding='utf-8'))
    assert payload['decision_evidence']['status'] in {
        'available',
        'degraded',
        'unavailable',
    }
    assert payload['player_forecast_validation']['status'] in {
        'available',
        'unavailable',
    }


def test_preprocess_to_canonical_dashboard_fixture_path(tmp_path, monkeypatch):
    runtime_dir = tmp_path / 'runtime'
    raw_dir = runtime_dir / 'inputs' / 'raw'
    season_dir = raw_dir / 'season_datasets'
    combined_dir = runtime_dir / 'inputs' / 'processed' / 'combined_datasets'
    snake_dir = runtime_dir / 'inputs' / 'processed' / 'snake_draft_datasets'
    unified_dir = runtime_dir / 'inputs' / 'processed' / 'unified_dataset'
    project_root = tmp_path / 'project'

    season_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    snake_dir.mkdir(parents=True, exist_ok=True)
    unified_dir.mkdir(parents=True, exist_ok=True)

    for season in range(2021, 2026):
        _make_raw_season_frame(season).to_csv(
            season_dir / f'{season}season.csv', index=False
        )

    vor_file = snake_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(vor_file)

    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_dir))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setattr(path_constants, 'RUNTIME_DIR', runtime_dir, raising=False)
    monkeypatch.setattr(path_constants, 'BASE_DIR', project_root, raising=False)
    monkeypatch.setattr(path_constants, 'RAW_DATA_DIR', raw_dir, raising=False)
    monkeypatch.setattr(
        path_constants, 'SEASON_DATASETS_DIR', season_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants, 'COMBINED_DATASETS_DIR', combined_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', snake_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants, 'UNIFIED_DATASET_DIR', unified_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_path',
        lambda: unified_dir / 'unified_dataset.json',
        raising=False,
    )
    monkeypatch.setattr(
        path_constants,
        'get_unified_dataset_csv_path',
        lambda: unified_dir / 'unified_dataset.csv',
        raising=False,
    )
    monkeypatch.setattr(cud, 'load_config', lambda: {'ppr': 0.5, 'top_rank': 120})
    monkeypatch.setattr(
        cud,
        '_load_player_context_snapshot',
        lambda current_year: pd.DataFrame(
            [
                {
                    '__match_name': 'alpha player',
                    '__match_position': 'RB',
                    'current_team': 'NYG',
                    'rookie_draft_round': 1,
                    'rookie_draft_pick': 12,
                    'rookie_combine_score': 0.7,
                    'depth_chart_rank': 1,
                    'context_available': True,
                },
                {
                    '__match_name': 'beta player',
                    '__match_position': 'WR',
                    'current_team': 'DAL',
                    'rookie_draft_round': None,
                    'rookie_draft_pick': None,
                    'rookie_combine_score': None,
                    'depth_chart_rank': 1,
                    'context_available': True,
                },
            ]
        ),
    )
    monkeypatch.setattr(cud, 'scrape_adp_data', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('scraping should not run when a current VOR file exists')))
    monkeypatch.setattr(cud, 'scrape_projection_data', cud.scrape_adp_data)
    monkeypatch.setattr(cud, 'calculate_vor_rankings', cud.scrape_adp_data)
    monkeypatch.setattr(cud, 'save_vor_data', cud.scrape_adp_data)

    processed, _ = preprocess_analysis_data.create_analysis_dataset(str(runtime_dir))
    combined_dataset_path = combined_dir / '2021-2025season_modern.csv'
    unified = cud.create_unified_dataset()
    unified_csv_path = unified_dir / 'unified_dataset.csv'
    unified_json_path = unified_dir / 'unified_dataset.json'
    season_table = aggregate_season_player_table(unified)
    target_frame = (
        unified.sort_values(['Name', 'Season'])
        .groupby('Name', as_index=False)
        .tail(1)
        .assign(
            Season=2026,
            fantasy_points=lambda df: df['FantPtPPR'],
            current_team=lambda df: df['current_team'].fillna(df['Tm']),
            depth_chart_rank=lambda df: df['depth_chart_rank'].fillna(1),
        )[
            [
                'Season',
                'Name',
                'Position',
                'fantasy_points',
                'current_team',
                'rookie_draft_round',
                'rookie_draft_pick',
                'rookie_combine_score',
                'depth_chart_rank',
            ]
        ]
        .reset_index(drop=True)
    )
    projection_table = build_posterior_projection_table(
        train_history=season_table,
        target_frame=target_frame,
        holdout_year=2026,
        min_history_seasons=0,
    )
    artifacts = build_draft_decision_artifacts(
        projection_table,
        league_settings=LeagueSettings(),
        context=DraftContext(current_pick_number=10),
        season_history=unified,
    )
    output_dir = path_constants.get_draft_strategy_dir(2026)
    diagnostics_dir = path_constants.get_pre_draft_diagnostics_dir(2026)
    saved = save_draft_decision_artifacts(
        artifacts,
        output_dir=output_dir,
        year=2026,
        dashboard_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
    )

    assert not processed.empty
    assert combined_dataset_path.exists()
    assert unified_csv_path.exists()
    assert unified_json_path.exists()
    assert not projection_table.empty
    assert saved['payload_path'] == path_constants.get_dashboard_payload_path(2026)
    assert saved['html_path'] == path_constants.get_dashboard_html_path(2026)
    html = saved['html_path'].read_text(encoding='utf-8')
    payload = json.loads(saved['payload_path'].read_text(encoding='utf-8'))
    assert 'Decision evidence' in html
    assert 'Projection breakdown' in html
    assert 'Season total mean' in html
    assert 'Detailed evidence' in html
    assert payload['decision_evidence']['status'] in {
        'available',
        'degraded',
        'unavailable',
    }
    assert payload['player_forecast_validation']['status'] in {
        'available',
        'unavailable',
    }
