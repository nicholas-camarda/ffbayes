from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ffbayes.data_pipeline.create_unified_dataset as cud
import ffbayes.utils.path_constants as path_constants
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


def test_resolve_existing_vor_csv_finds_legacy_path(tmp_path, monkeypatch):
    runtime_dir = tmp_path / 'runtime' / 'snake_draft'
    organized_dir = tmp_path / 'runtime' / 'organized'
    legacy_dir = (
        tmp_path
        / 'legacy'
        / 'ProjectsRuntime'
        / 'ffbayes'
        / 'data'
        / 'processed'
        / 'snake_draft_datasets'
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        path_constants, 'SNAKE_DRAFT_DATASETS_DIR', runtime_dir, raising=False
    )
    monkeypatch.setattr(
        path_constants,
        'get_vor_strategy_dir',
        lambda year: organized_dir,
        raising=False,
    )

    legacy_file = legacy_dir / get_vor_csv_filename(2026)
    _write_vor_snapshot(legacy_file)

    monkeypatch.setattr(
        cud.Path,
        'home',
        lambda: tmp_path / 'legacy',
        raising=False,
    )

    resolved = cud._resolve_existing_vor_csv(2026)

    assert resolved == legacy_file


def test_create_unified_dataset_reuses_existing_vor_and_writes_compact_outputs(
    tmp_path, monkeypatch
):
    runtime_dir = tmp_path / 'runtime'
    snake_dir = runtime_dir / 'data' / 'processed' / 'snake_draft_datasets'
    unified_dir = runtime_dir / 'data' / 'processed' / 'unified_dataset'
    organized_dir = runtime_dir / 'results' / '2026' / 'pre_draft' / 'vor_strategy'
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
    assert result.loc[result['Name'] == 'Alpha Player', 'current_vor_rank'].iloc[0] == 1
    assert json.loads(json_path.read_text(encoding='utf-8'))[0]['Name'] == 'Alpha Player'
