from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import ffbayes.analysis.draft_decision_backtest as draft_decision_backtest


class _FakeDateTime:
    @classmethod
    def now(cls):
        return SimpleNamespace(year=2026)


def _write_season_csv(path, year: int) -> None:
    pd.DataFrame(
        {
            'Season': [year],
            'Name': ['Alpha Player'],
            'Position': ['RB'],
            'FantPt': [150.0],
        }
    ).to_csv(path / f'{year}season.csv', index=False)


def test_load_season_history_with_freshness_fails_closed_without_override(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in [2021, 2022, 2023, 2024]:
        _write_season_csv(season_dir, year)

    monkeypatch.setattr(draft_decision_backtest, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(draft_decision_backtest, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(draft_decision_backtest, 'datetime', _FakeDateTime)
    monkeypatch.delenv('FFBAYES_ALLOW_STALE_SEASON', raising=False)

    with pytest.raises(RuntimeError, match='Latest expected season 2025 is missing'):
        draft_decision_backtest.load_season_history_with_freshness()

    assert not (raw_dir / 'draft_backtest_freshness.json').exists()


def test_load_season_history_with_freshness_records_degraded_override(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in [2021, 2022, 2023, 2024]:
        _write_season_csv(season_dir, year)

    monkeypatch.setattr(draft_decision_backtest, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(draft_decision_backtest, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(draft_decision_backtest, 'datetime', _FakeDateTime)
    monkeypatch.setenv('FFBAYES_ALLOW_STALE_SEASON', 'true')

    season_history, manifest = draft_decision_backtest.load_season_history_with_freshness()

    assert not season_history.empty
    assert manifest['freshness']['status'] == 'degraded'
    assert manifest['freshness']['override_used'] is True
    saved_manifest = json.loads(
        (raw_dir / 'draft_backtest_freshness.json').read_text(encoding='utf-8')
    )
    assert saved_manifest['freshness']['status'] == 'degraded'
    assert saved_manifest['override_used'] is True
