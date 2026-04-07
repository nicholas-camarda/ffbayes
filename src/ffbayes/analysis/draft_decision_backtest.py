#!/usr/bin/env python3
"""Draft decision backtest entrypoint."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    LeagueSettings,
    run_draft_backtest,
)
from ffbayes.utils.analysis_windows import (
    build_freshness_manifest,
    get_analysis_years,
    resolve_analysis_window,
    stale_data_override_enabled,
    write_freshness_manifest,
)
from ffbayes.utils.path_constants import (
    RAW_DATA_DIR,
    SEASON_DATASETS_DIR,
    get_draft_decision_backtest_path,
    get_draft_strategy_dir,
)

logger = logging.getLogger(__name__)


def load_season_history_with_freshness() -> tuple[pd.DataFrame, dict]:
    files = sorted(SEASON_DATASETS_DIR.glob('*season.csv'))
    if not files:
        raise FileNotFoundError(f'No season CSVs found in {SEASON_DATASETS_DIR}')
    allow_stale = stale_data_override_enabled()
    expected_years = get_analysis_years(datetime.now().year)
    found_years = [
        int(file_path.name.replace('season.csv', ''))
        for file_path in files
        if file_path.name.endswith('season.csv')
    ]
    freshness_window = resolve_analysis_window(
        found_years,
        reference_year=datetime.now().year,
        allow_stale=allow_stale,
    )
    freshness_manifest = build_freshness_manifest(
        freshness_window,
        source_name='draft_backtest',
        source_path=SEASON_DATASETS_DIR,
        found_files=files,
    )
    write_freshness_manifest(
        freshness_manifest,
        RAW_DATA_DIR / 'draft_backtest_freshness.json',
    )
    for warning in freshness_window.warnings:
        logger.warning('%s', warning)
    if freshness_window.freshness_status == 'degraded':
        logger.warning(
            'Running draft backtest with degraded freshness because '
            'FFBAYES_ALLOW_STALE_SEASON=true was explicitly set.'
        )
    if expected_years:
        files = [
            file_path
            for file_path in files
            if int(file_path.name.replace('season.csv', '')) in expected_years
        ]
    return (
        pd.concat((pd.read_csv(file_path) for file_path in files), ignore_index=True),
        freshness_manifest,
    )


def load_season_history() -> pd.DataFrame:
    season_history, _ = load_season_history_with_freshness()
    return season_history


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    current_year = datetime.now().year
    season_history, freshness_manifest = load_season_history_with_freshness()
    settings = LeagueSettings()
    backtest = run_draft_backtest(season_history, settings)
    backtest['freshness_manifest'] = freshness_manifest
    backtest['freshness'] = freshness_manifest.get('freshness', {})
    year_range = (
        f'{min(backtest["holdout_years"])}-{max(backtest["holdout_years"])}'
        if backtest.get('holdout_years')
        else str(current_year)
    )
    output_path = Path(
        get_draft_decision_backtest_path(current_year, year_range=year_range)
    )
    output_path.write_text(
        json.dumps(backtest, default=str, indent=2), encoding='utf-8'
    )
    historical_path = get_draft_strategy_dir(current_year) / (
        f'historical_strategy_backtest_{year_range}.json'
    )
    historical_payload = dict(backtest)
    historical_payload['comparison_type'] = 'historical_strategy_backtest'
    historical_path.write_text(
        json.dumps(historical_payload, default=str, indent=2), encoding='utf-8'
    )
    logger.info('Draft decision backtest saved to: %s', output_path)
    logger.info('Historical strategy backtest saved to: %s', historical_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
