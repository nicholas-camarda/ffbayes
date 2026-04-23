#!/usr/bin/env python3
"""Build the canonical combined historical dataset for analysis.

This step reads season-level weekly inputs from
`inputs/raw/season_datasets/`, enforces the supported rolling five-season
analysis window, computes core derived columns such as the seven-game rolling
average, and writes the machine-readable combined dataset under
`inputs/processed/combined_datasets/`.
"""

import glob
import os
from pathlib import Path
from typing import Any

# Add scripts/utils to path for progress monitoring
import pandas as pd

_ProgressMonitor: Any = None
try:
    from ffbayes.utils.progress_monitor import (
        ProgressMonitor as _ImportedProgressMonitor,
    )
except Exception:
    pass
else:
    _ProgressMonitor = _ImportedProgressMonitor

ProgressMonitor: Any = _ProgressMonitor

POSITION_ID_MAP = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'DST': 4, 'K': 5}


def _encode_team_codes(values: pd.Series, team_names: pd.Index) -> pd.Series:
    """Map team labels to stable integer codes, using -1 for unknown values."""
    return pd.Series(team_names.get_indexer(values), index=values.index).astype(int)


def create_analysis_dataset(path_to_data_directory):
    """Create the canonical combined analysis dataset from raw season files."""
    print('Loading and preprocessing data for analysis...')

    # Read in the datasets and combine - look for season.csv files in season_datasets subdirectory
    from ffbayes.utils.analysis_windows import (
        build_freshness_manifest,
        get_analysis_years,
        resolve_analysis_window,
        stale_data_override_enabled,
        write_freshness_manifest,
    )
    from ffbayes.utils.path_constants import RAW_DATA_DIR, SEASON_DATASETS_DIR

    all_files = glob.glob(str(SEASON_DATASETS_DIR / '*season.csv'))
    if not all_files:
        raise ValueError(f'No season data files found in {SEASON_DATASETS_DIR}')

    from datetime import datetime

    current_year = datetime.now().year
    target_years = get_analysis_years(current_year)
    allow_stale = stale_data_override_enabled()

    # Filter files to only include the last 5 years
    filtered_files = []
    for file in all_files:
        filename = os.path.basename(file)
        year = int(filename.replace('season.csv', ''))
        if year in target_years:
            filtered_files.append(file)

    if not filtered_files:
        raise ValueError(
            f'No season data files found for the last 5 years ({target_years})'
        )

    freshness_window = resolve_analysis_window(
        [
            int(os.path.basename(file).replace('season.csv', ''))
            for file in filtered_files
        ],
        reference_year=current_year,
        allow_stale=allow_stale,
    )
    freshness_manifest = build_freshness_manifest(
        freshness_window,
        source_name='season_datasets',
        source_path=SEASON_DATASETS_DIR,
        found_files=[Path(file) for file in filtered_files],
    )
    write_freshness_manifest(
        freshness_manifest, RAW_DATA_DIR / 'preprocess_freshness_manifest.json'
    )

    print(
        f'Using {len(filtered_files)} season files for last 5 years: {[os.path.basename(f) for f in filtered_files]}'
    )
    print(f'Target years: {target_years}')
    if freshness_window.warnings:
        for warning in freshness_window.warnings:
            print(f'⚠️  {warning}')
    if freshness_window.freshness_status == 'degraded':
        print(
            '⚠️  Proceeding with a degraded preprocessing window because '
            'FFBAYES_ALLOW_STALE_SEASON=true was explicitly set.'
        )

    data_temp = pd.concat((pd.read_csv(f) for f in filtered_files), ignore_index=True)
    print(f'Combined data shape: {data_temp.shape}')
    print(f'Available columns: {data_temp.columns.tolist()}')
    print(f'Season range: {data_temp["Season"].min()} - {data_temp["Season"].max()}')

    # Sort properly
    data = data_temp.sort_values(
        by=['Season', 'Name', 'G#'], ascending=[True, True, True]
    )

    # One-hot-encode the positions
    data['pos_id'] = data['Position']
    data['position'] = data['Position']
    data = pd.get_dummies(data, columns=['position'])

    # Identify teams with integer encoding
    team_names = pd.Index(
        pd.unique(
            pd.concat([data['Opp'], data['Tm']], ignore_index=True).dropna()
        )
    )
    data['opp_team'] = _encode_team_codes(data['Opp'], team_names)
    data['team'] = _encode_team_codes(data['Tm'], team_names)

    # Create home/away indicator - Away column contains team names, so if Away == Tm, it's away
    data['is_home'] = (data['Away'] != data['Tm']).astype(int)

    # Position encoding
    data['pos_id'] = data['Position'].map(POSITION_ID_MAP).fillna(-1).astype(int)

    # Calculate seven game rolling average
    num_day_roll_avg = 7
    data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].transform(
        lambda x: x.rolling(num_day_roll_avg, min_periods=num_day_roll_avg).mean()
    )

    # Rank based on the 7-game average
    ranks = data.groupby(['Name', 'Season'])['7_game_avg'].rank(
        pct=False, method='average'
    )
    quartile_ranks = pd.qcut(ranks, 4, labels=False, duplicates='drop')
    data['rank'] = quartile_ranks.tolist()

    data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

    # Remove only rows with critical missing data, not all NA
    # Keep rows where we have the essential columns
    essential_cols = ['Name', 'Position', 'Season', 'FantPt', '7_game_avg']
    data = data.dropna(subset=essential_cols)

    # Convert rank to integer, handling NaN values
    data['rank'] = data['rank'].fillna(0).astype(int)

    print(f'After cleaning, data shape: {data.shape}')
    print(
        f'Season range after cleaning: {data["Season"].min()} - {data["Season"].max()}'
    )

    # Save combined dataset with consistent naming
    from ffbayes.utils.path_constants import COMBINED_DATASETS_DIR

    output_dir = str(COMBINED_DATASETS_DIR)

    # Create filename with year range for consistency
    year_range = f'{data["Season"].min()}-{data["Season"].max()}'
    output_file = os.path.join(output_dir, f'{year_range}season_modern.csv')

    data.to_csv(output_file, index=False)
    print(f'Combined dataset saved to: {output_file}')
    print(f'Shape: {data.shape}')

    return data, team_names


def main(args=None):
    """Main preprocessing function with standardized interface."""
    from ffbayes.utils.script_interface import create_standardized_interface

    interface = create_standardized_interface(
        'ffbayes-preprocess',
        'Data preprocessing for analysis with standardized interface',
    )

    # Parse arguments
    if args is None:
        args = interface.parse_arguments()

    # Add data-specific arguments
    parser = interface.setup_argument_parser()
    parser = interface.add_data_arguments(parser)
    args = parser.parse_args()

    # Set up logging
    logger = interface.setup_logging(args)

    # Get data directory
    data_dir = interface.get_data_directory(args)
    logger.info(f'Using data directory: {data_dir}')

    # Check for quick test mode
    if args.quick_test:
        logger.warning(
            '⚠️  QUICK_TEST mode explicitly enabled - processing limited data'
        )
        logger.warning('⚠️  This will not provide production-quality results')

    # Preprocess data for analysis
    data, team_names = interface.handle_errors(create_analysis_dataset, str(data_dir))

    logger.info('Preprocessing completed!')
    logger.info(f'Final dataset: {data.shape}')
    logger.info(f'Teams: {len(team_names)}')

    interface.log_completion('Data preprocessing completed successfully')

    return data, team_names


if __name__ == '__main__':
    main()
