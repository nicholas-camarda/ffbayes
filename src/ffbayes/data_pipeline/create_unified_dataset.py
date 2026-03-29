#!/usr/bin/env python3
"""
create_unified_dataset.py - Unified Data Pipeline
Single source of truth for all fantasy football data loading, validation, and cleaning.

This script:
1. SCRAPES ADP data from FantasyPros
2. SCRAPES projection data from FantasyPros for all positions
3. Calculates VOR rankings
4. Loads historical NFL data
5. Creates a clean, validated unified dataset

This unified dataset will be used by:
- Baseline (naive) model
- Monte Carlo model
- Hybrid MC + Bayesian model
- VOR snake draft strategy
- Any other analysis requiring fantasy football data

No side effects - only creates the unified dataset.
"""

import logging
import os
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as BS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline configuration
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
ALLOW_STALE_SEASON = os.getenv('FFBAYES_ALLOW_STALE_SEASON', 'false').lower() == 'true'


def _normalize_player_name(value) -> str:
    """Normalize player names for deterministic matching."""
    if pd.isna(value):
        return ''

    name = ' '.join(str(value).strip().split()).lower()
    for suffix in [' jr.', ' sr.', ' iii', ' iv', ' v', ' ii', ' i']:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _normalize_match_position(value) -> str:
    """Normalize positions so common aliases compare consistently."""
    position = ' '.join(str(value).strip().upper().split())
    if position in {'D/ST', 'DEF', 'DEFENSE'}:
        return 'DST'
    if position in {'SAF', 'S'}:
        return 'S'
    if position.startswith('RB'):
        return 'RB'
    if position.startswith('WR'):
        return 'WR'
    if position.startswith('QB'):
        return 'QB'
    if position.startswith('TE'):
        return 'TE'
    if position.startswith('K'):
        return 'K'
    return position


def _resolve_existing_vor_csv(current_year: int) -> Path | None:
    """Find an existing current-year VOR CSV in the canonical runtime locations."""
    from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
    from ffbayes.utils.vor_filename_generator import get_vor_csv_filename

    filename = get_vor_csv_filename(current_year)
    from ffbayes.utils.path_constants import get_vor_strategy_dir

    legacy_runtime_root = Path.home() / 'ProjectsRuntime' / 'ffbayes'
    legacy_processed_dir = legacy_runtime_root / 'data' / 'processed' / 'snake_draft_datasets'
    legacy_organized_dir = (
        legacy_runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'results' / 'vor_strategy'
    )

    candidates = [
        SNAKE_DRAFT_DATASETS_DIR / filename,
        get_vor_strategy_dir(current_year) / filename,
        legacy_processed_dir / filename,
        legacy_organized_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    wildcard_matches = []
    for directory in [
        SNAKE_DRAFT_DATASETS_DIR,
        get_vor_strategy_dir(current_year),
        legacy_processed_dir,
        legacy_organized_dir,
    ]:
        wildcard_matches.extend(
            sorted(directory.glob(f'snake-draft_ppr-*_{current_year}.csv'))
        )
    if wildcard_matches:
        return max(wildcard_matches, key=lambda path: path.stat().st_mtime)

    return None


def _load_vor_source(current_year: int, config: Dict) -> tuple[pd.DataFrame, Path, bool]:
    """Load the current-year market/VOR snapshot or build it if absent."""
    existing_vor = _resolve_existing_vor_csv(current_year)
    if existing_vor is not None:
        logger.info('🔄 Reusing current-year VOR snapshot: %s', existing_vor)
        return pd.read_csv(existing_vor), existing_vor, False

    logger.info('🔄 No current-year VOR snapshot found; rebuilding it now...')
    projection_data = scrape_projection_data(config)
    if projection_data is None:
        raise RuntimeError('Failed to scrape projection data from FantasyPros')

    vor_data = calculate_vor_rankings(projection_data, config)
    vor_file_path = Path(save_vor_data(vor_data, config))
    return vor_data, vor_file_path, True


def _build_player_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse the historical table to one row per player-position snapshot."""
    if data is None or data.empty:
        return pd.DataFrame(columns=['Name', 'Position'])

    snapshot = data.copy()
    snapshot['Name'] = snapshot['Name'].map(_normalize_player_name)
    snapshot['Position'] = snapshot['Position'].map(_normalize_match_position)
    sort_columns = [
        column
        for column in ['Season', 'G#', 'Date', 'source_updated_at']
        if column in snapshot.columns
    ]
    for column in ['Season', 'G#']:
        if column in snapshot.columns:
            snapshot[column] = pd.to_numeric(snapshot[column], errors='coerce')
    if 'Date' in snapshot.columns:
        snapshot['Date'] = pd.to_datetime(snapshot['Date'], errors='coerce')
    if 'source_updated_at' in snapshot.columns:
        snapshot['source_updated_at'] = pd.to_datetime(
            snapshot['source_updated_at'], errors='coerce'
        )
    if sort_columns:
        snapshot = snapshot.sort_values(sort_columns)
    snapshot = snapshot.drop_duplicates(subset=['Name', 'Position'], keep='last')
    return snapshot.reset_index(drop=True)


def _prepare_market_snapshot(vor_df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Normalize the current-year market snapshot into a player-level table."""
    if vor_df is None or vor_df.empty:
        return pd.DataFrame(
            columns=[
                'Name',
                'Position',
                'player_name',
                'position',
                'current_vor_value',
                'current_vor_rank',
            ]
        )

    snapshot = vor_df.copy()
    rename_map = {
        'PLAYER': 'player_name',
        'POS': 'position',
        'AVG': 'adp',
        'FPTS': 'proj_points_mean',
        'VOR': 'current_vor_value',
        'VALUERANK': 'current_vor_rank',
    }
    snapshot = snapshot.rename(columns=rename_map)
    snapshot['player_name'] = snapshot['player_name'].map(_normalize_player_name)
    snapshot['position'] = snapshot['position'].map(_normalize_match_position)
    snapshot['Name'] = snapshot['player_name']
    snapshot['Position'] = snapshot['position']
    snapshot['current_vor_value'] = pd.to_numeric(
        snapshot.get('current_vor_value'), errors='coerce'
    )
    snapshot['current_vor_rank'] = pd.to_numeric(
        snapshot.get('current_vor_rank'), errors='coerce'
    )
    snapshot['adp'] = pd.to_numeric(snapshot.get('adp'), errors='coerce')
    snapshot['proj_points_mean'] = pd.to_numeric(
        snapshot.get('proj_points_mean'), errors='coerce'
    )
    snapshot['vor_value'] = snapshot['current_vor_value']
    snapshot['vor_global_rank'] = snapshot['current_vor_rank']
    snapshot['vor_match_confidence'] = 1.0
    snapshot['source_name'] = f'current_market_snapshot_{current_year}'
    snapshot['source_updated_at'] = pd.Timestamp.now()
    snapshot['current_vor'] = snapshot['current_vor_value']
    snapshot['current_vor_rank_label'] = snapshot['current_vor_rank']
    return snapshot.drop_duplicates(subset=['Name', 'Position'], keep='last')


def _attach_market_snapshot(data: pd.DataFrame, market_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Attach market snapshot columns to the historical table using unique players."""
    if data is None or data.empty or market_snapshot is None or market_snapshot.empty:
        return data.copy()

    df = data.copy()
    df['__match_name'] = df['Name'].map(_normalize_player_name)
    df['__match_position'] = df['Position'].map(_normalize_match_position)

    snapshot = _build_player_snapshot(df)
    if snapshot.empty:
        return df.drop(columns=['__match_name', '__match_position'], errors='ignore')
    snapshot['__match_name'] = snapshot['Name']
    snapshot['__match_position'] = snapshot['Position']

    market = market_snapshot.copy()
    market['__match_name'] = market['Name'].map(_normalize_player_name)
    market['__match_position'] = market['Position'].map(_normalize_match_position)
    market = market.drop_duplicates(subset=['__match_name', '__match_position'], keep='last')

    match_columns = [
        column
        for column in [
            'player_name',
            'position',
            'adp',
            'proj_points_mean',
            'current_vor_value',
            'current_vor_rank',
            'vor_value',
            'vor_global_rank',
            'vor_match_confidence',
            'source_name',
            'source_updated_at',
            'current_vor',
            'current_vor_rank_label',
        ]
        if column in market.columns
    ]

    merged = snapshot.merge(
        market[['__match_name', '__match_position', *match_columns]],
        left_on=['Name', 'Position'],
        right_on=['__match_name', '__match_position'],
        how='left',
        suffixes=('', '_market'),
    )

    unmatched = merged[merged['current_vor_rank'].isna()].copy()
    if not unmatched.empty:
        from fuzzywuzzy import fuzz

        market_index = {}
        for position, group in market.groupby('__match_position'):
            normalized_names = [
                (idx, _normalize_player_name(name))
                for idx, name in enumerate(group['player_name'].tolist())
            ]
            market_index[position] = {
                'frame': group.reset_index(drop=True),
                'normalized_names': normalized_names,
            }

        cache: dict[tuple[str, str], tuple[int, int] | None] = {}
        for snapshot_idx, row in unmatched.iterrows():
            key = (row['Name'], row['Position'])
            if key in cache:
                match_idx, score = cache[key] or (None, None)
            else:
                best_score = 0
                best_idx = None
                position_bucket = market_index.get(row['Position'])
                if position_bucket is not None:
                    for idx, candidate_name in position_bucket['normalized_names']:
                        scores = [
                            fuzz.ratio(row['Name'], candidate_name),
                            fuzz.partial_ratio(row['Name'], candidate_name),
                            fuzz.token_sort_ratio(row['Name'], candidate_name),
                            fuzz.token_set_ratio(row['Name'], candidate_name),
                        ]
                        score = max(scores)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                cache[key] = (best_idx, best_score) if best_idx is not None else None
                match_idx, score = cache[key] or (None, None)

            if match_idx is None:
                continue

            candidate = market_index[row['Position']]['frame'].iloc[match_idx]
            for column in match_columns:
                if column in merged.columns and column in candidate.index:
                    merged.at[snapshot_idx, column] = candidate[column]
            merged.at[snapshot_idx, 'vor_match_confidence'] = (
                float(score) / 100.0 if score is not None else np.nan
            )
            merged.at[snapshot_idx, 'current_vor_value'] = candidate.get(
                'current_vor_value', np.nan
            )
            merged.at[snapshot_idx, 'current_vor_rank'] = candidate.get(
                'current_vor_rank', np.nan
            )
            merged.at[snapshot_idx, 'vor_value'] = candidate.get(
                'vor_value', np.nan
            )
            merged.at[snapshot_idx, 'vor_global_rank'] = candidate.get(
                'vor_global_rank', np.nan
            )
            merged.at[snapshot_idx, 'current_vor'] = candidate.get(
                'current_vor', np.nan
            )

    lookup = merged[
        [
            '__match_name',
            '__match_position',
            *[
                column
                for column in [
                    'player_name',
                    'position',
                    'adp',
                    'proj_points_mean',
                    'current_vor_value',
                    'current_vor_rank',
                    'vor_value',
                    'vor_global_rank',
                    'vor_match_confidence',
                    'source_name',
                    'source_updated_at',
                    'current_vor',
                    'current_vor_rank_label',
                ]
                if column in merged.columns
            ],
        ]
    ].drop_duplicates(subset=['__match_name', '__match_position'], keep='last')

    df = df.merge(
        lookup,
        on=['__match_name', '__match_position'],
        how='left',
        suffixes=('', '_market'),
    )
    for column in ['player_name', 'position', 'adp', 'proj_points_mean', 'current_vor_value', 'current_vor_rank', 'vor_value', 'vor_global_rank', 'vor_match_confidence', 'source_name', 'source_updated_at', 'current_vor', 'current_vor_rank_label']:
        market_column = f'{column}_market'
        if market_column in df.columns:
            if column in df.columns:
                df[column] = df[column].combine_first(df[market_column])
                df = df.drop(columns=[market_column])
            else:
                df[column] = df[market_column]
                df = df.drop(columns=[market_column])

    if 'vor_value' not in df.columns:
        df['vor_value'] = np.nan
    if 'vor_global_rank' not in df.columns:
        df['vor_global_rank'] = np.nan
    if 'vor_match_confidence' not in df.columns:
        df['vor_match_confidence'] = np.nan
    if 'current_vor' not in df.columns:
        df['current_vor'] = df['vor_value']
    if 'current_vor_rank' not in df.columns:
        df['current_vor_rank'] = df['vor_global_rank']
    return df.drop(columns=['__match_name', '__match_position'], errors='ignore')


def load_config() -> Dict:
    """Load VOR configuration from centralized config."""
    try:
        from ffbayes.utils.config_loader import get_config
        from ffbayes.utils.training_config import VOR_CONFIG

        config_loader = get_config()
        config = VOR_CONFIG.copy()
        config.update(
            {
                'ppr': config_loader.get_vor_setting('ppr'),
                'top_rank': config_loader.get_vor_setting('top_rank'),
            }
        )
        logger.info(
            f'VOR Configuration: PPR={config["ppr"]}, Top Rank={config["top_rank"]}'
        )
        return config
    except ImportError:
        # Fallback to environment variables
        from ffbayes.utils.training_config import VOR_CONFIG

        config = VOR_CONFIG.copy()
        config.update(
            {
                'ppr': float(os.getenv('VOR_PPR', '0.5')),
                'top_rank': int(os.getenv('VOR_TOP_RANK', '120')),
            }
        )
        logger.info(
            f'VOR Configuration: PPR={config["ppr"]}, Top Rank={config["top_rank"]}'
        )
        return config


def scrape_adp_data(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape ADP data from FantasyPros.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with ADP data or None if failed
    """
    logger.info('🔄 Scraping ADP data from FantasyPros...')

    try:
        res = requests.get(config['base_url_adp'], timeout=30)
        res.raise_for_status()

        soup = BS(res.content, 'html.parser')
        table = soup.find('table', {'id': 'data'})

        if not table:
            logger.error('Could not find ADP table on page')
            return None

        df = pd.read_html(StringIO(str(table)))[0]
        logger.info(f'Raw ADP data: {len(df)} players')

        # Filter and clean data
        df = df[['Player Team (Bye)', 'POS', 'AVG']]
        df['PLAYER'] = df['Player Team (Bye)'].apply(lambda x: ' '.join(x.split()[:-2]))
        df['POS'] = df['POS'].apply(lambda x: x[:2])
        df = df[['PLAYER', 'POS', 'AVG']].sort_values(by='AVG')

        logger.info(f'✅ Processed ADP data: {len(df)} players')
        return df

    except requests.RequestException as e:
        logger.error(f'Failed to fetch ADP data: {e}')
        return None
    except Exception as e:
        logger.error(f'Error processing ADP data: {e}')
        return None


def scrape_projection_data(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape projection data for all positions from FantasyPros.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with projection data or None if failed
    """
    logger.info('🔄 Scraping projection data from FantasyPros for all positions...')

    final_df = pd.DataFrame()

    for position in config['positions']:
        try:
            url = config['base_url_projections'].format(position=position)
            logger.info(f'   Fetching {position.upper()} projections...')

            res = requests.get(url, timeout=30)
            res.raise_for_status()

            soup = BS(res.content, 'html.parser')
            table = soup.find('table', {'id': 'data'})

            if not table:
                logger.warning(f'Could not find projection table for {position}')
                continue

            df = pd.read_html(StringIO(str(table)))[0]

            # Clean column names
            df.columns = df.columns.droplevel(level=0)
            df['PLAYER'] = df['Player'].apply(lambda x: ' '.join(x.split()[:-1]))

            # Adjust for PPR scoring
            if 'REC' in df.columns:
                df['FPTS'] = df['FPTS'] + config['ppr'] * df['REC']

            df['POS'] = position.upper()
            df = df[['PLAYER', 'POS', 'FPTS']]

            final_df = pd.concat([final_df, df])
            logger.info(f'   ✅ Added {len(df)} {position.upper()} players')

        except requests.RequestException as e:
            logger.error(f'Failed to fetch {position} projections: {e}')
        except Exception as e:
            logger.error(f'Error processing {position} projections: {e}')

    if final_df.empty:
        logger.error('No projection data could be loaded')
        return None

    final_df = final_df.sort_values(by='FPTS', ascending=False)
    logger.info(f'✅ Total projection data: {len(final_df)} players')
    return final_df


def calculate_vor_rankings(projection_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate Value Over Replacement for each player.

    Args:
        projection_df: DataFrame with projection data
        config: Configuration dictionary

    Returns:
        DataFrame with VOR calculations
    """
    logger.info('🔄 Calculating Value Over Replacement...')

    # Define replacement players (last starter in each position)
    replacement_players = {
        'RB': 'RB12',  # 12th RB (last starter in 12-team league)
        'WR': 'WR24',  # 24th WR (last starter in 12-team league)
        'TE': 'TE12',  # 12th TE (last starter in 12-team league)
        'QB': 'QB12',  # 12th QB (last starter in 12-team league)
    }

    # Get replacement values
    replacement_values = {}
    for position, rank_name in replacement_players.items():
        pos_data = projection_df[projection_df['POS'] == position].copy()
        if len(pos_data) >= 12:
            # Get the 12th player (index 11)
            replacement_values[position] = pos_data.iloc[11]['FPTS']
            replacement_player = pos_data.iloc[11]['PLAYER']
            logger.info(
                f'   Replacement {position}: {replacement_player} = {replacement_values[position]:.1f} points'
            )
        else:
            logger.warning(f'Not enough {position} players for replacement calculation')
            replacement_values[position] = 0

    # Calculate VOR
    projection_df['VOR'] = projection_df.apply(
        lambda row: row['FPTS'] - replacement_values.get(row['POS'], 0), axis=1
    )

    projection_df = projection_df.sort_values(by='VOR', ascending=False)
    projection_df['VALUERANK'] = projection_df['VOR'].rank(ascending=False)

    logger.info(
        f'✅ VOR calculation complete. Top player: {projection_df.iloc[0]["PLAYER"]} (VOR: {projection_df.iloc[0]["VOR"]:.1f})'
    )
    return projection_df


def save_vor_data(vor_df: pd.DataFrame, config: Dict) -> str:
    """
    Save VOR data to the snake draft datasets directory.

    Args:
        vor_df: DataFrame with VOR calculations
        config: Configuration dictionary

    Returns:
        Path to saved file
    """
    current_year = datetime.now().year

    # Use path constants for output directory
    from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR

    output_dir = SNAKE_DRAFT_DATASETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save VOR data
    filename = f'snake-draft_ppr-{config["ppr"]}_vor_top-{config["top_rank"]}_{current_year}.csv'
    output_path = output_dir / filename

    vor_df.to_csv(output_path, index=False)
    logger.info(f'✅ VOR data saved to: {output_path}')

    return str(output_path)


def load_combined_dataset(data_directory=None):
    """Load the most recent combined dataset from the canonical data directory."""
    logger.info('🔄 Loading combined dataset...')
    from ffbayes.utils.analysis_windows import (
        build_freshness_manifest,
        resolve_analysis_window,
        write_freshness_manifest,
    )
    from ffbayes.utils.path_constants import COMBINED_DATASETS_DIR, RAW_DATA_DIR

    if data_directory in (None, '', 'datasets'):
        output_dir = COMBINED_DATASETS_DIR
    else:
        data_root = Path(data_directory).expanduser()
        output_dir = data_root / 'combined_datasets'

    analysis_files = list(output_dir.glob('*season_modern.csv'))

    if not analysis_files:
        raise ValueError(
            f'No combined datasets found in {output_dir}. Run preprocessing first.'
        )

    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    logger.info(f'📊 Loading: {latest_file}')

    data = pd.read_csv(latest_file)
    logger.info(f'✅ Loaded: {data.shape}')
    logger.info(f'✅ Years: {sorted(data["Season"].unique())}')

    freshness_window = resolve_analysis_window(
        [
            int(year)
            for year in pd.Series(data['Season']).dropna().astype(int).unique().tolist()
        ],
        allow_stale=True,
    )
    freshness_manifest = build_freshness_manifest(
        freshness_window,
        source_name='combined_datasets',
        source_path=latest_file,
        found_files=[latest_file],
    )
    try:
        write_freshness_manifest(
            freshness_manifest, RAW_DATA_DIR / 'unified_dataset_freshness.json'
        )
    except PermissionError as exc:
        logger.warning(
            'Skipping unified dataset freshness manifest write: %s', exc
        )

    if (
        not ALLOW_STALE_SEASON
        and freshness_window.latest_expected_year is not None
        and freshness_window.latest_expected_year not in freshness_window.found_years
    ):
        raise RuntimeError(
            f'Missing latest expected season {freshness_window.latest_expected_year} in combined dataset.'
        )

    return data


def validate_data_quality(data):
    """Validate data quality and break immediately if issues found."""
    logger.info('🔍 Validating data quality...')

    # Check required columns
    required_columns = ['Name', 'Position', 'FantPt', 'Season', 'G#', 'Opp', 'Tm']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    # Check for missing values in critical columns
    critical_columns = ['Name', 'Position', 'FantPt', 'Season']
    for col in critical_columns:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            raise ValueError(
                f'Column {col} has {missing_count} missing values. Fix data quality first.'
            )

    # Check data types
    if not pd.api.types.is_numeric_dtype(data['FantPt']):
        raise ValueError('FantPt column is not numeric. Fix data quality first.')

    if not pd.api.types.is_numeric_dtype(data['Season']):
        raise ValueError('Season column is not numeric. Fix data quality first.')

    logger.info('✅ Data validation passed')


def clean_data_types(data):
    """Clean data types without any fallbacks - break if issues found."""
    logger.info('🧹 Cleaning data types...')

    # Create copy to avoid modifying original
    data = data.copy()

    # Ensure numeric columns are properly typed
    numeric_columns = ['FantPt', 'G#', 'Season']
    for col in numeric_columns:
        if col in data.columns:
            # Check for non-numeric values
            non_numeric = (
                data[col]
                .apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float)))
                .sum()
            )
            if non_numeric > 0:
                raise ValueError(
                    f'Column {col} contains {non_numeric} non-numeric values. Fix data quality first.'
                )

            # Convert to proper numeric type
            data[col] = pd.to_numeric(data[col])
            logger.info(f'   ✅ {col}: {data[col].dtype}')

    # Handle is_home column if it exists
    if 'is_home' in data.columns:
        # Check if it's already numeric (0/1) or boolean
        if pd.api.types.is_numeric_dtype(data['is_home']):
            # Check if it's binary (0/1)
            unique_values = set(data['is_home'].unique())
            if unique_values.issubset({0, 1}):
                logger.info(
                    f'   ✅ is_home: Already binary (0/1) - {data["is_home"].dtype}'
                )
            else:
                raise ValueError(
                    f'Column is_home contains non-binary values: {unique_values}. Fix data quality first.'
                )
        elif pd.api.types.is_bool_dtype(data['is_home']):
            logger.info(f'   ✅ is_home: Already boolean - {data["is_home"].dtype}')
        else:
            # Check for non-numeric/non-boolean values
            non_binary = (
                data['is_home']
                .apply(lambda x: not pd.isna(x) and x not in [0, 1, True, False])
                .sum()
            )
            if non_binary > 0:
                raise ValueError(
                    'Column is_home contains non-binary values. Fix data quality first.'
                )

            # Convert to binary
            data['is_home'] = (data['is_home'] == 1).astype(int)
            logger.info('   ✅ is_home: Converted to binary (0/1)')

    logger.info(f'✅ Data type cleaning complete. Shape: {data.shape}')
    return data


def add_vor_rankings(data, vor_file_path):
    """Add VOR rankings to the dataset using the newly scraped data."""
    logger.info('🔄 Adding VOR rankings to dataset...')

    try:
        # Load VOR data
        ranking_data = pd.read_csv(vor_file_path)
        logger.info(f'   📊 Loaded VOR data: {len(ranking_data)} players')

        # Import fuzzy matching
        from fuzzywuzzy import fuzz

        def normalize_name(name):
            """Normalize player names for better matching."""
            if pd.isna(name):
                return ''

            # Convert to string and clean
            name = str(name).strip()

            # Remove common suffixes that cause mismatches
            suffixes = [' Jr.', ' Sr.', ' III', ' IV', ' V', ' II', ' I']
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break

            # Remove extra spaces and convert to lowercase
            name = ' '.join(name.split()).lower()

            return name

        def find_best_match(player_name, player_pos, vor_data, threshold=85):
            """Find best VOR match using fuzzy string matching."""
            normalized_name = normalize_name(player_name)

            if not normalized_name:
                return None

            best_match = None
            best_score = 0

            # Filter VOR data by position first for efficiency
            pos_matches = vor_data[vor_data['POS'] == player_pos]

            for _, vor_row in pos_matches.iterrows():
                vor_name = vor_row['PLAYER']
                vor_pos = vor_row['POS']

                # Position must match exactly
                if vor_pos != player_pos:
                    continue

                # Calculate fuzzy match score
                normalized_vor_name = normalize_name(vor_name)

                # Try different matching strategies
                scores = [
                    fuzz.ratio(normalized_name, normalized_vor_name),
                    fuzz.partial_ratio(normalized_name, normalized_vor_name),
                    fuzz.token_sort_ratio(normalized_name, normalized_vor_name),
                    fuzz.token_set_ratio(normalized_name, normalized_vor_name),
                ]

                # Use the best score
                score = max(scores)

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = vor_row

            return best_match, best_score

        logger.info('   🔍 Performing fuzzy name matching...')

        # Initialize VOR ranking/values columns
        data['vor_global_rank'] = 121  # Default worst rank
        data['vor_match_confidence'] = 0.0  # Track match quality
        if 'vor_value' not in data.columns:
            data['vor_value'] = np.nan

        # Track matching statistics
        exact_matches = 0
        fuzzy_matches = 0
        no_matches = 0

        # Process each unique player in the data
        unique_players = data[['Name', 'Position']].drop_duplicates()

        for _, player_row in unique_players.iterrows():
            player_name = player_row['Name']
            player_pos = player_row['Position']

            # Find best VOR match
            match_result = find_best_match(player_name, player_pos, ranking_data)

            if match_result is None:
                no_matches += 1
                continue

            vor_match, confidence = match_result

            if vor_match is not None:
                # Update all rows for this player
                player_mask = (data['Name'] == player_name) & (
                    data['Position'] == player_pos
                )

                if confidence >= 95:  # High confidence match
                    exact_matches += 1
                    data.loc[player_mask, 'vor_global_rank'] = int(
                        vor_match['VALUERANK']
                    )
                    data.loc[player_mask, 'vor_match_confidence'] = confidence / 100.0
                    if 'VOR' in vor_match:
                        try:
                            data.loc[player_mask, 'vor_value'] = float(vor_match['VOR'])
                        except Exception:
                            pass
                elif confidence >= 85:  # Medium confidence match
                    fuzzy_matches += 1
                    data.loc[player_mask, 'vor_global_rank'] = int(
                        vor_match['VALUERANK']
                    )
                    data.loc[player_mask, 'vor_match_confidence'] = confidence / 100.0
                    if 'VOR' in vor_match:
                        try:
                            data.loc[player_mask, 'vor_value'] = float(vor_match['VOR'])
                        except Exception:
                            pass
                else:
                    no_matches += 1

        # Print matching statistics
        total_players = len(unique_players)
        logger.info(f'   ✅ Exact matches (95%+): {exact_matches}')
        logger.info(f'   ✅ Fuzzy matches (85%+): {fuzzy_matches}')
        logger.info(f'   ⚠️  No matches: {no_matches}')
        logger.info(
            f'   📊 Match rate: {((exact_matches + fuzzy_matches) / total_players * 100):.1f}%'
        )

        # Final VOR ranking statistics
        valid_ranks = data[data['vor_global_rank'] != 121]['vor_global_rank']
        if len(valid_ranks) > 0:
            logger.info(
                f'   ✅ VOR ranking range: {valid_ranks.min()}-{valid_ranks.max()}'
            )
            logger.info(f'   ✅ Players with VOR rankings: {len(valid_ranks)}')

        return data

    except Exception as e:
        logger.error(f'   ❌ Error adding VOR ranking: {e}')
        data['vor_global_rank'] = 121  # Default worst rank
        data['vor_match_confidence'] = 0.0
        return data


def add_predraft_composites(data, risk_tolerance: str = 'medium'):
    """Compute and embed pre-draft composite features into unified dataset.

    - Uses historical per-game data to compute player-season metrics, collapsed to player-level
      by taking the latest available season per player.
    - Uses VOR numeric values (vor_value) when available to compute RAV and tier cliffs.
    """
    logger.info(
        '🔄 Adding pre-draft composite features (RAV, tier cliffs, role strength, momentum)...'
    )
    from ffbayes.analysis.advanced_stats_calculator import (
        calculate_advanced_stats_from_existing_data,
    )

    alpha_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75}
    alpha = alpha_map.get(str(risk_tolerance).lower(), 0.5)

    df = data.copy()

    # Build advanced stats per game, then collapse to latest player-season
    try:
        enriched = calculate_advanced_stats_from_existing_data(df)
    except Exception as e:
        logger.warning(
            f'Advanced stats calculation failed; continuing with minimal features: {e}'
        )
        enriched = df.copy()
        for c in [
            'consistency_score',
            'floor_ceiling_spread',
            'team_usage_pct',
            'recent_form',
            'season_trend',
        ]:
            if c not in enriched.columns:
                enriched[c] = np.nan

    # Aggregate per player-season
    agg = (
        enriched.groupby(['Name', 'Position', 'Season'])
        .agg(
            consistency_score=('consistency_score', 'mean'),
            floor_ceiling_spread=('floor_ceiling_spread', 'mean'),
            team_usage_pct=('team_usage_pct', 'mean'),
            recent_form=('recent_form', 'mean'),
            season_trend=('season_trend', 'mean'),
        )
        .reset_index()
    )

    # Take latest season metrics per player
    latest_idx = agg.groupby('Name')['Season'].transform('max') == agg['Season']
    latest = agg[latest_idx].drop(columns=['Season']).copy()
    latest = latest.rename(
        columns={
            'consistency_score': 'consistency_score_latest',
            'floor_ceiling_spread': 'floor_ceiling_spread_latest',
            'team_usage_pct': 'team_usage_pct_latest',
            'recent_form': 'recent_form_latest',
            'season_trend': 'season_trend_latest',
        }
    )

    # Role strength z per position using latest team_usage_pct
    if 'team_usage_pct_latest' in latest.columns:
        latest['role_strength_z'] = np.nan
        for pos, sub in latest.groupby('Position'):
            vals = sub['team_usage_pct_latest']
            mu = vals.mean()
            sd = vals.std(ddof=0)
            if sd and not np.isclose(sd, 0.0):
                latest.loc[sub.index, 'role_strength_z'] = (vals - mu) / sd
            else:
                latest.loc[sub.index, 'role_strength_z'] = 0.0
    else:
        latest['role_strength_z'] = np.nan

    # Merge latest features to all rows by Name
    df = df.merge(latest, on=['Name', 'Position'], how='left')

    # Compute RAV if we have vor_value; else leave NaN
    if 'vor_value' in df.columns:
        cv = df['consistency_score_latest'].fillna(0.0)
        df['RAV'] = df['vor_value'].fillna(0.0) / (1.0 + alpha * cv)
    else:
        df['RAV'] = np.nan

    # Tier cliff distance by position based on vor_value
    df['tier_cliff_distance'] = np.nan
    if 'vor_value' in df.columns:
        # Build unique per-player vor_value table (prefer latest row per player)
        player_vor = (
            df[['Name', 'Position', 'vor_value']]
            .drop_duplicates()
            .dropna(subset=['vor_value'])
        )
        for pos, sub in player_vor.groupby('Position'):
            sub_sorted = sub.sort_values('vor_value', ascending=False)
            idx = sub_sorted['Name'].tolist()
            values = sub_sorted['vor_value'].values
            if len(values) > 1:
                diffs = values[:-1] - values[1:]
                cliff_map = {idx[i]: diffs[i] for i in range(len(diffs))}
                # Last in rank has NaN
                cliff_map[idx[-1]] = np.nan
                mask = df['Name'].isin(cliff_map.keys()) & (df['Position'] == pos)
                df.loc[mask, 'tier_cliff_distance'] = df.loc[mask, 'Name'].map(
                    cliff_map
                )

    return df


def calculate_basic_features(data):
    """Calculate basic features needed by all models."""
    logger.info('📊 Calculating basic features...')

    # Create copy to avoid modifying original
    data = data.copy()

    # Calculate 7-game average (if not already present)
    if '7_game_avg' not in data.columns:
        logger.info('   Calculating 7-game averages...')
        data = data.sort_values(['Name', 'Season', 'G#'])
        data['7_game_avg'] = (
            data.groupby(['Name', 'Season'])['FantPt']
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    # Calculate basic position indicators
    for pos in ['QB', 'RB', 'WR', 'TE']:
        data[f'is_{pos.lower()}'] = (data['Position'] == pos).astype(int)

    # Calculate basic team/opponent features
    data['opp_idx'] = pd.factorize(data['Opp'])[0]

    logger.info(f'✅ Basic features calculated. Shape: {data.shape}')
    return data


def add_adp_data(data: pd.DataFrame, adp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ADP into unified dataset using fuzzy matching, per player-position.

    Adds columns: adp (float), adp_rank (int), adp_match_confidence (0..1)
    """
    logger.info('🔄 Adding ADP data to unified dataset...')
    df = data.copy()
    try:
        from fuzzywuzzy import fuzz
    except Exception as e:
        logger.error(f'Fuzzy matching unavailable: {e}')
        raise

    # Prepare ADP frame
    adp = adp_df.copy()
    if 'AVG' in adp.columns:
        adp['adp'] = pd.to_numeric(adp['AVG'], errors='coerce')
    else:
        # If schema differs, attempt to find plausible ADP column
        candidates = [c for c in adp.columns if c.lower() in ('avg', 'adp')]
        adp['adp'] = (
            pd.to_numeric(adp[candidates[0]], errors='coerce') if candidates else np.nan
        )
    adp = adp.dropna(subset=['adp'])
    adp['adp_rank'] = adp['adp'].rank(method='first')

    def normalize_name(name):
        if pd.isna(name):
            return ''
        name = str(name).strip()
        suffixes = [' Jr.', ' Sr.', ' III', ' IV', ' V', ' II', ' I']
        for s in suffixes:
            if name.endswith(s):
                name = name[: -len(s)]
                break
        return ' '.join(name.split()).lower()

    def find_best_match(player_name, player_pos, adp_data, threshold=85):
        norm_name = normalize_name(player_name)
        if not norm_name:
            return None
        best = None
        best_score = 0
        pos_col = (
            'POS'
            if 'POS' in adp_data.columns
            else ('position' if 'position' in adp_data.columns else None)
        )
        pos_matches = (
            adp_data if pos_col is None else adp_data[adp_data[pos_col] == player_pos]
        )
        for _, row in pos_matches.iterrows():
            vor_name = normalize_name(row.get('PLAYER', row.get('player', '')))
            scores = [
                fuzz.ratio(norm_name, vor_name),
                fuzz.partial_ratio(norm_name, vor_name),
                fuzz.token_sort_ratio(norm_name, vor_name),
                fuzz.token_set_ratio(norm_name, vor_name),
            ]
            score = max(scores)
            if score > best_score and score >= threshold:
                best_score = score
                best = row
        return (best, best_score) if best is not None else None

    # Initialize columns
    if 'adp' not in df.columns:
        df['adp'] = np.nan
    if 'adp_rank' not in df.columns:
        df['adp_rank'] = np.nan
    if 'adp_match_confidence' not in df.columns:
        df['adp_match_confidence'] = 0.0

    unique_players = df[['Name', 'Position']].drop_duplicates()
    exact = fuzzy = missing = 0
    for _, r in unique_players.iterrows():
        name = r['Name']
        pos = r['Position']
        match = find_best_match(name, pos, adp)
        if match is None:
            missing += 1
            continue
        m, conf = match
        mask = (df['Name'] == name) & (df['Position'] == pos)
        try:
            df.loc[mask, 'adp'] = float(m['adp'])
            df.loc[mask, 'adp_rank'] = int(m['adp_rank'])
            df.loc[mask, 'adp_match_confidence'] = conf / 100.0
            if conf >= 95:
                exact += 1
            else:
                fuzzy += 1
        except Exception:
            missing += 1

    total = len(unique_players)
    logger.info(
        f'   ✅ ADP matches exact: {exact}, fuzzy: {fuzzy}, missing: {missing}, rate: {((exact + fuzzy) / max(1, total)) * 100:.1f}%'
    )
    return df


def create_unified_dataset(data_directory=None):
    """Create unified, clean dataset for all models."""
    logger.info('=' * 60)
    logger.info('Creating Unified Fantasy Football Dataset')
    logger.info('=' * 60)

    try:
        current_year = datetime.now().year
        config = load_config()

        phase_start = time.perf_counter()
        data = load_combined_dataset(data_directory)
        logger.info(
            'Phase load_combined_dataset completed in %.1fs with %d rows',
            time.perf_counter() - phase_start,
            len(data),
        )

        phase_start = time.perf_counter()
        validate_data_quality(data)
        data = clean_data_types(data)
        logger.info(
            'Phase validate_and_clean completed in %.1fs',
            time.perf_counter() - phase_start,
        )

        phase_start = time.perf_counter()
        vor_data, vor_file_path, rebuilt_vor = _load_vor_source(current_year, config)
        market_snapshot = _prepare_market_snapshot(vor_data, current_year)
        logger.info(
            'Phase attach_current_market_snapshot prepared %d market rows in %.1fs (%s)',
            len(market_snapshot),
            time.perf_counter() - phase_start,
            'rebuilt' if rebuilt_vor else 'reused',
        )

        phase_start = time.perf_counter()
        data = _attach_market_snapshot(data, market_snapshot)
        logger.info(
            'Phase attach_current_market_snapshot merged snapshot onto %d rows in %.1fs',
            len(data),
            time.perf_counter() - phase_start,
        )

        phase_start = time.perf_counter()
        data = calculate_basic_features(data)
        logger.info(
            'Phase calculate_basic_features completed in %.1fs',
            time.perf_counter() - phase_start,
        )

        phase_start = time.perf_counter()
        risk_tol = os.getenv('RISK_TOLERANCE', 'medium')
        data = add_predraft_composites(data, risk_tolerance=risk_tol)
        logger.info(
            'Phase compute_composites completed in %.1fs',
            time.perf_counter() - phase_start,
        )

        phase_start = time.perf_counter()
        from ffbayes.utils.path_constants import (
            get_unified_dataset_csv_path,
            get_unified_dataset_path,
        )

        output_path = get_unified_dataset_path()
        csv_path = get_unified_dataset_csv_path()
        workspace_runtime_root = Path(__file__).resolve().parents[3] / '.ffbayes_runtime'
        fallback_dir = workspace_runtime_root / 'data' / 'processed' / 'unified_dataset'
        fallback_csv_path = fallback_dir / 'unified_dataset.csv'
        fallback_json_path = fallback_dir / 'unified_dataset.json'

        write_target_csv = csv_path
        write_target_json = output_path
        try:
            data.to_csv(write_target_csv, index=False)
            data.to_json(write_target_json, orient='records')
        except PermissionError as exc:
            logger.warning(
                'Primary unified dataset paths are not writable; falling back to %s',
                fallback_dir,
            )
            fallback_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(fallback_csv_path, index=False)
            data.to_json(fallback_json_path, orient='records')
            write_target_csv = fallback_csv_path
            write_target_json = fallback_json_path
        logger.info(
            'Phase write_outputs completed in %.1fs: csv=%s json=%s',
            time.perf_counter() - phase_start,
            write_target_csv,
            write_target_json,
        )
        logger.info('✅ Unified dataset saved in compact machine-readable formats')
        logger.info('✅ Current-year VOR source: %s', vor_file_path)
        logger.info('✅ Final shape: %s', data.shape)
        logger.info('✅ Available columns: %s', list(data.columns))

        return data

    except Exception as e:
        logger.error(f'❌ Failed to create unified dataset: {e}')
        raise


def main():
    """Main function for creating unified dataset."""
    try:
        data = create_unified_dataset()
        logger.info('\n🎉 Unified dataset creation successful!')
        logger.info(
            'All models should now use this dataset instead of individual data loading.'
        )
        logger.info('VOR data has been scraped from FantasyPros and integrated.')

    except Exception as e:
        logger.error(f'\n💥 Unified dataset creation failed: {e}')
        logger.error('Fix data quality issues before proceeding.')
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
