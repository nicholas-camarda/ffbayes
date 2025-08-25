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
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup as BS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline configuration
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'



def load_config() -> Dict:
    """Load VOR configuration from centralized config."""
    try:
        from ffbayes.utils.config_loader import get_config
        from ffbayes.utils.training_config import VOR_CONFIG
        
        config_loader = get_config()
        config = VOR_CONFIG.copy()
        config.update({
            'ppr': config_loader.get_vor_setting('ppr'),
            'top_rank': config_loader.get_vor_setting('top_rank')
        })
        logger.info(f"VOR Configuration: PPR={config['ppr']}, Top Rank={config['top_rank']}")
        return config
    except ImportError:
        # Fallback to environment variables
        from ffbayes.utils.training_config import VOR_CONFIG
        config = VOR_CONFIG.copy()
        config.update({
            'ppr': float(os.getenv('VOR_PPR', '0.5')),
            'top_rank': int(os.getenv('VOR_TOP_RANK', '120'))
        })
        logger.info(f"VOR Configuration: PPR={config['ppr']}, Top Rank={config['top_rank']}")
        return config

def scrape_adp_data(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape ADP data from FantasyPros.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with ADP data or None if failed
    """
    logger.info("🔄 Scraping ADP data from FantasyPros...")
    
    try:
        res = requests.get(config['base_url_adp'], timeout=30)
        res.raise_for_status()
        
        soup = BS(res.content, 'html.parser')
        table = soup.find('table', {'id': 'data'})
        
        if not table:
            logger.error("Could not find ADP table on page")
            return None
            
        df = pd.read_html(StringIO(str(table)))[0]
        logger.info(f"Raw ADP data: {len(df)} players")
        
        # Filter and clean data
        df = df[['Player Team (Bye)', 'POS', 'AVG']]
        df['PLAYER'] = df['Player Team (Bye)'].apply(
            lambda x: ' '.join(x.split()[:-2])
        )
        df['POS'] = df['POS'].apply(lambda x: x[:2])
        df = df[['PLAYER', 'POS', 'AVG']].sort_values(by='AVG')
        
        logger.info(f"✅ Processed ADP data: {len(df)} players")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch ADP data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing ADP data: {e}")
        return None

def scrape_projection_data(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape projection data for all positions from FantasyPros.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with projection data or None if failed
    """
    logger.info("🔄 Scraping projection data from FantasyPros for all positions...")
    
    final_df = pd.DataFrame()
    
    for position in config['positions']:
        try:
            url = config['base_url_projections'].format(position=position)
            logger.info(f"   Fetching {position.upper()} projections...")
            
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            
            soup = BS(res.content, 'html.parser')
            table = soup.find('table', {'id': 'data'})
            
            if not table:
                logger.warning(f"Could not find projection table for {position}")
                continue
                
            df = pd.read_html(StringIO(str(table)))[0]
            
            # Clean column names
            df.columns = df.columns.droplevel(level=0)
            df['PLAYER'] = df['Player'].apply(
                lambda x: ' '.join(x.split()[:-1])
            )
            
            # Adjust for PPR scoring
            if 'REC' in df.columns:
                df['FPTS'] = df['FPTS'] + config['ppr'] * df['REC']
            
            df['POS'] = position.upper()
            df = df[['PLAYER', 'POS', 'FPTS']]
            
            final_df = pd.concat([final_df, df])
            logger.info(f"   ✅ Added {len(df)} {position.upper()} players")
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {position} projections: {e}")
        except Exception as e:
            logger.error(f"Error processing {position} projections: {e}")
    
    if final_df.empty:
        logger.error("No projection data could be loaded")
        return None
    
    final_df = final_df.sort_values(by='FPTS', ascending=False)
    logger.info(f"✅ Total projection data: {len(final_df)} players")
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
    logger.info("🔄 Calculating Value Over Replacement...")
    
    # Define replacement players (last starter in each position)
    replacement_players = {
        'RB': 'RB12',  # 12th RB (last starter in 12-team league)
        'WR': 'WR24',  # 24th WR (last starter in 12-team league) 
        'TE': 'TE12',  # 12th TE (last starter in 12-team league)
        'QB': 'QB12'   # 12th QB (last starter in 12-team league)
    }
    
    # Get replacement values
    replacement_values = {}
    for position, rank_name in replacement_players.items():
        pos_data = projection_df[projection_df['POS'] == position].copy()
        if len(pos_data) >= 12:
            # Get the 12th player (index 11)
            replacement_values[position] = pos_data.iloc[11]['FPTS']
            replacement_player = pos_data.iloc[11]['PLAYER']
            logger.info(f"   Replacement {position}: {replacement_player} = {replacement_values[position]:.1f} points")
        else:
            logger.warning(f"Not enough {position} players for replacement calculation")
            replacement_values[position] = 0
    
    # Calculate VOR
    projection_df['VOR'] = projection_df.apply(
        lambda row: row['FPTS'] - replacement_values.get(row['POS'], 0), 
        axis=1
    )
    
    projection_df = projection_df.sort_values(by='VOR', ascending=False)
    projection_df['VALUERANK'] = projection_df['VOR'].rank(ascending=False)
    
    logger.info(f"✅ VOR calculation complete. Top player: {projection_df.iloc[0]['PLAYER']} (VOR: {projection_df.iloc[0]['VOR']:.1f})")
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
    filename = f"snake-draft_ppr-{config['ppr']}_vor_top-{config['top_rank']}_{current_year}.csv"
    output_path = output_dir / filename
    
    vor_df.to_csv(output_path, index=False)
    logger.info(f"✅ VOR data saved to: {output_path}")
    
    return str(output_path)

def load_combined_dataset(data_directory):
    """Load the most recent combined dataset from the data directory."""
    logger.info("🔄 Loading combined dataset...")
    
    output_dir = Path(data_directory) / 'combined_datasets'
    analysis_files = list(output_dir.glob('*season_modern.csv'))
    
    if not analysis_files:
        raise ValueError(f"No combined datasets found in {output_dir}. Run preprocessing first.")
    
    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"📊 Loading: {latest_file}")
    
    data = pd.read_csv(latest_file)
    logger.info(f"✅ Loaded: {data.shape}")
    logger.info(f"✅ Years: {sorted(data['Season'].unique())}")
    
    return data

def validate_data_quality(data):
    """Validate data quality and break immediately if issues found."""
    logger.info("🔍 Validating data quality...")
    
    # Check required columns
    required_columns = ['Name', 'Position', 'FantPt', 'Season', 'G#', 'Opp', 'Tm']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values in critical columns
    critical_columns = ['Name', 'Position', 'FantPt', 'Season']
    for col in critical_columns:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            raise ValueError(f"Column {col} has {missing_count} missing values. Fix data quality first.")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(data['FantPt']):
        raise ValueError("FantPt column is not numeric. Fix data quality first.")
    
    if not pd.api.types.is_numeric_dtype(data['Season']):
        raise ValueError("Season column is not numeric. Fix data quality first.")
    
    logger.info("✅ Data validation passed")

def clean_data_types(data):
    """Clean data types without any fallbacks - break if issues found."""
    logger.info("🧹 Cleaning data types...")
    
    # Create copy to avoid modifying original
    data = data.copy()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['FantPt', 'G#', 'Season']
    for col in numeric_columns:
        if col in data.columns:
            # Check for non-numeric values
            non_numeric = data[col].apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float))).sum()
            if non_numeric > 0:
                raise ValueError(f"Column {col} contains {non_numeric} non-numeric values. Fix data quality first.")
            
            # Convert to proper numeric type
            data[col] = pd.to_numeric(data[col])
            logger.info(f"   ✅ {col}: {data[col].dtype}")
    
    # Handle is_home column if it exists
    if 'is_home' in data.columns:
        # Check if it's already numeric (0/1) or boolean
        if pd.api.types.is_numeric_dtype(data['is_home']):
            # Check if it's binary (0/1)
            unique_values = set(data['is_home'].unique())
            if unique_values.issubset({0, 1}):
                logger.info(f"   ✅ is_home: Already binary (0/1) - {data['is_home'].dtype}")
            else:
                raise ValueError(f"Column is_home contains non-binary values: {unique_values}. Fix data quality first.")
        elif pd.api.types.is_bool_dtype(data['is_home']):
            logger.info(f"   ✅ is_home: Already boolean - {data['is_home'].dtype}")
        else:
            # Check for non-numeric/non-boolean values
            non_binary = data['is_home'].apply(lambda x: not pd.isna(x) and x not in [0, 1, True, False]).sum()
            if non_binary > 0:
                raise ValueError("Column is_home contains non-binary values. Fix data quality first.")
            
            # Convert to binary
            data['is_home'] = (data['is_home'] == 1).astype(int)
            logger.info("   ✅ is_home: Converted to binary (0/1)")
    
    logger.info(f"✅ Data type cleaning complete. Shape: {data.shape}")
    return data

def add_vor_rankings(data, vor_file_path):
    """Add VOR rankings to the dataset using the newly scraped data."""
    logger.info("🔄 Adding VOR rankings to dataset...")
    
    try:
        # Load VOR data
        ranking_data = pd.read_csv(vor_file_path)
        logger.info(f"   📊 Loaded VOR data: {len(ranking_data)} players")
        
        # Import fuzzy matching
        from fuzzywuzzy import fuzz
        
        def normalize_name(name):
            """Normalize player names for better matching."""
            if pd.isna(name):
                return ""
            
            # Convert to string and clean
            name = str(name).strip()
            
            # Remove common suffixes that cause mismatches
            suffixes = [' Jr.', ' Sr.', ' III', ' IV', ' V', ' II', ' I']
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
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
                    fuzz.token_set_ratio(normalized_name, normalized_vor_name)
                ]
                
                # Use the best score
                score = max(scores)
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = vor_row
            
            return best_match, best_score
        
        logger.info("   🔍 Performing fuzzy name matching...")
        
        # Initialize VOR ranking column
        data['vor_global_rank'] = 121  # Default worst rank
        data['vor_match_confidence'] = 0.0  # Track match quality
        
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
                player_mask = (data['Name'] == player_name) & (data['Position'] == player_pos)
                
                if confidence >= 95:  # High confidence match
                    exact_matches += 1
                    data.loc[player_mask, 'vor_global_rank'] = int(vor_match['VALUERANK'])
                    data.loc[player_mask, 'vor_match_confidence'] = confidence / 100.0
                elif confidence >= 85:  # Medium confidence match
                    fuzzy_matches += 1
                    data.loc[player_mask, 'vor_global_rank'] = int(vor_match['VALUERANK'])
                    data.loc[player_mask, 'vor_match_confidence'] = confidence / 100.0
                else:
                    no_matches += 1
        
        # Print matching statistics
        total_players = len(unique_players)
        logger.info(f"   ✅ Exact matches (95%+): {exact_matches}")
        logger.info(f"   ✅ Fuzzy matches (85%+): {fuzzy_matches}")
        logger.info(f"   ⚠️  No matches: {no_matches}")
        logger.info(f"   📊 Match rate: {((exact_matches + fuzzy_matches) / total_players * 100):.1f}%")
        
        # Final VOR ranking statistics
        valid_ranks = data[data['vor_global_rank'] != 121]['vor_global_rank']
        if len(valid_ranks) > 0:
            logger.info(f"   ✅ VOR ranking range: {valid_ranks.min()}-{valid_ranks.max()}")
            logger.info(f"   ✅ Players with VOR rankings: {len(valid_ranks)}")
        
        return data
        
    except Exception as e:
        logger.error(f"   ❌ Error adding VOR ranking: {e}")
        data['vor_global_rank'] = 121  # Default worst rank
        data['vor_match_confidence'] = 0.0
        return data

def calculate_basic_features(data):
    """Calculate basic features needed by all models."""
    logger.info("📊 Calculating basic features...")
    
    # Create copy to avoid modifying original
    data = data.copy()
    
    # Calculate 7-game average (if not already present)
    if '7_game_avg' not in data.columns:
        logger.info("   Calculating 7-game averages...")
        data = data.sort_values(['Name', 'Season', 'G#'])
        data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    
    # Calculate basic position indicators
    for pos in ['QB', 'RB', 'WR', 'TE']:
        data[f'is_{pos.lower()}'] = (data['Position'] == pos).astype(int)
    
    # Calculate basic team/opponent features
    data['opp_idx'] = pd.factorize(data['Opp'])[0]
    
    logger.info(f"✅ Basic features calculated. Shape: {data.shape}")
    return data

def create_unified_dataset(data_directory='datasets'):
    """Create unified, clean dataset for all models."""
    logger.info("=" * 60)
    logger.info("Creating Unified Fantasy Football Dataset")
    logger.info("=" * 60)
    
    try:
        # Step 1: Scrape VOR data from FantasyPros
        logger.info("Step 1: Scraping VOR data from FantasyPros...")
        config = load_config()
        
        # Scrape ADP data
        adp_data = scrape_adp_data(config)
        if adp_data is None:
            raise RuntimeError("Failed to scrape ADP data from FantasyPros")
        
        # Scrape projection data
        projection_data = scrape_projection_data(config)
        if projection_data is None:
            raise RuntimeError("Failed to scrape projection data from FantasyPros")
        
        # Calculate VOR rankings
        vor_data = calculate_vor_rankings(projection_data, config)
        
        # Save VOR data
        vor_file_path = save_vor_data(vor_data, config)
        
        # Step 2: Load historical NFL data
        logger.info("Step 2: Loading historical NFL data...")
        data = load_combined_dataset(data_directory)
        
        # Step 3: Validate quality
        logger.info("Step 3: Validating data quality...")
        validate_data_quality(data)
        
        # Step 4: Clean data types
        logger.info("Step 4: Cleaning data types...")
        data = clean_data_types(data)
        
        # Step 5: Add VOR rankings
        logger.info("Step 5: Adding VOR rankings...")
        data = add_vor_rankings(data, vor_file_path)
        
        # Step 6: Calculate basic features
        logger.info("Step 6: Calculating basic features...")
        data = calculate_basic_features(data)
        
        # Step 7: Save unified dataset
        logger.info("Step 7: Saving unified dataset...")
        from ffbayes.utils.path_constants import (
            get_unified_dataset_csv_path,
            get_unified_dataset_excel_path,
            get_unified_dataset_path,
        )

        # Get paths (directories will be created automatically)
        output_path = get_unified_dataset_path()
        excel_path = get_unified_dataset_excel_path()
        csv_path = get_unified_dataset_csv_path()
        
        # Save as Excel for human readability (not JSON for computers)
        data.to_excel(excel_path, index=False, engine='openpyxl')
        
        # Also save as CSV for compatibility
        data.to_csv(csv_path, index=False)
        
        # Keep JSON for programmatic use but make it clear it's not for humans
        data.to_json(output_path, orient='records', indent=2)
        
        logger.info("✅ Unified dataset saved in multiple formats:")
        logger.info(f"   📊 Excel (human-readable): {excel_path}")
        logger.info(f"   📋 CSV (compatibility): {csv_path}")
        logger.info(f"   🤖 JSON (programmatic): {output_path}")
        logger.info(f"✅ Final shape: {data.shape}")
        logger.info(f"✅ Available columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to create unified dataset: {e}")
        raise

def main():
    """Main function for creating unified dataset."""
    try:
        data = create_unified_dataset()
        logger.info("\n🎉 Unified dataset creation successful!")
        logger.info("All models should now use this dataset instead of individual data loading.")
        logger.info("VOR data has been scraped from FantasyPros and integrated.")
        
    except Exception as e:
        logger.error(f"\n💥 Unified dataset creation failed: {e}")
        logger.error("Fix data quality issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
