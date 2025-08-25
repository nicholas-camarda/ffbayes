#!/usr/bin/env python3
"""
Traditional VOR (Value Over Replacement) Draft Strategy
Took this from the internet - this is not mine!! Made some small changes just for me
This module generates traditional fantasy football draft rankings based on Value Over Replacement (VOR).
It scrapes ADP and projection data from FantasyPros and calculates VOR for each player.

Key Features:
- Scrapes ADP data from FantasyPros
- Scrapes projection data for all positions
- Calculates VOR based on replacement player values
- Generates CSV and Excel outputs with draft strategy
- Integrates with the pipeline configuration system
"""

import logging
import os
from datetime import date
from io import StringIO
from typing import Dict, Optional, Tuple

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
        from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
        config = VOR_CONFIG.copy()
        config.update({
            'ppr': config_loader.get_vor_setting('ppr'),
            'top_rank': config_loader.get_vor_setting('top_rank'),
            'output_dir': str(SNAKE_DRAFT_DATASETS_DIR)
        })
        # Also set the organized output directory for the pipeline
        from datetime import datetime

        from ffbayes.utils.path_constants import get_vor_strategy_dir
        config['organized_output_dir'] = str(get_vor_strategy_dir(datetime.now().year))
        logger.info(f"Configuration: PPR={config['ppr']}, Top Rank={config['top_rank']}")
        return config
    except ImportError:
        # Fallback to environment variables
        from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
        from ffbayes.utils.training_config import VOR_CONFIG
        config = VOR_CONFIG.copy()
        config.update({
            'ppr': float(os.getenv('VOR_PPR', '0.5')),
            'top_rank': int(os.getenv('VOR_TOP_RANK', '120')),
            'output_dir': str(SNAKE_DRAFT_DATASETS_DIR)
        })
        # Also set the organized output directory for the pipeline
        from datetime import datetime

        from ffbayes.utils.path_constants import get_vor_strategy_dir
        config['organized_output_dir'] = str(get_vor_strategy_dir(datetime.now().year))
        logger.info(f"Configuration: PPR={config['ppr']}, Top Rank={config['top_rank']}")
        return config

def make_adp_df(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape ADP data from FantasyPros.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with ADP data or None if failed
    """
    logger.info("Scraping ADP data from FantasyPros...")
    
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
        
        logger.info(f"Processed ADP data: {len(df)} players")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch ADP data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing ADP data: {e}")
        return None

def make_projection_df(config: Dict) -> Optional[pd.DataFrame]:
    """
    Scrape projection data for all positions from FantasyPros.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with projection data or None if failed
    """
    logger.info("Scraping projection data from FantasyPros...")
    
    final_df = pd.DataFrame()
    
    for position in config['positions']:
        try:
            url = config['base_url_projections'].format(position=position)
            logger.info(f"Fetching {position.upper()} projections...")
            
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
            logger.info(f"Added {len(df)} {position.upper()} players")
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {position} projections: {e}")
        except Exception as e:
            logger.error(f"Error processing {position} projections: {e}")
    
    if final_df.empty:
        logger.error("No projection data could be loaded")
        return None
    
    final_df = final_df.sort_values(by='FPTS', ascending=False)
    logger.info(f"Total projection data: {len(final_df)} players")
    return final_df

def calculate_vor(df: pd.DataFrame, replacement_players: Dict[str, str], config: Dict) -> pd.DataFrame:
    """
    Calculate Value Over Replacement for each player.
    
    Args:
        df: DataFrame with projection data
        replacement_players: Dictionary of replacement players by position
        config: Configuration dictionary
        
    Returns:
        DataFrame with VOR calculations
    """
    logger.info("Calculating Value Over Replacement...")
    
    # Get replacement values
    replacement_values = {}
    for position, player in replacement_players.items():
        player_data = df[df['PLAYER'] == player]
        if not player_data.empty:
            replacement_values[position] = player_data.iloc[0]['FPTS']
            logger.info(f"Replacement {position}: {player} = {replacement_values[position]:.1f} points")
        else:
            logger.warning(f"Could not find replacement player {player} for {position}")
            replacement_values[position] = 0
    
    # Calculate VOR
    df['VOR'] = df.apply(
        lambda row: row['FPTS'] - replacement_values.get(row['POS'], 0), 
        axis=1
    )
    
    df = df.sort_values(by='VOR', ascending=False)
    df['VALUERANK'] = df['VOR'].rank(ascending=False)
    
    logger.info(f"VOR calculation complete. Top player: {df.iloc[0]['PLAYER']} (VOR: {df.iloc[0]['VOR']:.1f})")
    return df

def generate_draft_strategy(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Generate comprehensive draft strategy and save to Excel.
    
    Args:
        df: DataFrame with VOR data
        output_file_path: Path to save Excel file
    """
    logger.info("Generating draft strategy...")
    
    # Categorize players by draft phase
    def categorize_draft_phase(row):
        if row['VALUERANK'] <= 70:
            return 'Early Draft'
        if 70 < row['VALUERANK'] <= 140:
            return 'Mid Draft'
        return 'Late Draft'
    
    df['Draft_Phase'] = df.apply(categorize_draft_phase, axis=1)
    
    # Create draft strategy
    draft_strategy = {
        'Round': list(range(1, 17)),
        'Primary Target': [
            'Top-tier RB or WR (highest VOR available)',
            'Another top-tier RB or WR (fill unfilled position)',
            'RB or WR (high VOR players)',
            'RB or WR (aim for 2 strong players in both positions)',
            'RB or WR (continue building depth)',
            'Mid-tier QBs and TEs',
            'Mid-tier QBs and TEs',
            'Build depth at RB and WR',
            'Build depth at RB and WR',
            'Top Defense/Special Teams',
            'High-upside RB/WR (breakout candidates)',
            'High-upside RB/WR (breakout candidates)',
            'High-upside RB/WR (breakout candidates)',
            'Defense/Special Teams',
            'Kicker',
            'Kicker (if not selected)',
        ],
        'Backup Plan': [
            'Top-tier TE if RBs/WRs gone',
            'Top-tier TE if available',
            'Top-tier QB if available',
            'QB if not selected',
            'TE if not selected',
            'Build depth at RB/WR',
            'Build depth at RB/WR',
            'Top Defense/Special Teams',
            'Top Defense/Special Teams',
            'Build depth at RB/WR',
            'Backup QB or TE',
            'Backup QB or TE',
            'Backup QB or TE',
            'High-upside bench players',
            'High-upside RB/WR',
            'High-upside RB/WR',
        ],
    }
    
    draft_strategy_df = pd.DataFrame(draft_strategy)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file_path) as writer:
        # VOR data by draft phase
        for phase in ['Early Draft', 'Mid Draft', 'Late Draft']:
            phase_data = df[df['Draft_Phase'] == phase].sort_values(by='VOR', ascending=False)
            sheet_name = f'{phase} (Rounds 1-5)' if phase == 'Early Draft' else \
                        f'{phase} (Rounds 6-10)' if phase == 'Mid Draft' else \
                        f'{phase} (Rounds 11-16)'
            phase_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Round-by-round strategy
        draft_strategy_df.to_excel(writer, sheet_name='Round-by-Round Strategy', index=False)
        
        # Summary
        summary_data = {
            'Draft Phase': ['Early Draft (Rounds 1-5)', 'Mid Draft (Rounds 6-10)', 'Late Draft (Rounds 11-16)'],
            'Primary Focus': [
                'Top WRs and RBs with highest VOR values',
                'Mid-tier WRs and RBs with good value',
                'High-upside players with breakout potential'
            ],
            'Secondary Focus': [
                'Top QB or TE if available',
                'Fill QB and TE positions',
                'Backup players and streaming options'
            ],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Strategy Summary', index=False)
    
    logger.info(f"Draft strategy saved to: {output_file_path}")

def save_results(df: pd.DataFrame, config: Dict) -> Tuple[str, str]:
    """
    Save VOR results to CSV and Excel files.
    
    Args:
        df: DataFrame with VOR data
        config: Configuration dictionary
        
    Returns:
        Tuple of (csv_path, excel_path)
    """
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Generate filenames
    current_year = date.today().strftime('%Y')
    csv_filename = f'snake-draft_ppr-{config["ppr"]}_vor_top-{config["top_rank"]}_{current_year}.csv'
    excel_filename = f'DRAFTING STRATEGY -- snake-draft_ppr-{config["ppr"]}_vor_top-{config["top_rank"]}_{current_year}.xlsx'
    
    csv_path = os.path.join(config['output_dir'], csv_filename)
    excel_path = os.path.join(config['output_dir'], excel_filename)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"VOR data saved to: {csv_path}")
    
    # Generate and save Excel strategy
    generate_draft_strategy(df, excel_path)
    
    # Also save to organized directory for pipeline compatibility
    if 'organized_output_dir' in config:
        organized_csv_path = os.path.join(config['organized_output_dir'], csv_filename)
        organized_excel_path = os.path.join(config['organized_output_dir'], excel_filename)
        
        # Ensure organized directory exists
        os.makedirs(config['organized_output_dir'], exist_ok=True)
        
        # Copy files to organized location
        import shutil
        shutil.copy2(csv_path, organized_csv_path)
        shutil.copy2(excel_path, organized_excel_path)
        logger.info(f"VOR data also saved to organized location: {organized_csv_path}")
    
    return csv_path, excel_path

def main():
    """Main function to run the VOR draft strategy pipeline."""
    logger.info("=" * 70)
    logger.info("Traditional VOR Draft Strategy Pipeline")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = load_config()
        
        # Scrape ADP data
        adp_df = make_adp_df(config)
        if adp_df is None:
            logger.error("Failed to load ADP data. Exiting.")
            return None
        
        # Find replacement players
        replacement_players = {'RB': '', 'WR': '', 'TE': '', 'QB': ''}
        for _, row in adp_df[:config['top_rank']].iterrows():
            position = row['POS']
            player = row['PLAYER']
            replacement_players[position] = player
        
        logger.info(f"Replacement players: {replacement_players}")
        
        # Scrape projection data
        projection_df = make_projection_df(config)
        if projection_df is None:
            logger.error("Failed to load projection data. Exiting.")
            return None
        
        # Calculate VOR
        vor_df = calculate_vor(projection_df, replacement_players, config)
        
        # Save results
        csv_path, excel_path = save_results(vor_df, config)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("VOR Draft Strategy Summary:")
        logger.info(f"- Top {config['top_rank']} players ranked by VOR")
        logger.info(f"- PPR scoring: {config['ppr']}")
        logger.info(f"- CSV output: {csv_path}")
        logger.info(f"- Excel strategy: {excel_path}")
        logger.info("=" * 70)
        
        return {
            'csv_path': csv_path,
            'excel_path': excel_path,
            'top_player': vor_df.iloc[0]['PLAYER'],
            'top_vor': vor_df.iloc[0]['VOR']
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return None

if __name__ == '__main__':
    main()
