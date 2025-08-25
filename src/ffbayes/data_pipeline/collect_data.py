#!/usr/bin/env python3
"""
01_collect_data.py - Enhanced Data Collection Pipeline
First step in the fantasy football analytics pipeline.
Collects raw NFL data from multiple sources with sophisticated processing.
"""


# Import progress monitoring utilities
import signal
import time
from datetime import datetime
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
from alive_progress import alive_bar

# Set up project paths - scripts should be run from project root
PROJECT_ROOT = Path.cwd()  # Current working directory should be project root
DATASETS_DIR = PROJECT_ROOT / 'datasets'
SEASON_DATASETS_DIR = DATASETS_DIR / 'season_datasets'
COMBINED_DATASETS_DIR = DATASETS_DIR / 'combined_datasets'

try:
    from ffbayes.utils.progress_monitor import ProgressMonitor
except Exception:
    ProgressMonitor = None

# Configuration
CURRENT_YEAR = datetime.now().year
# Automatically calculate the last 5 years
# This ensures the script always collects the most recent 5 years of data
# without needing manual updates each year
# Note: Data is typically only available up to the previous year
DEFAULT_YEARS = list(range(CURRENT_YEAR - 5, CURRENT_YEAR))

# Timeout configuration
API_TIMEOUT = 30  # seconds

# Standard fantasy football defense scoring rules
DEFENSE_SCORING = {
    'sack': 1.0,           # 1 point per sack
    'interception': 2.0,    # 2 points per interception
    'fumble_recovery': 2.0, # 2 points per fumble recovery
    'safety': 2.0,          # 2 points per safety
    'defensive_td': 6.0,    # 6 points per defensive touchdown
    'kickoff_td': 6.0,      # 6 points per kickoff return touchdown
    'punt_td': 6.0,         # 6 points per punt return touchdown
    'points_allowed_0': 10.0,   # 10 points for 0 points allowed
    'points_allowed_1_6': 7.0,  # 7 points for 1-6 points allowed
    'points_allowed_7_13': 4.0, # 4 points for 7-13 points allowed
    'points_allowed_14_20': 1.0, # 1 point for 14-20 points allowed
    'points_allowed_21_27': 0.0, # 0 points for 21-27 points allowed
    'points_allowed_28_34': -1.0, # -1 point for 28-34 points allowed
    'points_allowed_35_plus': -4.0, # -4 points for 35+ points allowed
}


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("API call timed out")


def check_data_availability(year):
    """Check if data is available for a given year with timeout and error handling."""
    try:
        # Set timeout for this function
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(API_TIMEOUT)
        
        # Test if we can get a small sample of data
        test_data = nfl.import_weekly_data([year])
        
        # Clear the alarm
        signal.alarm(0)
        
        return True, len(test_data)
    except TimeoutError:
        print(f"      ⏰ Timeout checking data availability for {year}")
        return False, "API timeout"
    except Exception as e:
        # Clear the alarm in case of other errors
        try:
            signal.alarm(0)
        finally:
            pass
        return False, str(e)


def create_dataset(year):
    """Create comprehensive dataset for a given year with proper error handling and timeout."""
    print(f"   🔄 Creating dataset for {year}...")
    
    try:
        # Set timeout for this function
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(API_TIMEOUT)
        
        # Get player weekly data
        players = nfl.import_weekly_data([year])
        print(f"      ✅ Players: {len(players):,} rows")
        
        # Clear the alarm
        signal.alarm(0)
        
        # Select relevant columns
        players_df = players[[
            'player_id',
            'player_display_name',
            'position',
            'recent_team',
            'season',
            'week',
            'season_type',
            'fantasy_points',
            'fantasy_points_ppr',
        ]]

        # Get schedule data with timeout
        signal.alarm(API_TIMEOUT)
        schedules = nfl.import_schedules([year])
        signal.alarm(0)
        print(f"      ✅ Schedule: {len(schedules):,} rows")
        
        schedules_df = schedules[[
            'game_id',
            'week',
            'season',
            'gameday',
            'game_type',
            'home_team',
            'away_team',
            'away_score',
            'home_score',
        ]]

        # Get team defense data with timeout
        signal.alarm(API_TIMEOUT)
        try:
            defense_data = nfl.import_weekly_pfr('def', [year])
            print(f"      ✅ Team Defense: {len(defense_data):,} rows")
        except Exception as e:
            print(f"      ⚠️  Team Defense data unavailable: {e}")
            defense_data = pd.DataFrame()
        signal.alarm(0)

        # Merge player data with schedule data for home/away indicators
        print("      🔄 Merging data...")
        
        # Debug: Check data before merge
        print(f"      🔍 Players data shape: {players_df.shape}")
        print(f"      🔍 Schedules data shape: {schedules_df.shape}")
        print(f"      🔍 Team Defense data shape: {defense_data.shape}")
        print(f"      🔍 Sample player teams: {players_df['recent_team'].unique()[:5]}")
        print(f"      🔍 Sample home teams: {schedules_df['home_team'].unique()[:5]}")
        print(f"      🔍 Sample away teams: {schedules_df['away_team'].unique()[:5]}")
        
        # Merge considering players as home team
        home_merge = players_df.merge(
            schedules_df,
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'home_team'],
            how='left',
        )
        print(f"      🔍 Home merge result: {home_merge.shape}")

        # Merge considering players as away team
        away_merge = players_df.merge(
            schedules_df,
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'away_team'],
            how='left',
        )
        print(f"      🔍 Away merge result: {away_merge.shape}")

        # Create home/away indicators (simplified approach)
        home_merge['is_home_team'] = home_merge['home_team'].notna()
        away_merge['is_away_team'] = away_merge['away_team'].notna()

        # Combine the merges using concat (like working script)
        merged_df = pd.concat([home_merge, away_merge])

        # Select final columns (simplified approach like working script)
        final_columns = [
            'player_id',
            'player_display_name',
            'position',
            'recent_team',
            'home_team',
            'season',
            'week',
            'season_type',
            'fantasy_points',
            'fantasy_points_ppr',
            'game_id',
            'gameday',
            'away_team',
            'away_score',
            'home_score',
            'is_home_team',  # Add this for Bayesian model compatibility
        ]
        
        # Filter to only include columns that exist
        existing_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[existing_columns]
        
        # Add compatibility columns for Monte Carlo script
        merged_df['Name'] = merged_df['player_display_name']
        merged_df['FantPt'] = merged_df['fantasy_points']
        merged_df['Season'] = merged_df['season']
        merged_df['G#'] = merged_df['week']
        
        # Add compatibility columns for Bayesian model
        merged_df['Position'] = merged_df['position']  # Capitalize for Bayesian model
        merged_df['is_home'] = merged_df['is_home_team']  # Boolean for Bayesian model
        
        # Add opponent column for Bayesian model
        merged_df['Opp'] = merged_df.apply(
            lambda x: x['away_team'] if x['is_home_team'] else x['home_team'], 
            axis=1
        )
        
        # Add compatibility columns for advanced stats calculator
        merged_df['Tm'] = merged_df['recent_team']  # Team column
        merged_df['Date'] = merged_df['gameday']  # Game date column
        
        # Calculate 7-game rolling average for fantasy points
        merged_df = merged_df.sort_values(['Name', 'Season', 'G#']).reset_index(drop=True)
        merged_df['7_game_avg'] = merged_df.groupby(['Name', 'Season'])['FantPt'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Add player rankings from VOR data
        merged_df = add_player_rankings(merged_df, year)
        
        # Add team defense data if available
        if not defense_data.empty:
            defense_df = process_team_defense_data(defense_data, schedules_df, year)
            if not defense_df.empty:
                print(f"      ✅ Processed team defense data: {len(defense_df):,} rows")
                # Combine with player data
                merged_df = pd.concat([merged_df, defense_df], ignore_index=True)
                print(f"      ✅ Combined dataset: {len(merged_df):,} rows")
        
        return merged_df
        
    except TimeoutError:
        print(f"      ❌ Timeout creating dataset for {year}")
        return pd.DataFrame()
    except Exception as e:
        print(f"      ❌ Error creating dataset for {year}: {e}")
        return pd.DataFrame()


def add_player_rankings(df, year):
    """Add player rankings from VOR data to the dataset."""
    try:
        # Look for VOR ranking data in datasets directory - use dynamic filename generator and path constants
        from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
        from ffbayes.utils.vor_filename_generator import get_vor_csv_filename
        vor_files = [
            str(SNAKE_DRAFT_DATASETS_DIR / get_vor_csv_filename(year)),
            str(SNAKE_DRAFT_DATASETS_DIR / get_vor_csv_filename(year-1)),  # Try previous year
            str(SNAKE_DRAFT_DATASETS_DIR / get_vor_csv_filename(year-2)),  # Try two years back
        ]
        
        ranking_data = None
        for file_path in vor_files:
            if Path(file_path).exists():
                print(f"      📊 Loading rankings from: {file_path}")
                ranking_data = pd.read_csv(file_path)
                break
        
        if ranking_data is not None:
            # Clean up player names for matching
            ranking_data['PLAYER_CLEAN'] = ranking_data['PLAYER'].str.strip()
            df['Name_CLEAN'] = df['Name'].str.strip()
            
            # Merge rankings based on player name and position
            df = df.merge(
                ranking_data[['PLAYER_CLEAN', 'POS', 'VALUERANK']],
                left_on=['Name_CLEAN', 'Position'],
                right_on=['PLAYER_CLEAN', 'POS'],
                how='left'
            )
            
            # Normalize VOR rankings to 0-3 scale (0=best, 3=worst)
            df['rank'] = df['VALUERANK'].fillna(120).astype(int)
            # Convert to 0-3 scale: 1-30=0, 31-60=1, 61-90=2, 91-120=3
            df['rank'] = ((df['rank'] - 1) // 30).clip(0, 3)

            # Clean up temporary columns
            df = df.drop(['PLAYER_CLEAN', 'Name_CLEAN', 'POS', 'VALUERANK'], axis=1, errors='ignore')
            
            print(f"      ✅ Added rankings for {df['rank'].notna().sum()} players")
        else:
            print(f"      ⚠️  No VOR ranking data found for {year}, using default ranks")
            df['rank'] = 3  # Default high rank (low priority) but within bounds
        
        return df
        
    except Exception as e:
        print(f"      ⚠️  Error adding rankings: {e}")
        df['rank'] = 3  # Default high rank (low priority) but within bounds
        return df


def process_team_defense_data(defense_data, schedules_df, year):
    """Process team defense data and calculate fantasy points."""
    try:
        # Aggregate defense stats by team, week, season (only use available columns)
        defense_agg = defense_data.groupby(['team', 'week', 'season']).agg({
            'def_sacks': 'sum',
            'def_ints': 'sum',
            'def_tackles_combined': 'sum',
            'def_missed_tackles': 'sum',
            'def_pressures': 'sum',
            'def_times_hitqb': 'sum',
            'def_times_hurried': 'sum',
            'def_times_blitzed': 'sum',
            'def_yards_allowed': 'sum',
            'def_receiving_td_allowed': 'sum',
            'def_completions_allowed': 'sum',
        }).reset_index()
        
        # Merge with schedule data to get opponent scores - handle both home and away teams
        # First merge with home teams
        home_defense = defense_agg.merge(
            schedules_df,
            left_on=['team', 'week', 'season'],
            right_on=['home_team', 'week', 'season'],
            how='left'
        )
        
        # Then merge with away teams
        away_defense = defense_agg.merge(
            schedules_df,
            left_on=['team', 'week', 'season'],
            right_on=['away_team', 'week', 'season'],
            how='left'
        )
        
        # Combine both merges
        defense_agg = pd.concat([home_defense, away_defense], ignore_index=True)
        
        # Debug: Check what columns we have after merge
        print(f"      🔍 Defense agg columns after merge: {defense_agg.columns.tolist()}")
        print(f"      🔍 Defense agg shape after merge: {defense_agg.shape}")
        
        # Debug: Check specific columns
        print(f"      🔍 home_team column exists: {'home_team' in defense_agg.columns}")
        print(f"      🔍 away_team column exists: {'away_team' in defense_agg.columns}")
        print(f"      🔍 Sample home_team values: {defense_agg['home_team'].head(3).tolist()}")
        
        # Calculate fantasy points for team defense
        defense_agg['fantasy_points'] = defense_agg.apply(calculate_defense_fantasy_points, axis=1)
        
        # Create standardized columns to match player data
        print("      🔍 Creating defense DataFrame...")
        try:
            defense_df = pd.DataFrame({
                'player_id': [f'DEF_{team}_{season}_{week}' for team, season, week in zip(defense_agg['team'], defense_agg['season'], defense_agg['week'])],
                'Name': [f'{team} Defense' for team in defense_agg['team']],  # Add Name column for compatibility
                'player_display_name': [f'{team} Defense' for team in defense_agg['team']],
                'position': 'DEF',
                'recent_team': defense_agg['team'].values,
                'season': defense_agg['season'].values,
                'Season': defense_agg['season'].values,  # Add Season column for compatibility
                'week': defense_agg['week'].values,
                'G#': defense_agg['week'].values,  # Add G# column for compatibility
                'season_type': 'REG',
                'fantasy_points': defense_agg['fantasy_points'].values,
                'fantasy_points_ppr': defense_agg['fantasy_points'].values,  # Same for defense
                'FantPt': defense_agg['fantasy_points'].values,  # Add FantPt column for compatibility
                'game_id': defense_agg['game_id'].values,
                'is_home_team': defense_agg['home_team'].notna().values,
                'opponent_score': defense_agg.apply(
                    lambda x: x['away_score'] if x['home_team'] == x['team'] else x['home_score'], 
                    axis=1
                ).values,
                'team_score': defense_agg.apply(
                    lambda x: x['home_score'] if x['home_team'] == x['team'] else x['away_score'], 
                    axis=1
                ).values,
                'home_team': defense_agg['home_team'].values,  # Add for Opp calculation
                'away_team': defense_agg['away_team'].values,  # Add for Opp calculation
            })
            print("      ✅ Defense DataFrame created successfully")
        except Exception as e:
            print(f"      ❌ Error creating defense DataFrame: {e}")
            raise
        
        # Add compatibility columns for Bayesian model
        defense_df['Position'] = 'DEF'  # Capitalize for Bayesian model
        defense_df['is_home'] = defense_df['is_home_team']  # Boolean for Bayesian model
        
        # Create Opp column by determining opponent from home/away status
        # We need to get the opponent from the original defense_agg data
        defense_df['Opp'] = defense_df.apply(
            lambda x: x['away_team'] if x['is_home_team'] else x['home_team'], 
            axis=1
        )
        
        # Add compatibility columns for advanced stats calculator
        defense_df['Tm'] = defense_df['recent_team']  # Team column
        # Create a proper date from season and week for team defense
        # Use a more robust approach to avoid invalid dates
        defense_df['Date'] = defense_df.apply(
            lambda x: f"{x['Season']}-01-01", axis=1
        )  # Use season-01-01 format as placeholder date (avoiding week-based dates)
        
        # Calculate 7-game rolling average for fantasy points (for defense)
        defense_df = defense_df.sort_values(['Name', 'Season', 'G#']).reset_index(drop=True)
        defense_df['7_game_avg'] = defense_df.groupby(['Name', 'Season'])['FantPt'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Debug: Check what columns we have in defense_df
        print(f"      🔍 Defense DataFrame columns: {defense_df.columns.tolist()}")
        print(f"      🔍 Defense DataFrame shape: {defense_df.shape}")
        
        print("      ✅ Added compatibility columns for Bayesian model")
        
        return defense_df

    except Exception as e:
        print(f"      ❌ Error processing team defense data: {e}")
        return pd.DataFrame()


def calculate_defense_fantasy_points(row):
    """Calculate fantasy points for team defense based on standard scoring rules."""
    try:
        points = 0.0
        
        # Basic defensive stats (only use available columns)
        points += (row.get('def_sacks', 0) * DEFENSE_SCORING['sack'])
        points += (row.get('def_ints', 0) * DEFENSE_SCORING['interception'])
        
        # Note: Many defensive stats like fumbles, safeties, TDs are not available in PFR data
        # This is a simplified scoring system based on available data
        
        # Points allowed scoring (simplified - would need opponent scores from schedule)
        # This is a placeholder - actual implementation would need opponent scoring data
        opponent_score = row.get('opponent_score', 0)
        if opponent_score == 0:
            points += DEFENSE_SCORING['points_allowed_0']
        elif opponent_score <= 6:
            points += DEFENSE_SCORING['points_allowed_1_6']
        elif opponent_score <= 13:
            points += DEFENSE_SCORING['points_allowed_7_13']
        elif opponent_score <= 20:
            points += DEFENSE_SCORING['points_allowed_14_20']
        elif opponent_score <= 27:
            points += DEFENSE_SCORING['points_allowed_21_27']
        elif opponent_score <= 34:
            points += DEFENSE_SCORING['points_allowed_28_34']
        else:
            points += DEFENSE_SCORING['points_allowed_35_plus']
        
        return round(points, 2)

    except Exception as e:
        print(f"      ❌ Error calculating defense fantasy points: {e}")
        return 0.0


def process_dataset(final_df, year):
    """Process the dataset into the final format."""
    if final_df is None or len(final_df) == 0:
        return None
    
    print(f"   🔄 Processing {year} data...")
    
    data_list = []
    max_rows = len(final_df)
    
    # Process data without nested progress bar
    for i, row in enumerate(final_df.itertuples()):
        try:
            # Extract player details - itertuples gives scalar values directly
            player_id = row.player_id
            player_name = row.player_display_name
            season = row.season
            position = row.position
            week = row.week
            fantasy_points_ppr = row.fantasy_points_ppr
            fantasy_points = row.fantasy_points
            game_date = row.gameday
            team = row.player_team
            home = row.home_team
            away = row.away_team
            
            # Simple scalar comparison - no need for complex extraction
            opponent = away if team == away else home
            game_injury_report_status = getattr(row, 'game_injury_report_status', None)
            practice_injury_report_status = getattr(row, 'practice_injury_report_status', None)
            
            # Create is_home indicator
            is_home = 1 if team == home else 0
            
            data_list.append([
                week,
                game_date,
                team,
                away,
                opponent,
                fantasy_points,
                fantasy_points_ppr,
                player_name,
                player_id,
                position,
                season,
                game_injury_report_status,
                practice_injury_report_status,
                is_home,  # New column for home/away
            ])
            
            # Update progress every 100 rows
            if i % 100 == 0:
                print(f"      📊 Processed {i:,}/{max_rows:,} rows...")
            
        except Exception as e:
            print(f"      ⚠️  Error processing row {i}: {e}")
            continue
    
    # Create final DataFrame
    columns = [
        'G#', 'Date', 'Tm', 'Away', 'Opp', 'FantPt', 'FantPtPPR',
        'Name', 'PlayerID', 'Position', 'Season', 'GameInjuryStatus',
        'PracticeInjuryStatus', 'is_home'
    ]
    
    df = pd.DataFrame(data_list, columns=columns)
    print(f"   ✅ Processed {len(df):,} rows for {year}")
    return df


def collect_data_by_year(year):
    """Collect and process data for a specific year."""
    print(f"\n📊 Processing year {year}...")
    
    # Check data availability first
    available, result = check_data_availability(year)
    if not available:
        print(f"   ❌ Data not available for {year}: {result}")
        return None
    
    # Create dataset
    final_df = create_dataset(year)
    if final_df is None:
        return None
    
    # Process dataset
    processed_df = process_dataset(final_df, year)
    if processed_df is None:
        return None
    
    # Save to file
    SEASON_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    filename = SEASON_DATASETS_DIR / f'{year}season.csv'
    processed_df.to_csv(filename, index=False)
    print(f"   💾 Saved to {filename}")
    
    return processed_df


def combine_datasets(directory_path, output_directory_path, years_to_process):
    """Combine all datasets into a single file."""
    print("\n🔗 Combining datasets...")
    
    # Get list of CSV files
    files = [f for f in directory_path.iterdir() if f.is_file() and f.suffix == '.csv' and 'season' in f.name]
    files = sorted(files)
    
    print(f"   Found {len(files)} season files")
    
    dfs = []
    with alive_bar(len(files), title="Combining Datasets") as bar:
        for f in files:
            try:
                data = pd.read_csv(f)
                dfs.append(data)
                print(f"      ✅ {f.name}: {len(data):,} rows")
                bar()
            except Exception as e:
                print(f"      ❌ Error reading {f.name}: {e}")
                bar()
    
    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        output_directory_path.mkdir(exist_ok=True)
        output_file = output_directory_path / 'combined_data.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"   💾 Combined dataset saved: {len(combined_df):,} rows")
        print(f"   📁 Check {output_directory_path} for combined dataset")
        return combined_df
    else:
        print("   ❌ No valid datasets to combine")
        return None


def collect_nfl_data(years=None):
    """Collect NFL data for specified years with enhanced functionality."""
    if years is None:
        years = DEFAULT_YEARS
    
    print(f"📊 Collecting NFL data for years: {years}")
    print(f"   📅 Automatically calculated last 5 available years from current year ({CURRENT_YEAR})")
    
    # Create directory structure
    SEASON_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = [f.name for f in SEASON_DATASETS_DIR.iterdir() if f.is_file()]
    
    successful_years = []
    
    # Use progress monitoring if available
    if ProgressMonitor:
        monitor = ProgressMonitor("Data Collection")
        monitor.start_timer()
        
        with monitor.monitor(len(years), "Processing Years"):
            for year in years:
                # Check if file already exists
                if any(file.startswith(str(year)) and 'season.csv' in file for file in existing_files):
                    print(f"⏭️  Skipping year {year} as file already exists")
                    continue
                
                # Check if data is available
                available, result = check_data_availability(year)
                if not available:
                    print(f"⏭️  Skipping year {year} - data not available: {result}")
                    continue
                
                # Process the year
                result = collect_data_by_year(year)
                if result is not None:
                    successful_years.append(year)
    else:
        # CRITICAL: Enhanced progress tracking required
        raise RuntimeError(
            "Enhanced progress tracking failed. "
            "Production data collection requires proper progress monitoring. "
            "No fallbacks allowed."
        )
    
    return successful_years


def main(args=None):
    """Main data collection function with standardized interface."""
    from ffbayes.utils.script_interface import create_standardized_interface
    
    interface = create_standardized_interface(
        "ffbayes-collect",
        "Enhanced NFL data collection pipeline with standardized interface"
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
    
    # Parse years if provided
    years = None
    if args.years:
        years = interface.parse_years(args.years)
        logger.info(f"Processing specified years: {years}")
    else:
        years = DEFAULT_YEARS
        logger.info(f"Processing default years: {years}")
    
    # Check for quick test mode
    if args.quick_test:
        logger.warning("⚠️  QUICK_TEST mode explicitly enabled - processing limited data")
        logger.warning("⚠️  This will not provide production-quality results")
        years = years[:2]  # Only process first 2 years in quick test mode
    
    # Check for force refresh
    if args.force_refresh:
        logger.info("Force refresh enabled - will overwrite existing data")
    
    start_time = time.time()
    
    # Collect data for specified years
    successful_years = interface.handle_errors(collect_nfl_data, years)
    
    # Combine datasets
    if successful_years:
        interface.handle_errors(combine_datasets, SEASON_DATASETS_DIR, COMBINED_DATASETS_DIR, successful_years)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Collection completed in {elapsed_time:.1f} seconds")
    logger.info(f"Successfully processed {len(successful_years)} years: {successful_years}")
    
    interface.log_completion("Data collection completed successfully")
    
    return successful_years


if __name__ == "__main__":
    main()
