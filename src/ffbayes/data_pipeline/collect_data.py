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
# Automatically calculate the last 10 years
# This ensures the script always collects the most recent 10 years of data
# without needing manual updates each year
# Note: Data is typically only available up to the previous year
DEFAULT_YEARS = list(range(CURRENT_YEAR - 10, CURRENT_YEAR))

# Timeout configuration
API_TIMEOUT = 30  # seconds


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

        # Merge player data with schedule data for home/away indicators
        print("      🔄 Merging data...")
        
        # Debug: Check data before merge
        print(f"      🔍 Players data shape: {players_df.shape}")
        print(f"      🔍 Schedules data shape: {schedules_df.shape}")
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
        ]
        
        # Filter to only include columns that exist
        available_columns = merged_df.columns.tolist()
        final_columns = [col for col in final_columns if col in available_columns]
        print(f"      🔍 Final columns to use: {final_columns}")
        
        final_df = merged_df[final_columns]

        # Clean up the data (simplified approach like working script)
        final_df = final_df[final_df['game_id'].notna()]
        final_df['player_team'] = final_df['recent_team']
        
        # Use robust filtering to avoid Series boolean evaluation issues
        mask = (
            (final_df['player_team'].astype(str) == final_df['home_team'].astype(str)) |
            (final_df['player_team'].astype(str) == final_df['away_team'].astype(str))
        )
        final_df = final_df[mask]
        final_df = final_df.drop_duplicates()
        
        # Now rename the column for consistency
        final_df = final_df.rename(columns={'recent_team': 'player_team'})

        # Add injury data if available
        try:
            print("      🔄 Adding injury data...")
            signal.alarm(API_TIMEOUT)
            injuries = nfl.import_injuries([year])
            signal.alarm(0)
            
            injuries = injuries[[
                'full_name',
                'position',
                'week',
                'season',
                'team',
                'season_type',
                'report_status',
                'practice_status',
            ]]
            injuries = injuries.rename(
                columns={'full_name': 'player_display_name', 'team': 'home_team'},
            )

            # Merge injury data
            final_df = pd.merge(
                final_df,
                injuries,
                on=['player_display_name', 'position', 'season', 'week', 'season_type', 'home_team'],
                how='left',
            )
            final_df = final_df.rename(
                columns={
                    'report_status': 'game_injury_report_status',
                    'practice_status': 'practice_injury_report_status',
                },
            )
            print("      ✅ Injury data added")
        except Exception as e:
            print(f"      ⚠️  Injury data not available: {e}")
            # Add empty injury columns
            final_df['game_injury_report_status'] = None
            final_df['practice_injury_report_status'] = None

        print(f"      ✅ Dataset created: {len(final_df):,} rows")
        return final_df

    except Exception as e:
        print(f"      ❌ Error creating dataset for {year}: {e}")
        return None


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
    print(f"   📅 Automatically calculated last 10 available years from current year ({CURRENT_YEAR})")
    
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
        # Fallback to basic progress bar
        with alive_bar(len(years), title="Collecting NFL Data", bar="smooth") as bar:
            for year in years:
                # Check if file already exists
                if any(file.startswith(str(year)) and 'season.csv' in file for file in existing_files):
                    print(f"⏭️  Skipping year {year} as file already exists")
                    bar()
                    continue
                
                # Check if data is available
                available, result = check_data_availability(year)
                if not available:
                    print(f"⏭️  Skipping year {year} - data not available: {result}")
                    bar()
                    continue
                
                # Process the year
                result = collect_data_by_year(year)
                if result is not None:
                    successful_years.append(year)
                
                bar()
    
    return successful_years


def main():
    """Main data collection function."""
    print("=" * 60)
    print("ENHANCED NFL DATA COLLECTION PIPELINE")
    print("=" * 60)
    print("Merged functionality from get_ff_data.py and get_ff_data_improved.py")
    print("=" * 60)
    
    start_time = time.time()
    
    # Collect data for the last 10 years
    successful_years = collect_nfl_data()  # Uses DEFAULT_YEARS (2015-2024)
    
    # Combine datasets
    if successful_years:
        combine_datasets(SEASON_DATASETS_DIR, COMBINED_DATASETS_DIR, successful_years)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Collection completed in {elapsed_time:.1f} seconds")
    print(f"✅ Successfully processed {len(successful_years)} years: {successful_years}")
    
    print("\n🎯 Next step: Run 02_validate_data.py")


if __name__ == "__main__":
    main()
