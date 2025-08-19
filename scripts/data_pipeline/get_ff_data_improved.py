#!/usr/bin/env conda run -n ffbayes python
"""
get_ff_data_improved.py - Enhanced Data Collection Pipeline
Improved version of the existing get_ff_data.py with better error handling,
organization, and fixes for data availability issues.
"""

import os
import time
from datetime import datetime

import nfl_data_py as nfl
import pandas as pd
from alive_progress import alive_bar

# Configuration
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
CURRENT_YEAR = datetime.now().year

def check_data_availability(year):
    """Check if data is available for a given year."""
    try:
        # Test if we can get a small sample of data
        test_data = nfl.import_weekly_data([year])
        return True, len(test_data)
    except Exception as e:
        return False, str(e)

def create_dataset(year):
    """Create comprehensive dataset for a given year with proper error handling."""
    print(f"   🔄 Creating dataset for {year}...")
    
    try:
        # Get player weekly data
        players = nfl.import_weekly_data([year])
        print(f"      ✅ Players: {len(players):,} rows")
        
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
        
        players_df = players_df.rename(
            columns={'recent_team': 'player_team', 'season_type': 'game_type'},
        )

        # Get schedule data
        schedules = nfl.import_schedules([year])
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
        
        # Merge considering players as home team
        home_merge = players_df.merge(
            schedules_df,
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'home_team'],
            how='left',
        )

        # Merge considering players as away team
        away_merge = players_df.merge(
            schedules_df,
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'away_team'],
            how='left',
        )

        # Create home/away indicators
        home_merge['is_home_team'] = home_merge['home_team'].notna()
        away_merge['is_away_team'] = away_merge['away_team'].notna()

        # Combine the merges
        merged_df = pd.concat([home_merge, away_merge])

        # Select final columns
        final_columns = [
            'player_id',
            'player_display_name',
            'position',
            'recent_team',
            'home_team',
            'season',
            'week',
            'game_type',
            'fantasy_points',
            'fantasy_points_ppr',
            'game_id',
            'gameday',
            'away_team',
            'away_score',
            'home_score',
        ]
        final_df = merged_df[final_columns]

        # Clean up the data
        final_df = final_df[final_df['game_id'].notna()]
        final_df['player_team'] = final_df['recent_team']
        final_df = final_df[
            (
                (final_df['player_team'] == final_df['home_team'])
                | (final_df['player_team'] == final_df['away_team'])
            )
        ]
        final_df = final_df.drop_duplicates()

        # Add injury data if available
        try:
            print("      🔄 Adding injury data...")
            injuries = nfl.import_injuries([year])
            injuries = injuries[[
                'full_name',
                'position',
                'week',
                'season',
                'team',
                'game_type',
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
                on=['player_display_name', 'position', 'season', 'week', 'game_type', 'home_team'],
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
    
    with alive_bar(max_rows, title=f"Processing {year} Data") as bar:
        for i, row in final_df.iterrows():
            try:
                # Extract player details
                player_id = row['player_id']
                player_name = row['player_display_name']
                season = row['season']
                position = row['position']
                week = row['week']
                fantasy_points_ppr = row['fantasy_points_ppr']
                fantasy_points = row['fantasy_points']
                game_date = row['gameday']
                team = row['recent_team']
                home = row['home_team']
                away = row['away_team']
                opponent = away if team == away else home
                game_injury_report_status = row.get('game_injury_report_status')
                practice_injury_report_status = row.get('practice_injury_report_status')
                
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
                
                bar()
                
            except Exception as e:
                print(f"      ⚠️  Error processing row {i}: {e}")
                bar()
    
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
    os.makedirs('datasets', exist_ok=True)
    filename = f'datasets/{year}season.csv'
    processed_df.to_csv(filename, index=False)
    print(f"   💾 Saved to {filename}")
    
    return processed_df

def combine_datasets(directory_path, output_directory_path, years_to_process):
    """Combine all datasets into a single file."""
    print("\n🔗 Combining datasets...")
    
    # Get list of CSV files
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv') and 'season' in f]
    files = sorted(files)
    
    print(f"   Found {len(files)} season files")
    
    dfs = []
    with alive_bar(len(files), title="Combining Datasets") as bar:
        for f in files:
            try:
                file_path = os.path.join(directory_path, f)
                data = pd.read_csv(file_path)
                dfs.append(data)
                print(f"      ✅ {f}: {len(data):,} rows")
                bar()
            except Exception as e:
                print(f"      ❌ Error reading {f}: {e}")
                bar()
    
    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        os.makedirs(output_directory_path, exist_ok=True)
        output_file = os.path.join(output_directory_path, 'combined_data.csv')
        combined_df.to_csv(output_file, index=False)
        print(f"   💾 Combined dataset saved: {len(combined_df):,} rows")
        print(f"   📁 Check {output_directory_path} for combined dataset")
        return combined_df
    else:
        print("   ❌ No valid datasets to combine")
        return None

def main(process_years=None):
    """Main data collection function."""
    print("=" * 60)
    print("ENHANCED NFL DATA COLLECTION PIPELINE")
    print("=" * 60)
    
    if process_years is None:
        process_years = DEFAULT_YEARS
    
    start_time = time.time()
    
    # Create directory structure
    os.makedirs('datasets', exist_ok=True)
    existing_files = os.listdir('datasets')
    
    # Process each year
    successful_years = []
    for year in process_years:
        # Check if file already exists
        if any(file.startswith(str(year)) for file in existing_files):
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
    
    # Combine datasets
    if successful_years:
        combine_datasets('datasets', 'combined_datasets', successful_years)
    
    elapsed_time = time.time() - start_time
    print(f"\n🎯 Collection completed in {elapsed_time:.1f} seconds")
    print(f"✅ Successfully processed {len(successful_years)} years: {successful_years}")

if __name__ == '__main__':
    main()
