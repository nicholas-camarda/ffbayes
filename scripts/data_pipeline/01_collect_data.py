#!/usr/bin/env python3
"""
01_collect_data.py - Data Collection Pipeline
First step in the fantasy football analytics pipeline.
Collects raw NFL data from multiple sources.
"""

import os
import time
from datetime import datetime

import nfl_data_py as nfl
from alive_progress import alive_bar


def collect_nfl_data(years=None):
    """Collect NFL data for specified years."""
    if years is None:
        current_year = datetime.now().year
        years = list(range(2015, current_year + 1))
    
    print(f"📊 Collecting NFL data for years: {years}")
    
    all_data = []
    
    # Use alive_progress for better progress monitoring
    with alive_bar(len(years), title="Collecting NFL Data", bar="smooth") as bar:
        for year in years:
            try:
                bar.text(f"Processing {year}...")
                
                # Get player weekly data
                players = nfl.import_weekly_data([year])
                print(f"      ✅ {year} Players: {len(players):,} rows")
                
                # Get schedule data
                schedules = nfl.import_schedules([year])
                print(f"      ✅ {year} Schedule: {len(schedules):,} rows")
                
                # Save individual year data
                os.makedirs("datasets", exist_ok=True)
                players.to_csv(f"datasets/{year}_players.csv", index=False)
                schedules.to_csv(f"datasets/{year}_schedule.csv", index=False)
                
                all_data.append(players)
                print(f"      💾 Saved {year} data")
                
                bar()  # Update progress bar
                
            except Exception as e:
                print(f"      ❌ Error processing {year}: {e}")
                bar()  # Still update progress bar even on error
    
    return all_data

def main():
    """Main data collection function."""
    print("=" * 60)
    print("NFL DATA COLLECTION PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Collect data for recent years (quick test)
    recent_years = [2023, 2024]  # Start with recent years for testing
    data = collect_nfl_data(recent_years)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Collection completed in {elapsed_time:.1f} seconds")
    print(f"📊 Total data collected: {len(data)} years")
    
    print("\n🎯 Next step: Run 02_validate_data.py")

if __name__ == "__main__":
    main()
