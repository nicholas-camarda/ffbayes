#!/usr/bin/env python3
"""
Quick Recent Data Collection
Collect only 2023-2025 data for fast testing and validation.
"""

import os
import time

import nfl_data_py as nfl
import pandas as pd


def main():
    print("=" * 60)
    print("QUICK RECENT DATA COLLECTION (2023-2025)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Only collect recent years for quick testing
    recent_years = [2023, 2024, 2025]
    
    print(f"Collecting data for years: {recent_years}")
    
    for year in recent_years:
        print(f"\n📊 Processing {year}...")
        
        try:
            # Get player data
            players = nfl.import_weekly_data([year])
            print(f"   ✅ Players: {len(players):,} rows")
            
            # Get schedule data
            schedules = nfl.import_schedules([year])
            print(f"   ✅ Schedule: {len(schedules):,} rows")
            
            # Save individual year data
            players.to_csv(f"datasets/{year}season_recent.csv", index=False)
            schedules.to_csv(f"datasets/{year}schedule_recent.csv", index=False)
            
            print(f"   💾 Saved {year} data")
            
        except Exception as e:
            print(f"   ❌ Error processing {year}: {e}")
    
    # Create a small combined dataset for quick testing
    print("\n🔗 Creating small combined dataset...")
    
    combined_data = []
    for year in recent_years:
        try:
            file_path = f"datasets/{year}season_recent.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_data.append(df)
                print(f"   ✅ Added {year}: {len(df):,} rows")
        except Exception as e:
            print(f"   ❌ Error reading {year}: {e}")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv("combined_datasets/recent_2023_2025.csv", index=False)
        print(f"   💾 Saved combined dataset: {len(combined_df):,} rows")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Collection completed in {elapsed_time:.1f} seconds")
    
    print("\n🎯 Next steps:")
    print("   1. Test Bayesian model with recent data")
    print("   2. Validate data quality")
    print("   3. Compare predictions with known outcomes")

if __name__ == "__main__":
    main()
