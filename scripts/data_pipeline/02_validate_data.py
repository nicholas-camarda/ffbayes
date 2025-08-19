#!/usr/bin/env python3
"""
02_validate_data.py - Data Validation Pipeline
Second step in the fantasy football analytics pipeline.
Validates data quality and completeness.
"""

import glob
import os
import time
from datetime import datetime

import pandas as pd
from alive_progress import alive_bar


def validate_data_quality():
    """Validate the quality and completeness of collected data."""
    print("🔍 Validating data quality...")
    
    # Check individual datasets
    player_files = glob.glob("datasets/*_players.csv")
    schedule_files = glob.glob("datasets/*_schedule.csv")
    
    validation_results = {
        'player_files': len(player_files),
        'schedule_files': len(schedule_files),
        'total_rows': 0,
        'missing_data': 0,
        'quality_score': 0
    }
    
    print(f"   📁 Found {len(player_files)} player datasets")
    print(f"   📁 Found {len(schedule_files)} schedule datasets")
    
    # Validate player data with progress bar
    if player_files:
        with alive_bar(len(player_files), title="Validating Player Data", bar="smooth") as bar:
            for file in sorted(player_files):
                try:
                    df = pd.read_csv(file)
                    year = file.split('_')[0].split('/')[-1]
                    validation_results['total_rows'] += len(df)
                    
                    # Check for missing data
                    missing_pct = (df.isnull().sum() / len(df) * 100).max()
                    if missing_pct > 50:
                        print(f"      ⚠️  {year}: High missing data ({missing_pct:.1f}%)")
                        validation_results['missing_data'] += 1
                    else:
                        print(f"      ✅ {year}: Good data quality ({missing_pct:.1f}% missing)")
                    
                    bar.text(f"Validated {year}")
                    bar()
                    
                except Exception as e:
                    print(f"      ❌ {file}: Error reading - {e}")
                    bar()
    
    # Calculate quality score
    if validation_results['player_files'] > 0:
        validation_results['quality_score'] = (
            (validation_results['player_files'] - validation_results['missing_data']) / 
            validation_results['player_files'] * 100
        )
    
    return validation_results

def check_data_completeness():
    """Check if we have all necessary data for analysis."""
    print("\n📊 Checking data completeness...")
    
    # Check for recent years
    current_year = datetime.now().year
    expected_years = list(range(2023, current_year + 1))
    
    missing_years = []
    for year in expected_years:
        player_file = f"datasets/{year}_players.csv"
        if not os.path.exists(player_file):
            missing_years.append(year)
    
    if missing_years:
        print(f"   ❌ Missing data for years: {missing_years}")
        return False
    else:
        print("   ✅ All expected recent years available")
        return True

def main():
    """Main data validation function."""
    print("=" * 60)
    print("DATA VALIDATION PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Validate data quality
    quality_results = validate_data_quality()
    
    # Check completeness
    is_complete = check_data_completeness()
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Validation completed in {elapsed_time:.1f} seconds")
    print(f"📊 Total rows: {quality_results['total_rows']:,}")
    print(f"🎯 Quality score: {quality_results['quality_score']:.1f}%")
    print(f"✅ Data complete: {is_complete}")
    
    if is_complete and quality_results['quality_score'] > 80:
        print("\n🎉 Data validation passed! Ready for processing.")
        print("🎯 Next step: Run 03_process_data.py")
    else:
        print("\n⚠️  Data validation issues found. Check data collection.")
        print("🔄 Re-run: 01_collect_data.py")

if __name__ == "__main__":
    main()
