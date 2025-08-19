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
    season_files = glob.glob("datasets/season_datasets/*season.csv")
    
    validation_results = {
        'season_files': len(season_files),
        'total_rows': 0,
        'missing_data': 0,
        'quality_score': 100
    }
    
    print(f"   📁 Found {len(season_files)} season datasets")
    
    # Validate season data with progress bar
    if season_files:
        with alive_bar(len(season_files), title="Validating Season Data", bar="smooth") as bar:
            for file in sorted(season_files):
                try:
                    df = pd.read_csv(file)
                    year = file.split('season.csv')[0].split('/')[-1]
                    validation_results['total_rows'] += len(df)
                    
                    # Check for missing data (exclude injury status columns which are expected to be missing)
                    # Focus on core fantasy football data columns
                    core_columns = ['G#', 'Date', 'Tm', 'Away', 'Opp', 'FantPt', 'FantPtPPR', 'Name', 'PlayerID', 'Position', 'Season', 'is_home']
                    df_core = df[core_columns]
                    
                    missing_counts = df_core.isna().sum()
                    missing_pct = (missing_counts / len(df_core) * 100).max()
                    
                    if missing_pct > 10:  # Lower threshold since we're only checking core columns
                        print(f"      ⚠️  {year}: High missing data in core columns ({missing_pct:.1f}%)")
                        validation_results['missing_data'] += 1
                    else:
                        print(f"      ✅ {year}: Good data quality ({missing_pct:.1f}% missing in core columns)")
                    
                    bar.text(f"Validated {year}")
                    bar()
                    
                except Exception as e:
                    print(f"      ❌ {file}: Error reading - {e}")
                    bar()
    
    # Calculate quality score
    if validation_results['season_files'] > 0:
        validation_results['quality_score'] = (
            (validation_results['season_files'] - validation_results['missing_data']) / 
            validation_results['season_files'] * 100
        )
    
    return validation_results

def check_data_completeness():
    """Check if we have all necessary data for analysis."""
    print("\n📊 Checking data completeness...")
    
    # Check for recent years (last 10 available years)
    current_year = datetime.now().year
    expected_years = list(range(current_year - 10, current_year))
    
    missing_years = []
    for year in expected_years:
        season_file = f"datasets/season_datasets/{year}season.csv"
        if not os.path.exists(season_file):
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
        print("🎯 Next step: Run analysis scripts")
    else:
        print("\n⚠️  Data validation issues found. Check data collection.")
        print("🔄 Re-run: 01_collect_data.py")

if __name__ == "__main__":
    main()
