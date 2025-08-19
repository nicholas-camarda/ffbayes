#!/usr/bin/env python3
"""
Quick Data Validation Script
Runs in under 2 minutes to check data completeness and quality.
"""

import glob
import time
from datetime import datetime

import pandas as pd


def main():
    print("=" * 60)
    print("QUICK DATA VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check individual datasets
    print("\n1. INDIVIDUAL DATASETS CHECK")
    print("-" * 30)
    
    dataset_files = glob.glob("datasets/*.csv")
    available_years = []
    
    for file in sorted(dataset_files):
        year = file.split('/')[-1].replace('season.csv', '')
        try:
            df = pd.read_csv(file)
            available_years.append(int(year))
            print(f"✅ {year}: {len(df):,} rows, {len(df.columns)} columns")
            
            # Quick quality check
            missing_pct = (df.isnull().sum() / len(df) * 100).max()
            if missing_pct > 50:
                print(f"   ⚠️  High missing data: {missing_pct:.1f}%")
                
        except Exception as e:
            print(f"❌ {year}: Error reading file - {e}")
    
    # Check combined datasets
    print("\n2. COMBINED DATASETS CHECK")
    print("-" * 30)
    
    combined_files = glob.glob("combined_datasets/*.csv")
    for file in sorted(combined_files):
        try:
            df = pd.read_csv(file)
            filename = file.split('/')[-1]
            print(f"✅ {filename}: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ {file}: Error reading file - {e}")
    
    # Check for missing recent data
    print("\n3. MISSING RECENT DATA ANALYSIS")
    print("-" * 30)
    
    current_year = datetime.now().year
    missing_years = []
    
    for year in range(2023, current_year + 1):
        if year not in available_years:
            missing_years.append(year)
            print(f"❌ Missing: {year} season data")
    
    if not missing_years:
        print("✅ All recent years (2023-2025) are available")
    else:
        print(f"⚠️  Missing {len(missing_years)} recent years: {missing_years}")
    
    # Check data freshness
    print("\n4. DATA FRESHNESS CHECK")
    print("-" * 30)
    
    if available_years:
        latest_year = max(available_years)
        print(f"📅 Latest available data: {latest_year}")
        
        if latest_year < current_year:
            print(f"⚠️  Data is {current_year - latest_year} year(s) old")
        else:
            print("✅ Data is current")
    
    # Quick test of data collection
    print("\n5. DATA COLLECTION TEST")
    print("-" * 30)
    
    try:
        import nfl_data_py as nfl
        print("✅ nfl_data_py is available")
        
        # Test if we can get recent data
        test_years = [2023, 2024, 2025]
        for year in test_years:
            try:
                data = nfl.import_weekly_data([year])
                print(f"✅ {year} data available via nfl_data_py: {len(data):,} rows")
            except Exception as e:
                print(f"❌ {year} data not available: {e}")
                
    except ImportError:
        print("❌ nfl_data_py not available")
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Validation completed in {elapsed_time:.1f} seconds")
    
    if missing_years:
        print(f"\n🚨 CRITICAL: Missing {len(missing_years)} recent years")
        print("   Next steps:")
        print("   1. Run: python scripts/get_ff_data.py")
        print("   2. Test with small dataset first")
        print("   3. Validate data quality")
    else:
        print("\n✅ All recent data available")
        print("   Next steps:")
        print("   1. Test Bayesian model with small dataset")
        print("   2. Add injury/weather data sources")
        print("   3. Optimize model performance")
    
    print(f"\n📊 Data coverage: {len(available_years)} years ({min(available_years)}-{max(available_years)})")
    print("🎯 Ready for quick model testing!")

if __name__ == "__main__":
    main()
