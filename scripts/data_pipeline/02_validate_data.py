#!/usr/bin/env python3
"""
02_validate_data.py - Data Validation Pipeline
Second step in the fantasy football analytics pipeline.
Validates data quality and completeness.
"""

import glob
import os

# Add scripts/utils to path for progress monitoring
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from alive_progress import alive_bar

sys.path.append(str(Path.cwd() / 'scripts' / 'utils'))
try:
    # this is a custom progress monitor that is used to monitor the progress of the data validation
    from progress_monitor import ProgressMonitor
except ImportError:
    ProgressMonitor = None


def validate_data_quality():
    """Validate the quality and completeness of collected data."""
    print("🔍 Validating data quality...")
    
    # Check individual datasets
    season_files = glob.glob("datasets/season_datasets/*season.csv")
    
    validation_results = {
        'season_files': len(season_files),
        'total_rows': 0,
        'missing_data': 0,
        'quality_score': 100,
        'errors': [],
        'warnings': [],
        'data_consistency': [],
        'statistical_checks': [],
        'outlier_detection': []
    }
    
    print(f"   📁 Found {len(season_files)} season datasets")
    
    if not season_files:
        validation_results['errors'].append("No season datasets found")
        return validation_results
    
    # Use enhanced progress monitoring if available
    if ProgressMonitor:
        monitor = ProgressMonitor("Data Validation")
        monitor.start_timer()
        
        with monitor.monitor(len(season_files), "Validating Season Data"):
            for file in sorted(season_files):
                try:
                    df = pd.read_csv(file)
                    year = file.split('season.csv')[0].split('/')[-1]
                    validation_results['total_rows'] += len(df)
                    
                    # Check for missing data (exclude injury status columns which are expected to be missing)
                    # Focus on core fantasy football data columns
                    core_columns = ['G#', 'Date', 'Tm', 'Away', 'Opp', 'FantPt', 'FantPtPPR', 'Name', 'PlayerID', 'Position', 'Season', 'is_home']
                    
                    # Check if all core columns exist
                    missing_cols = [col for col in core_columns if col not in df.columns]
                    if missing_cols:
                        error_msg = f"{year}: Missing core columns: {missing_cols}"
                        validation_results['errors'].append(error_msg)
                        print(f"      ❌ {error_msg}")
                        continue
                    
                    df_core = df[core_columns]
                    
                    missing_counts = df_core.isna().sum()
                    missing_pct = (missing_counts / len(df_core) * 100).max()
                    
                    if missing_pct > 10:  # Lower threshold since we're only checking core columns
                        warning_msg = f"{year}: High missing data in core columns ({missing_pct:.1f}%)"
                        validation_results['warnings'].append(warning_msg)
                        print(f"      ⚠️  {warning_msg}")
                        validation_results['missing_data'] += 1
                    else:
                        print(f"      ✅ {year}: Good data quality ({missing_pct:.1f}% missing in core columns)")
                    
                    # Enhanced validation checks
                    validation_results = perform_enhanced_validation(df, year, validation_results)
                    
                except Exception as e:
                    error_msg = f"{file}: Error reading - {e}"
                    validation_results['errors'].append(error_msg)
                    print(f"      ❌ {error_msg}")
    else:
        # Fallback to basic progress bar
        with alive_bar(len(season_files), title="Validating Season Data", bar="smooth") as bar:
            for file in sorted(season_files):
                try:
                    df = pd.read_csv(file)
                    year = file.split('season.csv')[0].split('/')[-1]
                    validation_results['total_rows'] += len(df)
                    
                    # Check for missing data (exclude injury status columns which are expected to be missing)
                    # Focus on core fantasy football data columns
                    core_columns = ['G#', 'Date', 'Tm', 'Away', 'Opp', 'FantPt', 'FantPtPPR', 'Name', 'PlayerID', 'Position', 'Season', 'is_home']
                    
                    # Check if all core columns exist
                    missing_cols = [col for col in core_columns if col not in df.columns]
                    if missing_cols:
                        error_msg = f"{year}: Missing core columns: {missing_cols}"
                        validation_results['errors'].append(error_msg)
                        print(f"      ❌ {error_msg}")
                        bar()
                        continue
                    
                    df_core = df[core_columns]
                    
                    missing_counts = df_core.isna().sum()
                    missing_pct = (missing_counts / len(df_core) * 100).max()
                    
                    if missing_pct > 10:  # Lower threshold since we're only checking core columns
                        warning_msg = f"{year}: High missing data in core columns ({missing_pct:.1f}%)"
                        validation_results['warnings'].append(warning_msg)
                        print(f"      ⚠️  {warning_msg}")
                        validation_results['missing_data'] += 1
                    else:
                        print(f"      ✅ {year}: Good data quality ({missing_pct:.1f}% missing in core columns)")
                    
                    # Enhanced validation checks
                    validation_results = perform_enhanced_validation(df, year, validation_results)
                    
                    bar.text(f"Validated {year}")
                    bar()
                    
                except Exception as e:
                    error_msg = f"{file}: Error reading - {e}"
                    validation_results['errors'].append(error_msg)
                    print(f"      ❌ {error_msg}")
                    bar()
    
    # Calculate quality score
    if validation_results['season_files'] > 0:
        validation_results['quality_score'] = (
            (validation_results['season_files'] - validation_results['missing_data']) / 
            validation_results['season_files'] * 100
        )
    
    return validation_results

def perform_enhanced_validation(df, year, validation_results):
    """Perform enhanced validation checks on the dataset."""
    try:
        # Data consistency checks
        consistency_checks = perform_data_consistency_checks(df, year)
        validation_results['data_consistency'].extend(consistency_checks)
        
        # Statistical validation
        statistical_checks = perform_statistical_validation(df, year)
        validation_results['statistical_checks'].extend(statistical_checks)
        
        # Outlier detection
        outlier_checks = perform_outlier_detection(df, year)
        validation_results['outlier_detection'].extend(outlier_checks)
        
    except Exception as e:
        error_msg = f"{year}: Enhanced validation error - {e}"
        validation_results['errors'].append(error_msg)
        print(f"      ❌ {error_msg}")
    
    return validation_results


def perform_data_consistency_checks(df, year):
    """Check data consistency across the dataset."""
    consistency_checks = []
    
    try:
        # Check date format consistency
        if 'Date' in df.columns:
            date_errors = 0
            for date_str in df['Date'].dropna():
                try:
                    pd.to_datetime(date_str)
                except:
                    date_errors += 1
            
            if date_errors > 0:
                consistency_checks.append(f"{year}: {date_errors} invalid date formats")
        
        # Check numeric column consistency
        numeric_columns = ['FantPt', 'FantPtPPR', 'G#', 'Season']
        for col in numeric_columns:
            if col in df.columns:
                non_numeric = df[col].apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float))).sum()
                if non_numeric > 0:
                    consistency_checks.append(f"{year}: {non_numeric} non-numeric values in {col}")
        
        # Check PlayerID separately (it can be string or numeric)
        if 'PlayerID' in df.columns:
            # PlayerID can be string or numeric, so we don't flag it as an error
            pass
        
        # Check position consistency
        if 'Position' in df.columns:
            # Include both offensive and defensive positions
            valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'CB', 'FS', 'SS', 'ILB', 'OLB', 'MLB', 'DE', 'DT', 'NT', 'G', 'C', 'T', 'OT', 'P', 'FB', 'HB', 'DB', 'S']
            invalid_positions = df[~df['Position'].isin(valid_positions)]['Position'].unique()
            if len(invalid_positions) > 0:
                consistency_checks.append(f"{year}: Invalid positions found: {invalid_positions}")
        
        # Check team abbreviation consistency
        if 'Tm' in df.columns:
            # NFL teams should be 2-3 characters
            invalid_teams = df[df['Tm'].str.len() > 3]['Tm'].unique()
            if len(invalid_teams) > 0:
                consistency_checks.append(f"{year}: Invalid team abbreviations: {invalid_teams}")
        
    except Exception as e:
        consistency_checks.append(f"{year}: Consistency check error - {e}")
    
    return consistency_checks


def perform_statistical_validation(df, year):
    """Perform statistical validation checks."""
    statistical_checks = []
    
    try:
        # Check fantasy point ranges
        if 'FantPt' in df.columns:
            # Ensure FantPt is numeric
            df_fant_pt = pd.to_numeric(df['FantPt'], errors='coerce')
            fant_pt_stats = df_fant_pt.describe()
            
            # Check for reasonable fantasy point ranges
            if fant_pt_stats['min'] < -10:  # Negative points possible for fumbles, etc.
                statistical_checks.append(f"{year}: Unusually low fantasy points: {fant_pt_stats['min']:.2f}")
            
            if fant_pt_stats['max'] > 100:  # Very high fantasy points
                statistical_checks.append(f"{year}: Unusually high fantasy points: {fant_pt_stats['max']:.2f}")
            
            # Check for reasonable standard deviation
            if fant_pt_stats['std'] > 50:
                statistical_checks.append(f"{year}: High fantasy point variance: {fant_pt_stats['std']:.2f}")
        
        # Check player ID ranges (only if numeric)
        if 'PlayerID' in df.columns:
            try:
                df_player_id = pd.to_numeric(df['PlayerID'], errors='coerce')
                if not df_player_id.isna().all():  # Only check if we have numeric values
                    player_id_stats = df_player_id.describe()
                    if player_id_stats['min'] < 0:
                        statistical_checks.append(f"{year}: Negative player IDs found")
                    
                    if player_id_stats['max'] > 999999:  # Reasonable upper limit
                        statistical_checks.append(f"{year}: Unusually high player IDs: {player_id_stats['max']}")
            except:
                # Skip PlayerID statistical validation if conversion fails
                pass
        
        # Check season consistency
        if 'Season' in df.columns:
            try:
                df_season = pd.to_numeric(df['Season'], errors='coerce')
                if not df_season.isna().all():  # Only check if we have numeric values
                    unique_seasons = df_season.unique()
                    if len(unique_seasons) > 1:
                        statistical_checks.append(f"{year}: Multiple seasons in single file: {unique_seasons}")
                    
                    for season in unique_seasons:
                        if pd.notna(season) and (season < 1990 or season > datetime.now().year + 1):
                            statistical_checks.append(f"{year}: Invalid season year: {season}")
            except:
                # Skip season validation if conversion fails
                pass
        
    except Exception as e:
        statistical_checks.append(f"{year}: Statistical validation error - {e}")
    
    return statistical_checks


def perform_outlier_detection(df, year):
    """Detect outliers in the dataset."""
    outlier_checks = []
    
    try:
        # Fantasy point outliers using IQR method
        if 'FantPt' in df.columns:
            # Ensure FantPt is numeric
            df_fant_pt = pd.to_numeric(df['FantPt'], errors='coerce')
            Q1 = df_fant_pt.quantile(0.25)
            Q3 = df_fant_pt.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_fant_pt[(df_fant_pt < lower_bound) | (df_fant_pt > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df_fant_pt)) * 100
                if outlier_percentage > 5:  # Flag if more than 5% are outliers
                    outlier_checks.append(f"{year}: {len(outliers)} fantasy point outliers ({outlier_percentage:.1f}%)")
        
        # Game number outliers
        if 'G#' in df.columns:
            # Ensure G# is numeric
            df_game = pd.to_numeric(df['G#'], errors='coerce')
            game_stats = df_game.describe()
            if game_stats['max'] > 20:  # NFL regular season is 18 games max
                outlier_checks.append(f"{year}: Unusually high game numbers: {game_stats['max']}")
        
        # Player ID outliers (only check if numeric)
        if 'PlayerID' in df.columns:
            # Try to convert to numeric, skip if not possible
            try:
                df_player_id = pd.to_numeric(df['PlayerID'], errors='coerce')
                if not df_player_id.isna().all():  # Only check if we have numeric values
                    player_id_outliers = df_player_id[df_player_id > 999999].count()
                    if player_id_outliers > 0:
                        outlier_checks.append(f"{year}: {player_id_outliers} unusually high player IDs")
            except:
                # Skip PlayerID outlier detection if conversion fails
                pass
        
    except Exception as e:
        outlier_checks.append(f"{year}: Outlier detection error - {e}")
    
    return outlier_checks


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
    
    # Enhanced error and warning reporting
    if quality_results['errors']:
        print(f"\n❌ ERRORS FOUND ({len(quality_results['errors'])}):")
        for error in quality_results['errors']:
            print(f"   • {error}")
    
    if quality_results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(quality_results['warnings'])}):")
        for warning in quality_results['warnings']:
            print(f"   • {warning}")
    
    # Enhanced validation results
    if quality_results['data_consistency']:
        print(f"\n🔍 DATA CONSISTENCY ISSUES ({len(quality_results['data_consistency'])}):")
        for issue in quality_results['data_consistency']:
            print(f"   • {issue}")
    
    if quality_results['statistical_checks']:
        print(f"\n📊 STATISTICAL VALIDATION ISSUES ({len(quality_results['statistical_checks'])}):")
        for issue in quality_results['statistical_checks']:
            print(f"   • {issue}")
    
    if quality_results['outlier_detection']:
        print(f"\n📈 OUTLIER DETECTION ({len(quality_results['outlier_detection'])}):")
        for issue in quality_results['outlier_detection']:
            print(f"   • {issue}")
    
    # Final status
    if is_complete and quality_results['quality_score'] > 80 and not quality_results['errors']:
        print("\n🎉 Data validation passed! Ready for processing.")
        print("🎯 Next step: Run analysis scripts")
    elif quality_results['errors']:
        print("\n❌ Critical validation errors found. Fix data collection issues first.")
        print("🔄 Re-run: 01_collect_data.py")
    else:
        print("\n⚠️  Data validation issues found. Check data collection.")
        print("🔄 Re-run: 01_collect_data.py")

if __name__ == "__main__":
    main()
