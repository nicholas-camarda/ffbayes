#!/usr/bin/env python3
"""
Centralized column standardization for fantasy football data.
Ensures consistent column names across all data sources and scripts.
"""

from typing import Dict, List

import pandas as pd

# Standard column names that all scripts should use
STANDARD_COLUMNS = {
    'player_name': 'Name',
    'position': 'Position', 
    'fantasy_points': 'Fantasy_Points',
    'vor_value': 'VOR_Value',
    'vor_rank': 'VOR_Rank',
    'adp': 'ADP',
    'team': 'Team',
    'season': 'Season',
    'games_played': 'Games_Played',
    'targets': 'Targets',
    'receptions': 'Receptions',
    'rushing_attempts': 'Rush_Attempts',
    'passing_attempts': 'Pass_Attempts',
    'touchdowns': 'Touchdowns',
    'interceptions': 'Interceptions',
    'fumbles': 'Fumbles'
}

# Column mappings for different data sources
COLUMN_MAPPINGS = {
    'vor_strategy': {
        'PLAYER': 'Name',
        'POS': 'Position',
        'FPTS': 'Fantasy_Points', 
        'VOR': 'VOR_Value',
        'VALUERANK': 'VOR_Rank',
        'Draft_Phase': 'Draft_Phase'
    },
    'unified_dataset': {
        'Name': 'Name',
        'Position': 'Position',
        'FantPt': 'Fantasy_Points',
        'Season': 'Season',
        'Tm': 'Team'
    },
    'hybrid_mc_results': {
        'name': 'Name',
        'position': 'Position',
        'mean': 'Mean_Projection',
        'std': 'Std_Projection'
    }
}


def standardize_columns(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Standardize column names for a given data source.
    
    Args:
        df: Input dataframe
        source_type: Type of data source ('vor_strategy', 'unified_dataset', etc.)
        
    Returns:
        Dataframe with standardized column names
    """
    if source_type not in COLUMN_MAPPINGS:
        print(f"⚠️  Warning: Unknown source type '{source_type}', returning original dataframe")
        return df
    
    mapping = COLUMN_MAPPINGS[source_type]
    
    # Create a copy to avoid modifying original
    df_std = df.copy()
    
    # Rename columns that exist in the mapping
    rename_dict = {}
    for old_name, new_name in mapping.items():
        if old_name in df_std.columns:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        df_std = df_std.rename(columns=rename_dict)
        print(f"✅ Standardized {len(rename_dict)} columns for {source_type}")
    else:
        print(f"⚠️  No columns found to standardize for {source_type}")
    
    return df_std


def get_standard_column_names() -> Dict[str, str]:
    """Get the standard column names mapping."""
    return STANDARD_COLUMNS.copy()


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], source_name: str = "dataframe") -> bool:
    """
    Validate that a dataframe has the required columns.
    
    Args:
        df: Dataframe to validate
        required_columns: List of required column names
        source_name: Name of the data source for error messages
        
    Returns:
        True if all required columns are present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns in {source_name}: {missing_columns}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    print(f"✅ All required columns present in {source_name}")
    return True


def suggest_column_mapping(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, str]:
    """
    Suggest column mappings based on similarity.
    
    Args:
        df: Source dataframe
        target_columns: Target column names
        
    Returns:
        Dictionary mapping source columns to target columns
    """
    suggestions = {}
    source_columns = list(df.columns)
    
    for target_col in target_columns:
        # Look for exact matches first
        if target_col in source_columns:
            suggestions[target_col] = target_col
            continue
        
        # Look for case-insensitive matches
        for source_col in source_columns:
            if source_col.lower() == target_col.lower():
                suggestions[target_col] = source_col
                break
        
        # Look for partial matches
        if target_col not in suggestions:
            for source_col in source_columns:
                if target_col.lower() in source_col.lower() or source_col.lower() in target_col.lower():
                    suggestions[target_col] = source_col
                    break
    
    return suggestions


# Common column validation functions
def validate_vor_strategy_columns(df: pd.DataFrame) -> bool:
    """Validate columns for VOR strategy data."""
    required = ['Name', 'Position', 'VOR_Value', 'VOR_Rank']
    return validate_required_columns(df, required, "VOR strategy")


def validate_player_data_columns(df: pd.DataFrame) -> bool:
    """Validate columns for player data."""
    required = ['Name', 'Position', 'Fantasy_Points']
    return validate_required_columns(df, required, "player data")


def validate_draft_strategy_columns(df: pd.DataFrame) -> bool:
    """Validate columns for draft strategy data."""
    required = ['Name', 'Position', 'VOR_Rank']
    return validate_required_columns(df, required, "draft strategy")


# Example usage and testing
if __name__ == "__main__":
    print("Column Standards Module")
    print("=" * 50)
    
    # Test standardization
    test_data = pd.DataFrame({
        'PLAYER': ['Saquon Barkley', 'Bijan Robinson'],
        'POS': ['RB', 'RB'],
        'FPTS': [294.65, 293.25],
        'VOR': [164.30, 162.90],
        'VALUERANK': [1, 2]
    })
    
    print("Original columns:", list(test_data.columns))
    
    standardized = standardize_columns(test_data, 'vor_strategy')
    print("Standardized columns:", list(standardized.columns))
    
    # Test validation
    validate_vor_strategy_columns(standardized)
