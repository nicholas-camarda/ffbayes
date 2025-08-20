#!/usr/bin/env python3
"""
03_preprocess_analysis_data.py - Data Preprocessing for Analysis
Third step in the fantasy football analytics pipeline.
Prepares data specifically for analysis scripts.
"""

import glob
import os

# Add scripts/utils to path for progress monitoring
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd() / 'scripts' / 'utils'))
try:
    from progress_monitor import ProgressMonitor
except ImportError:
    ProgressMonitor = None


def create_analysis_dataset(path_to_data_directory):
    """Create and preprocess the fantasy football dataset for analysis."""
    print("Loading and preprocessing data for analysis...")
    
    # Read in the datasets and combine - look for season.csv files in season_datasets subdirectory
    all_files = glob.glob(os.path.join(path_to_data_directory, 'season_datasets', '*season.csv'))
    if not all_files:
        raise ValueError(f"No season data files found in {path_to_data_directory}/season_datasets/")
    
    print(f"Found {len(all_files)} season files: {[os.path.basename(f) for f in all_files]}")
    
    data_temp = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    print(f"Combined data shape: {data_temp.shape}")
    print(f"Available columns: {data_temp.columns.tolist()}")
    print(f"Season range: {data_temp['Season'].min()} - {data_temp['Season'].max()}")
    
    # Sort properly
    data = data_temp.sort_values(
        by=['Season', 'Name', 'G#'], ascending=[True, True, True]
    )

    # One-hot-encode the positions
    data['pos_id'] = data['Position']
    data['position'] = data['Position']
    data = pd.get_dummies(data, columns=['position'])

    # Identify teams with integer encoding
    ids = np.array([k for k in data['Opp'].unique()])
    team_names = ids.copy()
    data['opp_team'] = data['Opp'].apply(lambda x: np.where(x == ids)[0][0])
    data['team'] = data['Tm'].apply(lambda x: np.where(x == ids)[0][0])

    # Create home/away indicator - Away column contains team names, so if Away == Tm, it's away
    data['is_home'] = (data['Away'] != data['Tm']).astype(int)

    # Position encoding
    pos_ids = np.array([k for k in data['pos_id'].unique()])
    pos_ids_nonan = pos_ids[np.where(pos_ids != 'nan')]
    onehot_pos_ids = list(map(int, data['pos_id'].isin(pos_ids_nonan)))
    data['pos_id'] = onehot_pos_ids

    # Calculate seven game rolling average
    num_day_roll_avg = 7
    data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].transform(
        lambda x: x.rolling(num_day_roll_avg, min_periods=num_day_roll_avg).mean()
    )

    # Rank based on the 7-game average
    ranks = data.groupby(['Name', 'Season'])['7_game_avg'].rank(
        pct=False, method='average'
    )
    quartile_ranks = pd.qcut(ranks, 4, labels=False, duplicates='drop')
    data['rank'] = quartile_ranks.tolist()

    data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

    # Remove only rows with critical missing data, not all NA
    # Keep rows where we have the essential columns
    essential_cols = ['Name', 'Position', 'Season', 'FantPt', '7_game_avg']
    data = data.dropna(subset=essential_cols)
    
    # Convert rank to integer, handling NaN values
    data['rank'] = data['rank'].fillna(0).astype(int)
    
    print(f"After cleaning, data shape: {data.shape}")
    print(f"Season range after cleaning: {data['Season'].min()} - {data['Season'].max()}")

    # Save combined dataset
    output_dir = 'datasets/combined_datasets'
    output_file = os.path.join(output_dir, 'combined_ff_data.csv')
    
    data.to_csv(output_file, index=False)
    print(f"Combined dataset saved to: {output_file}")
    print(f"Shape: {data.shape}")
    
    return data, team_names


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("DATA PREPROCESSING FOR ANALYSIS")
    print("=" * 60)
    
    # Preprocess data for analysis
    data, team_names = create_analysis_dataset('datasets')
    
    print("\n✅ Preprocessing completed!")
    print(f"📊 Final dataset: {data.shape}")
    print(f"🏈 Teams: {len(team_names)}")
    print("🎯 Next step: Run analysis scripts")


if __name__ == "__main__":
    main()
