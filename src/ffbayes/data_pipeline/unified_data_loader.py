#!/usr/bin/env python3
"""
unified_data_loader.py - Unified Data Loading Utility
Simple module for all models to load the unified dataset.

Usage:
    from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset
    
    data = load_unified_dataset()
"""


import pandas as pd


def load_unified_dataset(data_directory='datasets'):
    """Load the unified dataset created by create_unified_dataset.py."""
    from ffbayes.utils.path_constants import get_unified_dataset_path
    dataset_path = get_unified_dataset_path()
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Unified dataset not found at {dataset_path}. "
            "Run create_unified_dataset.py first."
        )
    
    data = pd.read_json(dataset_path)
    return data

def get_player_data(player_name, data=None):
    """Get all data for a specific player from the unified dataset."""
    if data is None:
        data = load_unified_dataset()
    
    player_data = data[data['Name'] == player_name]
    
    if len(player_data) == 0:
        raise ValueError(f"Player '{player_name}' not found in unified dataset")
    
    return player_data

def get_position_data(position, data=None):
    """Get all data for a specific position from the unified dataset."""
    if data is None:
        data = load_unified_dataset()
    
    position_data = data[data['Position'] == position]
    
    if len(position_data) == 0:
        raise ValueError(f"Position '{position}' not found in unified dataset")
    
    return position_data

def get_season_data(season, data=None):
    """Get all data for a specific season from the unified dataset."""
    if data is None:
        data = load_unified_dataset()
    
    season_data = data[data['Season'] == season]
    
    if len(season_data) == 0:
        raise ValueError(f"Season {season} not found in unified dataset")
    
    return season_data
