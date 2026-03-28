#!/usr/bin/env python3
"""
Hybrid Data Integration Module

This module handles the integration of VOR (Value Over Replacement) data with 
Hybrid Monte Carlo/Bayesian uncertainty data to create a comprehensive hybrid dataset
for the draft strategy Excel file.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


def load_vor_data(vor_file_path: str) -> pd.DataFrame:
    """
    Load VOR data from CSV file.
    
    Args:
        vor_file_path: Path to VOR CSV file
        
    Returns:
        DataFrame with VOR data
        
    Raises:
        FileNotFoundError: If VOR file doesn't exist
        ValueError: If VOR file is invalid
    """
    logger.info(f"Loading VOR data from: {vor_file_path}")
    
    if not Path(vor_file_path).exists():
        raise FileNotFoundError(f"VOR file not found: {vor_file_path}")
    
    try:
        vor_data = pd.read_csv(vor_file_path)
        
        # Validate required columns
        required_columns = ['PLAYER', 'POS', 'FPTS', 'VOR', 'VALUERANK']
        missing_columns = [col for col in required_columns if col not in vor_data.columns]
        
        if missing_columns:
            raise ValueError(f"VOR file missing required columns: {missing_columns}")
        
        logger.info(f"✅ Loaded VOR data: {len(vor_data)} players")
        return vor_data
        
    except Exception as e:
        logger.error(f"Failed to load VOR data: {e}")
        raise ValueError(f"Invalid VOR file: {e}")


def load_bayesian_data(bayesian_file_path: str) -> Dict:
    """
    Load Bayesian uncertainty data from JSON file.
    
    Args:
        bayesian_file_path: Path to Bayesian JSON file
        
    Returns:
        Dictionary with Bayesian data
        
    Raises:
        FileNotFoundError: If Bayesian file doesn't exist
        ValueError: If Bayesian file is invalid
    """
    logger.info(f"Loading Bayesian data from: {bayesian_file_path}")
    
    if not Path(bayesian_file_path).exists():
        raise FileNotFoundError(f"Bayesian file not found: {bayesian_file_path}")
    
    try:
        with open(bayesian_file_path, 'r') as f:
            bayesian_data = json.load(f)
        
        # Validate data structure
        if not isinstance(bayesian_data, dict):
            raise ValueError("Bayesian data must be a dictionary")
        
        # Check that it contains player data
        if not bayesian_data:
            raise ValueError("Bayesian data is empty")
        
        # Validate first player's structure
        first_player = list(bayesian_data.keys())[0]
        player_data = bayesian_data[first_player]
        
        if not isinstance(player_data, dict):
            raise ValueError("Player data must be a dictionary")
        
        logger.info(f"✅ Loaded Bayesian data: {len(bayesian_data)} players")
        return bayesian_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Bayesian JSON: {e}")
        raise ValueError(f"Invalid Bayesian JSON file: {e}")
    except Exception as e:
        logger.error(f"Failed to load Bayesian data: {e}")
        raise ValueError(f"Invalid Bayesian file: {e}")


def create_hybrid_dataset(vor_data: pd.DataFrame, bayesian_data: Dict) -> pd.DataFrame:
    """
    Create hybrid dataset by combining VOR rankings with Bayesian uncertainty scores.
    
    Args:
        vor_data: DataFrame with VOR data
        bayesian_data: Dictionary with Bayesian uncertainty data
        
    Returns:
        DataFrame with combined VOR and Bayesian data
    """
    logger.info("Creating hybrid dataset...")
    
    # Create a copy of VOR data to avoid modifying original
    hybrid_data = vor_data.copy()
    
    # Initialize new columns for Bayesian data
    hybrid_data['uncertainty_score'] = 0.5  # Default uncertainty
    hybrid_data['mean_projection'] = hybrid_data['FPTS']  # Default to VOR projection
    hybrid_data['std_projection'] = 0.0  # Default standard deviation
    hybrid_data['bayesian_rank'] = hybrid_data['VALUERANK']  # Default to VOR rank
    
    # Match players and add Bayesian data
    matched_count = 0
    for idx, row in hybrid_data.iterrows():
        player_name = row['PLAYER']
        
        if player_name in bayesian_data:
            player_bayes = bayesian_data[player_name]
            
            # Extract Monte Carlo data
            if 'monte_carlo' in player_bayes:
                mc_data = player_bayes['monte_carlo']
                hybrid_data.at[idx, 'mean_projection'] = mc_data.get('mean', row['FPTS'])
                hybrid_data.at[idx, 'std_projection'] = mc_data.get('std', 0.0)
            
            # Extract Bayesian uncertainty
            if 'bayesian_uncertainty' in player_bayes:
                bayes_data = player_bayes['bayesian_uncertainty']
                hybrid_data.at[idx, 'uncertainty_score'] = bayes_data.get('overall_uncertainty', 0.5)
            
            # Extract VOR validation rank if available
            if 'vor_validation' in player_bayes:
                vor_val = player_bayes['vor_validation']
                hybrid_data.at[idx, 'bayesian_rank'] = vor_val.get('global_rank', row['VALUERANK'])
            
            matched_count += 1
        else:
            logger.warning(f"Player not found in Bayesian data: {player_name}")
    
    logger.info(f"✅ Created hybrid dataset: {len(hybrid_data)} players, {matched_count} matched")
    
    # Add composite score that combines VOR and uncertainty
    hybrid_data['composite_score'] = (
        hybrid_data['VOR'] * (1 - hybrid_data['uncertainty_score'])
    )
    
    # Sort by composite score for better rankings
    hybrid_data = hybrid_data.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    return hybrid_data


def validate_hybrid_data(hybrid_data: pd.DataFrame) -> None:
    """
    Validate hybrid dataset for required columns and data quality.
    
    Args:
        hybrid_data: DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating hybrid dataset...")
    
    if hybrid_data.empty:
        raise ValueError("Hybrid dataset is empty")
    
    # Check required columns
    required_columns = [
        'PLAYER', 'POS', 'FPTS', 'VOR', 'VALUERANK',
        'uncertainty_score', 'mean_projection', 'std_projection', 'composite_score'
    ]
    
    missing_columns = [col for col in required_columns if col not in hybrid_data.columns]
    if missing_columns:
        raise ValueError(f"Hybrid dataset missing required columns: {missing_columns}")
    
    # Check data types
    if hybrid_data['VOR'].dtype not in ['float64', 'int64']:
        raise ValueError("VOR column must be numeric")
    
    if hybrid_data['uncertainty_score'].dtype not in ['float64', 'int64']:
        raise ValueError("uncertainty_score column must be numeric")
    
    # Check for reasonable uncertainty scores (0-1 range)
    invalid_uncertainty = hybrid_data[
        (hybrid_data['uncertainty_score'] < 0) | 
        (hybrid_data['uncertainty_score'] > 1)
    ]
    
    if not invalid_uncertainty.empty:
        logger.warning(f"Found {len(invalid_uncertainty)} players with invalid uncertainty scores")
    
    logger.info("✅ Hybrid dataset validation passed")


def load_hybrid_data_sources() -> tuple[pd.DataFrame, Dict]:
    """
    Load both VOR and Bayesian data sources using default paths.
    
    Returns:
        Tuple of (vor_data, bayesian_data)
        
    Raises:
        FileNotFoundError: If either data source is missing
    """
    from datetime import datetime

    from ffbayes.utils.path_constants import get_hybrid_mc_dir, get_vor_strategy_dir
    
    current_year = datetime.now().year
    
    # Get VOR file path
    vor_dir = get_vor_strategy_dir(current_year)
    vor_file = vor_dir / f"snake-draft_ppr-0.5_vor_top-120_{current_year}.csv"
    
    # Get Bayesian file path
    bayesian_dir = get_hybrid_mc_dir(current_year)
    bayesian_file = bayesian_dir / "hybrid_model_results.json"
    
    # Load data
    vor_data = load_vor_data(str(vor_file))
    bayesian_data = load_bayesian_data(str(bayesian_file))
    
    return vor_data, bayesian_data


def main():
    """Main function for testing the hybrid data integration."""
    try:
        # Load data sources
        vor_data, bayesian_data = load_hybrid_data_sources()
        
        # Create hybrid dataset
        hybrid_data = create_hybrid_dataset(vor_data, bayesian_data)
        
        # Validate
        validate_hybrid_data(hybrid_data)
        
        print("🎉 Hybrid data integration successful!")
        print(f"📊 Hybrid dataset: {len(hybrid_data)} players")
        print("📈 Top 5 players by composite score:")
        for i, row in hybrid_data.head().iterrows():
            print(f"   {i+1}. {row['PLAYER']} ({row['POS']}) - VOR: {row['VOR']:.1f}, Uncertainty: {row['uncertainty_score']:.2f}")
        
        return hybrid_data
        
    except Exception as e:
        logger.error(f"❌ Hybrid data integration failed: {e}")
        raise


if __name__ == "__main__":
    main()
