#!/usr/bin/env python3
"""
Risk-Adjusted Ranking System

This module implements risk-adjusted rankings that combine VOR value with 
Bayesian uncertainty analysis, incorporating user risk tolerance preferences.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


def load_risk_tolerance_config(config_file_path: str = None) -> str:
    """
    Load risk tolerance setting from user configuration.
    
    Args:
        config_file_path: Path to user config file. If None, uses default path.
        
    Returns:
        Risk tolerance setting: 'low', 'medium', or 'high'
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If risk tolerance setting is invalid
    """
    if config_file_path is None:
        from ffbayes.utils.path_constants import CONFIG_DIR
        config_file_path = CONFIG_DIR / "user_config.json"
    
    logger.info(f"Loading risk tolerance from: {config_file_path}")
    
    if not Path(config_file_path).exists():
        raise FileNotFoundError(f"User config file not found: {config_file_path}")
    
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        # Extract risk tolerance from nested structure
        risk_tolerance = config.get('league_settings', {}).get('risk_tolerance', 'medium')
        
        # Validate risk tolerance setting
        valid_settings = ['low', 'medium', 'high']
        if risk_tolerance not in valid_settings:
            logger.warning(f"Invalid risk tolerance '{risk_tolerance}', defaulting to 'medium'")
            risk_tolerance = 'medium'
        
        logger.info(f"✅ Risk tolerance: {risk_tolerance}")
        return risk_tolerance
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise ValueError(f"Invalid config file format: {e}")
    except Exception as e:
        logger.error(f"Failed to load risk tolerance: {e}")
        raise ValueError(f"Error loading risk tolerance: {e}")


def create_risk_categorization(hybrid_data: pd.DataFrame, risk_tolerance: str) -> pd.DataFrame:
    """
    Create risk categorization based on uncertainty scores and risk tolerance.
    
    Args:
        hybrid_data: DataFrame with hybrid data including uncertainty scores
        risk_tolerance: User's risk tolerance ('low', 'medium', 'high')
        
    Returns:
        DataFrame with added risk categorization columns
    """
    logger.info(f"Creating risk categorization with tolerance: {risk_tolerance}")
    
    # Create a copy to avoid modifying original
    data = hybrid_data.copy()
    
    # Define risk thresholds based on tolerance
    if risk_tolerance == 'low':
        # Conservative thresholds - more players categorized as high risk
        low_threshold = 0.2
        high_threshold = 0.4
    elif risk_tolerance == 'high':
        # Aggressive thresholds - more players categorized as low risk
        low_threshold = 0.4
        high_threshold = 0.6
    else:  # medium
        # Balanced thresholds
        low_threshold = 0.3
        high_threshold = 0.5
    
    # Create risk categories
    def categorize_risk(uncertainty):
        if uncertainty <= low_threshold:
            return 'Low Risk'
        elif uncertainty <= high_threshold:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    # Apply categorization
    data['risk_category'] = data['uncertainty_score'].apply(categorize_risk)
    
    # Add color coding for Excel
    risk_colors = {
        'Low Risk': 'green',
        'Medium Risk': 'yellow', 
        'High Risk': 'red'
    }
    data['risk_color'] = data['risk_category'].map(risk_colors)
    
    # Log distribution
    risk_distribution = data['risk_category'].value_counts()
    logger.info(f"✅ Risk distribution: {dict(risk_distribution)}")
    
    return data


def calculate_composite_rankings(hybrid_data: pd.DataFrame, risk_tolerance: str = 'medium') -> pd.DataFrame:
    """
    Calculate composite rankings that balance VOR value with uncertainty.
    
    Args:
        hybrid_data: DataFrame with hybrid data
        risk_tolerance: User's risk tolerance preference
        
    Returns:
        DataFrame with composite rankings and risk-adjusted values
    """
    logger.info(f"Calculating composite rankings with risk tolerance: {risk_tolerance}")
    
    # Create a copy to avoid modifying original
    data = hybrid_data.copy()
    
    # Define risk adjustment factors based on tolerance
    if risk_tolerance == 'low':
        # Conservative - heavily penalize uncertainty
        uncertainty_weight = 1.5
        vor_weight = 1.0
    elif risk_tolerance == 'high':
        # Aggressive - less penalty for uncertainty
        uncertainty_weight = 0.5
        vor_weight = 1.0
    else:  # medium
        # Balanced approach
        uncertainty_weight = 1.0
        vor_weight = 1.0
    
    # Calculate risk-adjusted value
    # Formula: VOR * (1 - uncertainty_score * uncertainty_weight)
    data['risk_adjusted_value'] = (
        data['VOR'] * (1 - data['uncertainty_score'] * uncertainty_weight)
    )
    
    # Calculate composite score (enhanced version)
    # Formula: VOR * (1 - uncertainty_score) + mean_projection_bonus
    # For players with missing Bayesian data (uncertainty_score = 1.0), use a more reasonable approach
    mean_projection_bonus = (data['mean_projection'] - data['FPTS']) * 0.1
    
    # Handle players with missing Bayesian data
    # Create composite score with better handling of missing Bayesian data
    data['composite_score'] = data['VOR'] * (1 - data['uncertainty_score']) + mean_projection_bonus
    
    # For players with missing Bayesian data (uncertainty_score = 1.0), use 80% of VOR
    missing_bayesian_mask = data['uncertainty_score'] >= 1.0
    data.loc[missing_bayesian_mask, 'composite_score'] = data.loc[missing_bayesian_mask, 'VOR'] * 0.8
    
    # Sort by composite score for rankings
    data = data.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    # Add composite rank
    data['composite_rank'] = range(1, len(data) + 1)
    
    # Add risk-adjusted rank using rank() function
    data['risk_adjusted_rank'] = data['risk_adjusted_value'].rank(ascending=False, method='min').astype(int)
    
    logger.info(f"✅ Composite rankings calculated for {len(data)} players")
    logger.info("📈 Top 3 by composite score:")
    for i, row in data.head(3).iterrows():
        logger.info(f"   {i+1}. {row['PLAYER']} - VOR: {row['VOR']:.1f}, Uncertainty: {row['uncertainty_score']:.2f}, Composite: {row['composite_score']:.1f}")
    
    return data


def generate_position_risk_analysis(hybrid_data: pd.DataFrame) -> Dict:
    """
    Generate position-based risk analysis and uncertainty breakdown.
    
    Args:
        hybrid_data: DataFrame with hybrid data
        
    Returns:
        Dictionary with position-based risk analysis
    """
    logger.info("Generating position-based risk analysis...")
    
    position_analysis = {}
    
    for position in hybrid_data['POS'].unique():
        pos_data = hybrid_data[hybrid_data['POS'] == position]
        
        analysis = {
            'player_count': len(pos_data),
            'avg_uncertainty': pos_data['uncertainty_score'].mean(),
            'std_uncertainty': pos_data['uncertainty_score'].std(),
            'avg_vor': pos_data['VOR'].mean(),
            'avg_composite_score': pos_data['composite_score'].mean(),
            'risk_distribution': pos_data['risk_category'].value_counts().to_dict(),
            'top_players': pos_data.head(5)[['PLAYER', 'VOR', 'uncertainty_score', 'composite_score']].to_dict('records')
        }
        
        position_analysis[position] = analysis
    
    logger.info(f"✅ Position analysis generated for {len(position_analysis)} positions")
    
    # Log summary
    for pos, analysis in position_analysis.items():
        logger.info(f"   {pos}: {analysis['player_count']} players, avg uncertainty: {analysis['avg_uncertainty']:.3f}")
    
    return position_analysis


def validate_risk_adjusted_data(risk_adjusted_data: pd.DataFrame) -> None:
    """
    Validate risk-adjusted dataset for required columns and data quality.
    
    Args:
        risk_adjusted_data: DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating risk-adjusted data...")
    
    if risk_adjusted_data.empty:
        raise ValueError("Risk-adjusted dataset is empty")
    
    # Check required columns
    required_columns = [
        'PLAYER', 'POS', 'VOR', 'uncertainty_score', 'composite_score',
        'composite_rank', 'risk_adjusted_value', 'risk_category', 'risk_color'
    ]
    
    missing_columns = [col for col in required_columns if col not in risk_adjusted_data.columns]
    if missing_columns:
        raise ValueError(f"Risk-adjusted dataset missing required columns: {missing_columns}")
    
    # Check data types
    if risk_adjusted_data['VOR'].dtype not in ['float64', 'int64']:
        raise ValueError("VOR column must be numeric")
    
    if risk_adjusted_data['uncertainty_score'].dtype not in ['float64', 'int64']:
        raise ValueError("uncertainty_score column must be numeric")
    
    if risk_adjusted_data['composite_score'].dtype not in ['float64', 'int64']:
        raise ValueError("composite_score column must be numeric")
    
    # Check for valid risk categories
    valid_categories = ['Low Risk', 'Medium Risk', 'High Risk']
    invalid_categories = risk_adjusted_data[
        ~risk_adjusted_data['risk_category'].isin(valid_categories)
    ]
    
    if not invalid_categories.empty:
        raise ValueError(f"Found invalid risk categories: {invalid_categories['risk_category'].unique()}")
    
    # Check for reasonable uncertainty scores (0-1 range)
    invalid_uncertainty = risk_adjusted_data[
        (risk_adjusted_data['uncertainty_score'] < 0) | 
        (risk_adjusted_data['uncertainty_score'] > 1)
    ]
    
    if not invalid_uncertainty.empty:
        logger.warning(f"Found {len(invalid_uncertainty)} players with invalid uncertainty scores")
    
    # Check that composite ranks are sequential
    expected_ranks = list(range(1, len(risk_adjusted_data) + 1))
    actual_ranks = risk_adjusted_data['composite_rank'].tolist()
    
    if actual_ranks != expected_ranks:
        raise ValueError("Composite ranks are not sequential")
    
    logger.info("✅ Risk-adjusted data validation passed")


def create_risk_adjusted_rankings(hybrid_data: pd.DataFrame, config_file_path: str = None) -> pd.DataFrame:
    """
    Create complete risk-adjusted rankings from hybrid data.
    
    Args:
        hybrid_data: DataFrame with hybrid data
        config_file_path: Path to user config file
        
    Returns:
        DataFrame with complete risk-adjusted rankings
    """
    logger.info("Creating risk-adjusted rankings...")
    
    try:
        # Load risk tolerance
        risk_tolerance = load_risk_tolerance_config(config_file_path)
        
        # Create risk categorization
        data_with_risk = create_risk_categorization(hybrid_data, risk_tolerance)
        
        # Calculate composite rankings
        risk_adjusted_data = calculate_composite_rankings(data_with_risk, risk_tolerance)
        
        # Generate position analysis
        position_analysis = generate_position_risk_analysis(risk_adjusted_data)
        
        # Validate final dataset
        validate_risk_adjusted_data(risk_adjusted_data)
        
        logger.info("🎉 Risk-adjusted rankings created successfully!")
        return risk_adjusted_data
        
    except Exception as e:
        logger.error(f"❌ Failed to create risk-adjusted rankings: {e}")
        raise


def main():
    """Main function for testing risk-adjusted rankings."""
    try:
        # Load hybrid data
        from ffbayes.data_pipeline.hybrid_data_integration import (
            create_hybrid_dataset,
            load_hybrid_data_sources,
        )
        
        vor_data, bayesian_data = load_hybrid_data_sources()
        hybrid_data = create_hybrid_dataset(vor_data, bayesian_data)
        
        # Create risk-adjusted rankings
        risk_adjusted_data = create_risk_adjusted_rankings(hybrid_data)
        
        print("🎉 Risk-adjusted rankings successful!")
        print(f"📊 Risk-adjusted dataset: {len(risk_adjusted_data)} players")
        print("📈 Top 5 players by composite score:")
        for i, row in risk_adjusted_data.head().iterrows():
            print(f"   {i+1}. {row['PLAYER']} ({row['POS']}) - VOR: {row['VOR']:.1f}, Uncertainty: {row['uncertainty_score']:.2f}, Composite: {row['composite_score']:.1f}, Risk: {row['risk_category']}")
        
        return risk_adjusted_data
        
    except Exception as e:
        logger.error(f"❌ Risk-adjusted rankings failed: {e}")
        raise


if __name__ == "__main__":
    main()
