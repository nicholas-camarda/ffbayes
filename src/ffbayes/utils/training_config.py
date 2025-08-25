"""
Centralized training configuration for FFBayes.

This module provides consistent training year configurations across the entire codebase,
eliminating hardcoded training years and ensuring scalability.
"""

import os
from datetime import datetime
from typing import List

# Default training years - can be overridden via environment variables
# Use 8 years of data for comprehensive analysis
DEFAULT_TRAINING_YEARS = list(range(datetime.now().year - 5, datetime.now().year))

# Current year for projections
CURRENT_YEAR = datetime.now().year


def get_default_training_years() -> List[int]:
    """
    Get the default training years for the current projection year.
    
    Returns:
        List of years to use for training (e.g., [2023, 2024])
    """
    return DEFAULT_TRAINING_YEARS.copy()


def get_training_years_for_year(projection_year: int) -> List[int]:
    """
    Get appropriate training years for a given projection year.
    
    Args:
        projection_year: The year being projected (e.g., 2025)
    
    Returns:
        List of years to use for training
    
    Examples:
        >>> get_training_years_for_year(2025)
        [2020, 2021, 2022, 2023, 2024]
        
        >>> get_training_years_for_year(2026)
        [2021, 2022, 2023, 2024, 2025]
    """
    # Use 8 years of data for comprehensive analysis
    return list(range(projection_year - 8, projection_year))


def get_current_training_years() -> List[int]:
    """
    Get training years for the current year.
    
    Returns:
        Training years for current year projections
    """
    return get_training_years_for_year(CURRENT_YEAR)


def get_recent_seasons_for_analysis() -> List[int]:
    """
    Get recent seasons for data analysis (last 5 years).
    
    Returns:
        List of recent seasons for comprehensive analysis
    """
    return list(range(CURRENT_YEAR - 4, CURRENT_YEAR + 1))


def get_test_train_split_years() -> List[int]:
    """
    Get years for train/test split in modeling.
    
    Returns:
        [train_year, test_year] for model validation
    """
    training_years = get_current_training_years()
    return [training_years[0], training_years[1]]  # [2023, 2024]


def get_monte_carlo_training_years() -> List[int]:
    """
    Get training years specifically for Monte Carlo simulations.
    
    Returns:
        Training years for Monte Carlo (same as default)
    """
    return get_default_training_years()


def get_bayesian_training_years() -> List[int]:
    """
    Get training years specifically for Bayesian modeling.
    
    Returns:
        Training years for Bayesian models (same as default)
    """
    return get_default_training_years()


def get_hybrid_training_years() -> List[int]:
    """
    Get training years specifically for hybrid models.
    
    Returns:
        Training years for hybrid models (same as default)
    """
    return get_default_training_years()


# Configuration validation
def validate_training_years(training_years: List[int]) -> bool:
    """
    Validate that training years are reasonable.
    
    Args:
        training_years: List of years to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not training_years:
        return False
    
    if len(training_years) < 1:
        return False
    
    # Check that years are reasonable (not too old, not in future)
    current_year = datetime.now().year
    for year in training_years:
        if year < 2010 or year > current_year:
            return False
    
    return True


# Environment variable support (for future extensibility)
def get_training_years_from_env() -> List[int]:
    """
    Get training years from environment variables if available.
    
    Returns:
        Training years from environment or default
    """
    
    env_years = os.getenv('FFBAYES_TRAINING_YEARS')
    if env_years:
        try:
            # Parse comma-separated years
            years = [int(y.strip()) for y in env_years.split(',')]
            if validate_training_years(years):
                return years
        except (ValueError, AttributeError):
            pass
    
    return get_default_training_years()


# VOR Scraping Configuration - Single source of truth
VOR_CONFIG = {
    'base_url_adp': 'https://www.fantasypros.com/nfl/adp/ppr-overall.php',
    'base_url_projections': 'https://www.fantasypros.com/nfl/projections/{position}.php?week=draft',
    'positions': ['rb', 'qb', 'te', 'wr']
}

# Note: output_dir is now managed by path_constants
# Use ffbayes.utils.path_constants.SNAKE_DRAFT_DATASETS_DIR instead
