#!/usr/bin/env python3
"""
Strategy Path Generator - Generate strategy file paths dynamically
Instead of hardcoding draft position and other parameters.
"""

import os
from datetime import datetime


def get_strategy_config():
    """Get strategy configuration from centralized config loader."""
    try:
        from ffbayes.utils.config_loader import get_config
        config_loader = get_config()
        return {
            'draft_position': config_loader.get_league_setting('draft_position'),
            'league_size': config_loader.get_league_setting('league_size'),
            'risk_tolerance': config_loader.get_league_setting('risk_tolerance')
        }
    except ImportError:
        # Fallback to environment variables if config loader not available
        return {
            'draft_position': int(os.getenv('DRAFT_POSITION', '10')),
            'league_size': int(os.getenv('LEAGUE_SIZE', '10')),
            'risk_tolerance': os.getenv('RISK_TOLERANCE', 'medium')
        }

def get_bayesian_strategy_path(current_year=None, draft_position=None):
    """Get the path to the Bayesian strategy JSON file."""
    if current_year is None:
        current_year = datetime.now().year
    
    if draft_position is None:
        config = get_strategy_config()
        draft_position = config['draft_position']
    
    from ffbayes.utils.path_constants import get_draft_strategy_dir
    return str(get_draft_strategy_dir(current_year) / f"draft_strategy_pos{draft_position}_{current_year}.json")

def get_bayesian_strategy_filename(current_year=None, draft_position=None):
    """Get just the Bayesian strategy filename."""
    if current_year is None:
        current_year = datetime.now().year
    
    if draft_position is None:
        config = get_strategy_config()
        draft_position = config['draft_position']
    
    return f"draft_strategy_pos{draft_position}_{current_year}.json"

def get_hybrid_mc_results_path(current_year=None):
    """Get the path to the Hybrid MC results file."""
    if current_year is None:
        current_year = datetime.now().year
    
    from ffbayes.utils.path_constants import get_hybrid_mc_dir
    return str(get_hybrid_mc_dir(current_year) / "hybrid_model_results.json")

def get_hybrid_mc_results_filename():
    """Get just the Hybrid MC results filename."""
    return "hybrid_model_results.json"

if __name__ == "__main__":
    """Test the strategy path generator."""
    print("🔧 Strategy Path Generator Test")
    print("=" * 40)
    
    # Test with current year
    config = get_strategy_config()
    print(f"📊 Configuration: Position={config['draft_position']}, League={config['league_size']}, Risk={config['risk_tolerance']}")
    
    print("\n🔗 Strategy Paths:")
    print(f"   Bayesian: {get_bayesian_strategy_path()}")
    print(f"   Hybrid MC: {get_hybrid_mc_results_path()}")
    
    # Test with custom values
    print("\n📅 Custom Position (5):")
    print(f"   Bayesian: {get_bayesian_strategy_path(draft_position=5)}")
    
    print("\n📅 Custom Year (2024):")
    print(f"   Bayesian: {get_bayesian_strategy_path(2024)}")
    print(f"   Hybrid MC: {get_hybrid_mc_results_path(2024)}")
