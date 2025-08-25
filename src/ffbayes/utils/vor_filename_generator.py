#!/usr/bin/env python3
"""
VOR Filename Generator - Generate VOR filenames dynamically based on configuration
Instead of hardcoding 'ppr-0.5' and 'top-120' everywhere.
"""

import os
from datetime import datetime


def get_vor_config():
    """Get VOR configuration from centralized config loader."""
    try:
        from ffbayes.utils.config_loader import get_config
        config_loader = get_config()
        return {
            'ppr': config_loader.get_vor_setting('ppr'),
            'top_rank': config_loader.get_vor_setting('top_rank')
        }
    except ImportError:
        # Fallback to environment variables if config loader not available
        return {
            'ppr': float(os.getenv('VOR_PPR', '0.5')),
            'top_rank': int(os.getenv('VOR_TOP_RANK', '120'))
        }

def generate_vor_filenames(current_year=None):
    """Generate VOR filenames based on current configuration."""
    if current_year is None:
        current_year = datetime.now().year
    
    config = get_vor_config()
    
    # Generate filenames
    csv_filename = f"snake-draft_ppr-{config['ppr']}_vor_top-{config['top_rank']}_{current_year}.csv"
    excel_filename = f"DRAFTING STRATEGY -- snake-draft_ppr-{config['ppr']}_vor_top-{config['top_rank']}_{current_year}.xlsx"
    
    return {
        'csv': csv_filename,
        'excel': excel_filename,
        'config': config
    }

def get_vor_csv_path(current_year=None):
    """Get the full path to the VOR CSV file."""
    if current_year is None:
        current_year = datetime.now().year
    
    filenames = generate_vor_filenames(current_year)
    from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
    return str(SNAKE_DRAFT_DATASETS_DIR / filenames['csv'])

def get_vor_excel_path(current_year=None):
    """Get the full path to the VOR Excel strategy file."""
    if current_year is None:
        current_year = datetime.now().year
    
    filenames = generate_vor_filenames(current_year)
    from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
    return str(SNAKE_DRAFT_DATASETS_DIR / filenames['excel'])

def get_vor_strategy_path(current_year=None):
    """Get the full path to the organized VOR strategy file."""
    if current_year is None:
        current_year = datetime.now().year
    
    filenames = generate_vor_filenames(current_year)
    from ffbayes.utils.path_constants import get_vor_strategy_dir
    return str(get_vor_strategy_dir(current_year) / filenames['excel'])

def get_vor_csv_filename(current_year=None):
    """Get just the VOR CSV filename."""
    if current_year is None:
        current_year = datetime.now().year
    
    filenames = generate_vor_filenames(current_year)
    return filenames['csv']

def get_vor_excel_filename(current_year=None):
    """Get just the VOR Excel filename."""
    if current_year is None:
        current_year = datetime.now().year
    
    filenames = generate_vor_filenames(current_year)
    return filenames['excel']

if __name__ == "__main__":
    """Test the filename generator."""
    print("🔧 VOR Filename Generator Test")
    print("=" * 40)
    
    # Test with current year
    filenames = generate_vor_filenames()
    print(f"📊 Configuration: PPR={filenames['config']['ppr']}, Top Rank={filenames['config']['top_rank']}")
    print(f"📁 CSV: {filenames['csv']}")
    print(f"📊 Excel: {filenames['excel']}")
    
    print("\n🔗 Full Paths:")
    print(f"   CSV: {get_vor_csv_path()}")
    print(f"   Excel: {get_vor_excel_path()}")
    print(f"   Strategy: {get_vor_strategy_path()}")
    
    # Test with custom year
    print("\n📅 Custom Year (2024):")
    filenames_2024 = generate_vor_filenames(2024)
    print(f"   CSV: {filenames_2024['csv']}")
    print(f"   Excel: {filenames_2024['excel']}")
