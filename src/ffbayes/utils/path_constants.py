#!/usr/bin/env python3
"""
Path Constants - Centralized path management for FFBayes
Eliminates hardcoded paths throughout the package.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path.cwd()
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"

# Data directories
DATASETS_DIR = BASE_DIR / "datasets"
SEASON_DATASETS_DIR = DATASETS_DIR / "season_datasets"
COMBINED_DATASETS_DIR = DATASETS_DIR / "combined_datasets"
SNAKE_DRAFT_DATASETS_DIR = DATASETS_DIR / "snake_draft_datasets"
UNIFIED_DATASET_DIR = DATASETS_DIR / "unified_dataset"

# Results directories (organized by year and type)
def get_results_dir(year: int = None) -> Path:
    """Get results directory for a specific year."""
    if year is None:
        from datetime import datetime
        year = datetime.now().year
    path = BASE_DIR / "results" / str(year)
    ensure_dir_exists(path)
    return path

def get_pre_draft_dir(year: int = None) -> Path:
    """Get pre-draft results directory."""
    path = get_results_dir(year) / "pre_draft"
    ensure_dir_exists(path)
    return path

def get_post_draft_dir(year: int = None) -> Path:
    """Get post-draft results directory."""
    path = get_results_dir(year) / "post_draft"
    ensure_dir_exists(path)
    return path



# Specific result subdirectories
def get_vor_strategy_dir(year: int = None) -> Path:
    """Get VOR strategy directory."""
    path = get_pre_draft_dir(year) / "vor_strategy"
    ensure_dir_exists(path)
    return path

def get_hybrid_mc_dir(year: int = None) -> Path:
    """Get Hybrid MC results directory."""
    path = get_pre_draft_dir(year) / "hybrid_mc_bayesian"
    ensure_dir_exists(path)
    return path

def get_draft_strategy_dir(year: int = None) -> Path:
    """Get draft strategy directory."""
    path = get_pre_draft_dir(year) / "draft_strategy"
    ensure_dir_exists(path)
    return path

def get_team_aggregation_dir(year: int = None) -> Path:
    """Get team aggregation directory."""
    path = get_post_draft_dir(year) / "team_aggregation"
    ensure_dir_exists(path)
    return path



def get_monte_carlo_dir(year: int = None) -> Path:
    """Get Monte Carlo results directory."""
    path = get_post_draft_dir(year) / "montecarlo_results"
    ensure_dir_exists(path)
    return path

# Plot directories
def get_plots_dir(year: int = None) -> Path:
    """Get plots directory for a specific year."""
    if year is None:
        from datetime import datetime
        year = datetime.now().year
    path = PLOTS_DIR / str(year)
    ensure_dir_exists(path)
    return path

def get_pre_draft_plots_dir(year: int = None) -> Path:
    """Get pre-draft plots directory."""
    path = get_plots_dir(year) / "pre_draft"
    ensure_dir_exists(path)
    return path

def get_post_draft_plots_dir(year: int = None) -> Path:
    """Get post-draft plots directory."""
    path = get_plots_dir(year) / "post_draft"
    ensure_dir_exists(path)
    return path

# Team files
def get_teams_dir() -> Path:
    """Get teams directory."""
    path = BASE_DIR / "my_ff_teams"
    ensure_dir_exists(path)
    return path

def get_default_team_file() -> Path:
    """Get default team file path."""
    from datetime import datetime
    current_year = datetime.now().year
    return get_teams_dir() / f"drafted_team_{current_year}.tsv"

# Configuration files
def get_user_config_file() -> Path:
    """Get user configuration file path."""
    return CONFIG_DIR / "user_config.json"

def get_pipeline_config_file() -> Path:
    """Get pipeline configuration file path."""
    return CONFIG_DIR / "pipeline_config.json"

def get_pre_draft_config_file() -> Path:
    """Get pre-draft pipeline configuration file path."""
    return CONFIG_DIR / "pipeline_pre_draft.json"

def get_post_draft_config_file() -> Path:
    """Get post-draft pipeline configuration file path."""
    return CONFIG_DIR / "pipeline_post_draft.json"

# File patterns
def get_season_data_pattern() -> str:
    """Get pattern for season data files."""
    return str(SEASON_DATASETS_DIR / "*season.csv")

def get_combined_data_pattern() -> str:
    """Get pattern for combined data files."""
    return str(COMBINED_DATASETS_DIR / "*_modern.csv")

def get_vor_data_pattern() -> str:
    """Get pattern for VOR data files."""
    return str(SNAKE_DRAFT_DATASETS_DIR / "snake-draft_ppr-*.csv")

def get_vor_strategy_pattern() -> str:
    """Get pattern for VOR strategy files."""
    return str(SNAKE_DRAFT_DATASETS_DIR / "DRAFTING STRATEGY -- snake-draft_ppr-*.xlsx")

# Utility functions
def ensure_dir_exists(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_relative_path(path: Path) -> str:
    """Get relative path from base directory."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)

# Common file paths
def get_unified_dataset_path() -> Path:
    """Get unified dataset path."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / "unified_dataset.json"

def get_unified_dataset_excel_path() -> Path:
    """Get unified dataset Excel path (human-readable)."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / "unified_dataset.xlsx"

def get_unified_dataset_csv_path() -> Path:
    """Get unified dataset CSV path (compatibility)."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / "unified_dataset.csv"

def get_latest_combined_dataset() -> Path:
    """Get path to latest combined dataset."""
    from ffbayes.utils.file_naming import get_latest_file_by_pattern
    latest_file = get_latest_file_by_pattern(
        COMBINED_DATASETS_DIR, 
        "*season_modern", 
        ".csv"
    )
    if latest_file:
        return latest_file
    else:
        # Fallback to default pattern
        return COMBINED_DATASETS_DIR / "2020-2024season_modern.csv"

def get_bayesian_model_dir(year: int = None) -> Path:
    """Get directory for Bayesian/Hybrid model outputs (pre-draft)."""
    return get_hybrid_mc_dir(year)

def create_all_required_directories(year: int = None) -> None:
    """
    Create all required directories for the pipeline.
    This is the single point of directory creation.
    
    Args:
        year: Year for year-based directories. If None, uses current year.
    """
    if year is None:
        from datetime import datetime
        year = datetime.now().year
    
    print("📁 Creating all required directories...")
    
    # Core directories
    ensure_dir_exists(LOGS_DIR)
    ensure_dir_exists(DATASETS_DIR)
    ensure_dir_exists(SEASON_DATASETS_DIR)
    ensure_dir_exists(COMBINED_DATASETS_DIR)
    ensure_dir_exists(SNAKE_DRAFT_DATASETS_DIR)
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    
    # Year-based results directories
    ensure_dir_exists(get_results_dir(year))
    ensure_dir_exists(get_pre_draft_dir(year))
    ensure_dir_exists(get_post_draft_dir(year))
    ensure_dir_exists(get_vor_strategy_dir(year))
    ensure_dir_exists(get_hybrid_mc_dir(year))
    ensure_dir_exists(get_draft_strategy_dir(year))
    ensure_dir_exists(get_team_aggregation_dir(year))
    ensure_dir_exists(get_monte_carlo_dir(year))
    
    # Year-based plots directories
    ensure_dir_exists(get_plots_dir(year))
    ensure_dir_exists(get_pre_draft_plots_dir(year))
    ensure_dir_exists(get_post_draft_plots_dir(year))
    
    # Team directory
    ensure_dir_exists(get_teams_dir())
    
    print(f"✅ All required directories created for year {year}")
