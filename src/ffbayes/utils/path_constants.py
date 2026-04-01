#!/usr/bin/env python3
"""
Path Constants - Centralized path management for FFBayes
Eliminates hardcoded paths throughout the package.
"""

import os
from pathlib import Path


# Base directories
def get_project_root() -> Path:
    """Return the canonical project root for the ffbayes repository."""
    env_root = os.getenv('FFBAYES_PROJECT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def get_runtime_root() -> Path:
    """Return the canonical runtime root for the active project.

    `~/ProjectsRuntime/ffbayes` is the canonical local runtime tree for this
    repository. Use `FFBAYES_RUNTIME_ROOT` to opt into a different location
    explicitly; we do not silently redirect writes to a repo-local fallback.
    """
    env_root = os.getenv('FFBAYES_RUNTIME_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.home() / 'ProjectsRuntime' / 'ffbayes').expanduser().resolve()


def get_cloud_root() -> Path:
    """Return the canonical cloud root for backed-up project artifacts.

    `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes` is the
    canonical cloud mirror location for this repository. Use
    `FFBAYES_CLOUD_ROOT` to opt into a different location explicitly.
    """
    env_root = os.getenv('FFBAYES_CLOUD_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (
        Path.home()
        / 'Library'
        / 'CloudStorage'
        / 'OneDrive-Personal'
        / 'SideProjects'
        / 'ffbayes'
    ).expanduser().resolve()


BASE_DIR = get_project_root()
RUNTIME_DIR = get_runtime_root()
PROJECTS_ROOT_DIR = Path.home() / 'Projects'
RUNTIME_DATA_DIR = RUNTIME_DIR / 'data'
RAW_DATA_DIR = RUNTIME_DATA_DIR / 'raw'
PROCESSED_DATA_DIR = RUNTIME_DATA_DIR / 'processed'
RAW_SEASON_DATASETS_DIR = RAW_DATA_DIR / 'season_datasets'
RAW_COMBINED_DATASETS_DIR = RAW_DATA_DIR / 'combined_datasets'
RUNTIME_RESULTS_DIR = RUNTIME_DIR / 'results'
RUNTIME_PLOTS_DIR = RUNTIME_DIR / 'plots'
RUNTIME_DASHBOARD_DIR = RUNTIME_DIR / 'dashboard'
REPO_DASHBOARD_DIR = BASE_DIR / 'dashboard'
SIDE_PROJECTS_ROOT_DIR = get_cloud_root()
CLOUD_RAW_DATA_DIR = get_cloud_root() / 'data' / 'raw'
CLOUD_PROCESSED_DATA_DIR = get_cloud_root() / 'data' / 'processed'
CLOUD_RESULTS_DIR = get_cloud_root() / 'results'
CLOUD_PLOTS_DIR = get_cloud_root() / 'plots'
CLOUD_DASHBOARD_DIR = get_cloud_root() / 'dashboard'
CLOUD_DOCS_IMAGES_DIR = get_cloud_root() / 'docs' / 'images'
PAGES_SITE_DIR = BASE_DIR / 'site'
SRC_DIR = BASE_DIR / 'src'
CONFIG_DIR = BASE_DIR / 'config'
LOGS_DIR = RUNTIME_DIR / 'logs'
PLOTS_DIR = RUNTIME_PLOTS_DIR

# Data directories
DATASETS_DIR = RUNTIME_DATA_DIR
SEASON_DATASETS_DIR = RAW_SEASON_DATASETS_DIR
COMBINED_DATASETS_DIR = PROCESSED_DATA_DIR / 'combined_datasets'
SNAKE_DRAFT_DATASETS_DIR = PROCESSED_DATA_DIR / 'snake_draft_datasets'
UNIFIED_DATASET_DIR = PROCESSED_DATA_DIR / 'unified_dataset'


# Results directories (organized by year and type)
def get_phase_name(phase: str | None = None) -> str:
    """Return the supported pipeline phase.

    FFBayes now supports a single phase: `pre_draft`.
    """

    resolved = (phase or os.getenv('FFBAYES_PIPELINE_PHASE') or 'pre_draft').lower()
    if resolved != 'pre_draft':
        raise ValueError(
            f'Unsupported pipeline phase: {resolved!r}. Only "pre_draft" is supported.'
        )
    return 'pre_draft'


def get_run_root(year: int = None) -> Path:
    """Get the root directory for a specific (pre-draft) run."""
    if year is None:
        env_year = os.getenv('FFBAYES_PIPELINE_YEAR')
        if env_year:
            try:
                year = int(env_year)
            except ValueError:
                year = None

    if year is None:
        from datetime import datetime

        year = datetime.now().year

    return RUNTIME_DIR / 'runs' / str(year) / 'pre_draft'


def get_pre_draft_run_root(year: int = None) -> Path:
    """Get the canonical pre-draft run root."""
    return get_run_root(year)


def get_pre_draft_artifacts_dir(year: int = None) -> Path:
    """Get the canonical pre-draft artifacts directory."""
    path = get_pre_draft_run_root(year) / 'artifacts'
    ensure_dir_exists(path)
    return path


def get_pre_draft_diagnostics_dir(year: int = None) -> Path:
    """Get the canonical pre-draft diagnostics directory."""
    path = get_pre_draft_run_root(year) / 'diagnostics'
    ensure_dir_exists(path)
    return path


def get_results_dir(year: int = None) -> Path:
    """Get the canonical results root for a given year (pre-draft artifacts)."""
    path = get_pre_draft_artifacts_dir(year)
    ensure_dir_exists(path)
    return path


def get_cloud_dashboard_dir(year: int = None) -> Path:
    """Get published dashboard directory for a specific year."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    path = CLOUD_DASHBOARD_DIR / str(year)
    ensure_dir_exists(path)
    return path


def get_cloud_pre_draft_dashboard_dir(year: int = None) -> Path:
    """Get published pre-draft dashboard directory."""
    path = get_cloud_dashboard_dir(year) / 'pre_draft'
    ensure_dir_exists(path)
    return path


def get_pre_draft_dir(year: int = None) -> Path:
    """Get the canonical pre-draft output directory."""
    return get_pre_draft_artifacts_dir(year)


def get_pre_draft_analysis_dir(year: int = None) -> Path:
    """Get pre-draft analysis directory (for features CSVs, etc.)."""
    path = get_pre_draft_dir(year) / 'analysis'
    ensure_dir_exists(path)
    return path


def get_cloud_results_dir(year: int = None) -> Path:
    """Get published results directory for a specific year."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    path = CLOUD_RESULTS_DIR / str(year)
    ensure_dir_exists(path)
    return path


def get_cloud_pre_draft_dir(year: int = None) -> Path:
    """Get published pre-draft results directory."""
    path = get_cloud_results_dir(year) / 'pre_draft'
    ensure_dir_exists(path)
    return path


# Specific result subdirectories
def get_vor_strategy_dir(year: int = None) -> Path:
    """Get VOR strategy directory."""
    path = get_pre_draft_artifacts_dir(year) / 'vor_strategy'
    ensure_dir_exists(path)
    return path


def get_hybrid_mc_dir(year: int = None) -> Path:
    """Get Hybrid MC results directory."""
    path = get_pre_draft_artifacts_dir(year) / 'hybrid_mc_bayesian'
    ensure_dir_exists(path)
    return path


def get_draft_strategy_dir(year: int = None) -> Path:
    """Get draft strategy directory."""
    path = get_pre_draft_artifacts_dir(year) / 'draft_strategy'
    ensure_dir_exists(path)
    return path


def get_draft_model_outputs_dir(year: int = None) -> Path:
    """Get draft model output directory."""
    path = get_draft_strategy_dir(year) / 'model_outputs'
    ensure_dir_exists(path)
    return path


def get_draft_board_path(year: int = None) -> Path:
    """Get workbook path for the draft board artifact."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    return get_draft_strategy_dir(year) / f'draft_board_{year}.xlsx'


def get_dashboard_payload_path(year: int = None) -> Path:
    """Get JSON payload path for the interactive draft dashboard."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    return get_draft_strategy_dir(year) / f'dashboard_payload_{year}.json'


def get_dashboard_html_path(year: int = None) -> Path:
    """Get HTML fallback path for the interactive draft dashboard."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    return get_draft_strategy_dir(year) / f'draft_board_{year}.html'


def get_draft_decision_backtest_path(year: int = None, year_range: str = None) -> Path:
    """Get JSON path for the draft decision backtest artifact."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    suffix = year_range or str(year)
    return get_draft_strategy_dir(year) / f'draft_decision_backtest_{suffix}.json'


def get_draft_strategy_comparison_dir(year: int = None) -> Path:
    """Get draft strategy comparison directory."""
    path = get_pre_draft_diagnostics_dir(year) / 'draft_strategy_comparison'
    ensure_dir_exists(path)
    return path


def get_validation_dir(year: int = None) -> Path:
    """Get directory for validation logs and diagnostics."""
    path = get_pre_draft_diagnostics_dir(year) / 'validation'
    ensure_dir_exists(path)
    return path


def get_team_aggregation_dir(year: int = None) -> Path:
    """Get team aggregation directory (pre-draft artifacts tree)."""
    path = get_results_dir(year) / 'team_aggregation'
    ensure_dir_exists(path)
    return path


def get_monte_carlo_dir(year: int = None) -> Path:
    """Get Monte Carlo results directory (pre-draft artifacts tree)."""
    path = get_results_dir(year) / 'montecarlo_results'
    ensure_dir_exists(path)
    return path


# Plot directories
def get_plots_dir(year: int = None) -> Path:
    """Get plots/diagnostics directory for a specific year."""
    path = get_pre_draft_diagnostics_dir(year)
    ensure_dir_exists(path)
    return path


def get_pre_draft_plots_dir(year: int = None) -> Path:
    """Get pre-draft plots directory."""
    return get_pre_draft_diagnostics_dir(year)


def get_cloud_plots_dir(year: int = None) -> Path:
    """Get published plots directory for a specific year."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    path = CLOUD_PLOTS_DIR / str(year)
    ensure_dir_exists(path)
    return path


def get_cloud_pre_draft_plots_dir(year: int = None) -> Path:
    """Get published pre-draft plots directory."""
    path = get_cloud_plots_dir(year) / 'pre_draft'
    ensure_dir_exists(path)
    return path


# Team files
def get_teams_dir() -> Path:
    """Get teams directory."""
    path = RAW_DATA_DIR / 'my_ff_teams'
    ensure_dir_exists(path)
    return path


def get_logs_dir() -> Path:
    """Get logs directory."""
    path = LOGS_DIR
    ensure_dir_exists(path)
    return path


def get_misc_datasets_dir() -> Path:
    """Get directory for user-managed raw helper datasets."""
    path = RAW_DATA_DIR / 'misc-datasets'
    ensure_dir_exists(path)
    return path


def get_raw_season_datasets_dir() -> Path:
    """Get runtime raw season dataset directory."""
    ensure_dir_exists(RAW_SEASON_DATASETS_DIR)
    return RAW_SEASON_DATASETS_DIR


def get_raw_combined_datasets_dir() -> Path:
    """Get runtime raw combined dataset directory."""
    ensure_dir_exists(RAW_COMBINED_DATASETS_DIR)
    return RAW_COMBINED_DATASETS_DIR


def get_cloud_docs_images_dir() -> Path:
    """Get cloud-published documentation image directory."""
    ensure_dir_exists(CLOUD_DOCS_IMAGES_DIR)
    return CLOUD_DOCS_IMAGES_DIR


def get_default_team_file() -> Path:
    """Get default team file path."""
    from datetime import datetime

    current_year = datetime.now().year
    return get_teams_dir() / f'drafted_team_{current_year}.tsv'


# Configuration files
def get_user_config_file() -> Path:
    """Get user configuration file path."""
    return CONFIG_DIR / 'user_config.json'


def get_pipeline_config_file() -> Path:
    """Get pipeline configuration file path."""
    return CONFIG_DIR / 'pipeline_config.json'


def get_pre_draft_config_file() -> Path:
    """Get pre-draft pipeline configuration file path."""
    return CONFIG_DIR / 'pipeline_pre_draft.json'


# File patterns
def get_season_data_pattern() -> str:
    """Get pattern for season data files."""
    return str(SEASON_DATASETS_DIR / '*season.csv')


def get_combined_data_pattern() -> str:
    """Get pattern for combined data files."""
    return str(COMBINED_DATASETS_DIR / '*_modern.csv')


def get_vor_data_pattern() -> str:
    """Get pattern for VOR data files."""
    return str(SNAKE_DRAFT_DATASETS_DIR / 'snake-draft_ppr-*.csv')


def get_vor_strategy_pattern() -> str:
    """Get pattern for VOR strategy files."""
    return str(SNAKE_DRAFT_DATASETS_DIR / 'DRAFTING STRATEGY -- snake-draft_ppr-*.xlsx')


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


def get_pages_site_dir() -> Path:
    """Get the repository-local GitHub Pages site directory."""
    path = BASE_DIR / 'site'
    ensure_dir_exists(path)
    return path


# Common file paths
def get_unified_dataset_path() -> Path:
    """Get unified dataset path."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / 'unified_dataset.json'


def get_unified_dataset_excel_path() -> Path:
    """Get unified dataset Excel path (human-readable)."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / 'unified_dataset.xlsx'


def get_unified_dataset_csv_path() -> Path:
    """Get unified dataset CSV path (compatibility)."""
    # Ensure the directory exists before returning the path
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    return UNIFIED_DATASET_DIR / 'unified_dataset.csv'


def get_latest_combined_dataset() -> Path:
    """Get path to latest combined dataset."""
    from ffbayes.utils.file_naming import get_latest_file_by_pattern

    latest_file = get_latest_file_by_pattern(
        COMBINED_DATASETS_DIR, '*season_modern', '.csv'
    )
    if latest_file:
        return latest_file
    raise FileNotFoundError(f'No combined datasets found in {COMBINED_DATASETS_DIR}')


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

    print('📁 Creating all required directories...')

    # Core directories
    ensure_dir_exists(LOGS_DIR)
    ensure_dir_exists(RUNTIME_DATA_DIR)
    ensure_dir_exists(RAW_DATA_DIR)
    ensure_dir_exists(PROCESSED_DATA_DIR)
    ensure_dir_exists(DATASETS_DIR)
    ensure_dir_exists(RAW_SEASON_DATASETS_DIR)
    ensure_dir_exists(RAW_COMBINED_DATASETS_DIR)
    ensure_dir_exists(SEASON_DATASETS_DIR)
    ensure_dir_exists(COMBINED_DATASETS_DIR)
    ensure_dir_exists(SNAKE_DRAFT_DATASETS_DIR)
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    ensure_dir_exists(RUNTIME_DASHBOARD_DIR)
    ensure_dir_exists(REPO_DASHBOARD_DIR)

    # Year-based pre-draft directories
    ensure_dir_exists(get_pre_draft_run_root(year))
    ensure_dir_exists(get_pre_draft_artifacts_dir(year))
    ensure_dir_exists(get_pre_draft_diagnostics_dir(year))
    ensure_dir_exists(get_vor_strategy_dir(year))
    ensure_dir_exists(get_hybrid_mc_dir(year))
    ensure_dir_exists(get_draft_strategy_dir(year))
    ensure_dir_exists(get_draft_model_outputs_dir(year))
    ensure_dir_exists(get_draft_strategy_comparison_dir(year))
    ensure_dir_exists(get_validation_dir(year))

    # Team directory
    ensure_dir_exists(get_teams_dir())
    ensure_dir_exists(get_misc_datasets_dir())

    print(f'✅ All required directories created for year {year}')
