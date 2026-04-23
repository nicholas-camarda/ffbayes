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
INPUTS_DIR = RUNTIME_DIR / 'inputs'
RAW_INPUTS_DIR = INPUTS_DIR / 'raw'
PROCESSED_INPUTS_DIR = INPUTS_DIR / 'processed'
RUNTIME_DATA_DIR = INPUTS_DIR
RAW_DATA_DIR = RAW_INPUTS_DIR
PROCESSED_DATA_DIR = PROCESSED_INPUTS_DIR
RAW_SEASON_DATASETS_DIR = RAW_INPUTS_DIR / 'season_datasets'
RAW_COMBINED_DATASETS_DIR = RAW_INPUTS_DIR / 'combined_datasets'
RUNTIME_RESULTS_DIR = RUNTIME_DIR / 'results'
RUNTIME_PLOTS_DIR = RUNTIME_DIR / 'plots'
RUNTIME_DASHBOARD_DIR = RUNTIME_DIR / 'dashboard'
REPO_DASHBOARD_DIR = BASE_DIR / 'dashboard'
SIDE_PROJECTS_ROOT_DIR = get_cloud_root()
CLOUD_RAW_DATA_DIR = get_cloud_root() / 'data' / 'raw'
CLOUD_PROCESSED_DATA_DIR = get_cloud_root() / 'data' / 'processed'
CLOUD_ANALYSIS_DIR = get_cloud_root() / 'Analysis'
PAGES_SITE_DIR = BASE_DIR / 'site'
SRC_DIR = BASE_DIR / 'src'
CONFIG_DIR = BASE_DIR / 'config'
LOGS_DIR = RUNTIME_DIR / 'logs'
PLOTS_DIR = RUNTIME_PLOTS_DIR

# Canonical local working input directories
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
    """Get the canonical root directory for a specific season."""
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

    return RUNTIME_DIR / 'seasons' / str(year)


def get_pre_draft_run_root(year: int = None) -> Path:
    """Return the canonical season root for the supported workflow."""
    return get_run_root(year)


def get_pre_draft_artifacts_dir(year: int = None) -> Path:
    """Get the canonical season output root for the supported workflow."""
    path = get_pre_draft_run_root(year)
    ensure_dir_exists(path)
    return path


def get_pre_draft_diagnostics_dir(year: int = None) -> Path:
    """Get the canonical diagnostics directory for a season."""
    path = get_pre_draft_run_root(year) / 'diagnostics'
    ensure_dir_exists(path)
    return path


def get_results_dir(year: int = None) -> Path:
    """Get the canonical season output root for a given year."""
    path = get_pre_draft_artifacts_dir(year)
    ensure_dir_exists(path)
    return path


def get_pre_draft_dir(year: int = None) -> Path:
    """Return the canonical season output root for the supported workflow."""
    return get_pre_draft_artifacts_dir(year)


def get_pre_draft_analysis_dir(year: int = None) -> Path:
    """Return the season-scoped analysis support directory."""
    path = get_pre_draft_dir(year) / 'analysis'
    ensure_dir_exists(path)
    return path


# Specific result subdirectories
def get_vor_strategy_dir(year: int = None) -> Path:
    """Get VOR strategy directory."""
    path = get_pre_draft_artifacts_dir(year) / 'vor_strategy'
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


def get_draft_retrospective_json_path(year: int = None) -> Path:
    """Get JSON path for the draft retrospective artifact."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    return get_draft_strategy_dir(year) / f'draft_retrospective_{year}.json'


def get_draft_retrospective_html_path(year: int = None) -> Path:
    """Get HTML path for the draft retrospective artifact."""
    if year is None:
        from datetime import datetime

        year = datetime.now().year
    return get_draft_strategy_dir(year) / f'draft_retrospective_{year}.html'


def get_finalized_drafts_dir(year: int = None) -> Path:
    """Get canonical runtime directory for imported finalized draft bundles."""
    path = get_draft_strategy_dir(year) / 'finalized_drafts'
    ensure_dir_exists(path)
    return path


def get_validation_dir(year: int = None) -> Path:
    """Get directory for validation logs and diagnostics."""
    path = get_pre_draft_diagnostics_dir(year) / 'validation'
    ensure_dir_exists(path)
    return path


def get_team_aggregation_dir(year: int = None) -> Path:
    """Return the season-scoped team aggregation directory."""
    path = get_results_dir(year) / 'team_aggregation'
    ensure_dir_exists(path)
    return path


def get_monte_carlo_dir(year: int = None) -> Path:
    """Return the season-scoped Monte Carlo results directory."""
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
    """Return the season-scoped diagnostics directory."""
    return get_pre_draft_diagnostics_dir(year)


def get_cloud_analysis_root() -> Path:
    """Get the cloud root for dated analysis snapshots."""
    ensure_dir_exists(CLOUD_ANALYSIS_DIR)
    return CLOUD_ANALYSIS_DIR


def get_cloud_analysis_snapshot_dir(snapshot_id: str | None = None) -> Path:
    """Get the dated cloud snapshot directory for a publish event."""
    if snapshot_id is None:
        from datetime import datetime

        snapshot_id = datetime.now().strftime('%Y-%m-%d')
    path = get_cloud_analysis_root() / snapshot_id
    ensure_dir_exists(path)
    return path


def get_cloud_data_dir() -> Path:
    """Get the cloud root for stable published data."""
    path = get_cloud_root() / 'data'
    ensure_dir_exists(path)
    return path


# Team files
def get_teams_dir() -> Path:
    """Return the deprecated legacy team-file directory.

    Legacy analyses still call this helper, but new workflows should prefer
    explicit `--team-file` inputs or finalized draft artifacts instead of an
    implicit team-file home.
    """
    return RAW_INPUTS_DIR / 'my_ff_teams'


def get_logs_dir() -> Path:
    """Get logs directory."""
    path = LOGS_DIR
    ensure_dir_exists(path)
    return path


def get_misc_datasets_dir() -> Path:
    """Get directory for user-managed raw helper datasets."""
    path = RAW_INPUTS_DIR / 'misc-datasets'
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


def get_default_team_file() -> Path:
    """Return the deprecated legacy default team file path."""
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


def reject_legacy_runtime_layout(runtime_root: Path | None = None) -> None:
    """Fail closed when deprecated runtime-root siblings are still populated."""
    root = runtime_root or get_runtime_root()
    legacy_dirs = [root / 'data', root / 'datasets', root / 'runs']
    existing = [path.name for path in legacy_dirs if path.exists()]
    if existing:
        raise RuntimeError(
            'Legacy runtime directories are still present under '
            f'{root}: {existing}. Remove them before continuing with the '
            'supported `inputs/` and `seasons/` layout.'
        )


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
    """Get the CSV companion to the canonical unified dataset JSON export."""
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
    """Get directory for canonical player-forecast outputs."""
    path = get_draft_model_outputs_dir(year) / 'player_forecast'
    ensure_dir_exists(path)
    return path


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

    reject_legacy_runtime_layout(RUNTIME_DIR)

    # Core directories
    ensure_dir_exists(LOGS_DIR)
    ensure_dir_exists(INPUTS_DIR)
    ensure_dir_exists(RAW_INPUTS_DIR)
    ensure_dir_exists(PROCESSED_INPUTS_DIR)
    ensure_dir_exists(RAW_SEASON_DATASETS_DIR)
    ensure_dir_exists(RAW_COMBINED_DATASETS_DIR)
    ensure_dir_exists(SEASON_DATASETS_DIR)
    ensure_dir_exists(COMBINED_DATASETS_DIR)
    ensure_dir_exists(SNAKE_DRAFT_DATASETS_DIR)
    ensure_dir_exists(UNIFIED_DATASET_DIR)
    ensure_dir_exists(RUNTIME_DASHBOARD_DIR)
    ensure_dir_exists(REPO_DASHBOARD_DIR)

    # Year-based season directories
    ensure_dir_exists(get_pre_draft_run_root(year))
    ensure_dir_exists(get_pre_draft_diagnostics_dir(year))
    ensure_dir_exists(get_vor_strategy_dir(year))
    ensure_dir_exists(get_draft_strategy_dir(year))
    ensure_dir_exists(get_draft_model_outputs_dir(year))
    ensure_dir_exists(get_bayesian_model_dir(year))
    ensure_dir_exists(get_finalized_drafts_dir(year))
    ensure_dir_exists(get_draft_retrospective_json_path(year).parent)
    ensure_dir_exists(get_validation_dir(year))

    ensure_dir_exists(get_misc_datasets_dir())

    print(f'✅ All required directories created for year {year}')
