"""
File naming utilities for consistent file naming conventions across FFBayes.

This module provides standardized naming patterns for all output files to ensure
scalability, readability, and programmatic access.
"""

from pathlib import Path
from typing import List, Optional


def get_monte_carlo_filename(
    current_year: int,
    training_years: List[int],
    file_type: str = "projections"
) -> str:
    """
    Generate a consistent, readable filename for Monte Carlo results.
    
    Args:
        current_year: The year being projected (e.g., 2025)
        training_years: List of years used for training (e.g., [2023, 2024])
        file_type: Type of file (e.g., "projections", "simulations")
    
    Returns:
        Consistent filename like "mc_projections_2025_trained_on_2023-2024.tsv"
    
    Examples:
        >>> get_monte_carlo_filename(2025, [2023, 2024])
        'mc_projections_2025_trained_on_2023-2024.tsv'
        
        >>> get_monte_carlo_filename(2026, [2022, 2023, 2024, 2025])
        'mc_projections_2026_trained_on_2022-2025.tsv'
    """
    # Sort training years for consistency
    training_years_sorted = sorted(training_years)
    
    if len(training_years_sorted) == 1:
        training_range = str(training_years_sorted[0])
    else:
        training_range = f"{training_years_sorted[0]}-{training_years_sorted[-1]}"
    
    return f"mc_{file_type}_{current_year}_trained_on_{training_range}.tsv"


def get_baseline_filename(current_year: int) -> str:
    """
    Generate consistent filename for baseline model results.
    
    Args:
        current_year: The year being projected
    
    Returns:
        Consistent filename like "baseline_model_results.json"
    """
    return "baseline_model_results.json"


def get_hybrid_mc_filename(current_year: int) -> str:
    """
    Generate consistent filename for hybrid MC model results.
    
    Args:
        current_year: The year being projected
    
    Returns:
        Consistent filename like "hybrid_mc_model_results.json"
    """
    return "hybrid_mc_model_results.json"


def get_team_aggregation_filename(current_year: int, team_name: Optional[str] = None) -> str:
    """
    Generate consistent filename for team aggregation results.
    
    Args:
        current_year: The year being analyzed
        team_name: Optional team identifier
    
    Returns:
        Consistent filename like "team_aggregation_results.json" or 
        "team_aggregation_results_my_team.json"
    """
    if team_name:
        return f"team_aggregation_results_{team_name}.json"
    return "team_aggregation_results.json"


def get_draft_strategy_filename(
    current_year: int,
    draft_position: int,
    league_size: int,
    risk_tolerance: str = "medium"
) -> str:
    """
    Generate consistent filename for draft strategy results.
    
    Args:
        current_year: The year being drafted
        draft_position: Draft position (1-based)
        league_size: Number of teams in league
        risk_tolerance: Risk tolerance level
    
    Returns:
        Consistent filename like "draft_strategy_pos3_12team_medium_risk.json"
    """
    return f"draft_strategy_pos{draft_position}_{league_size}team_{risk_tolerance}_risk.json"


def get_vor_strategy_filename(current_year: int, ppr: float = None, top_n: int = None) -> str:
    """
    Generate consistent filename for VOR strategy results.
    
    Args:
        current_year: The year being analyzed
        ppr: Points per reception (uses centralized config if None)
        top_n: Number of top players (uses centralized config if None)
    
    Returns:
        Consistent filename like "vor_strategy_ppr0.5_top120.csv"
    """
    if ppr is None or top_n is None:
        try:
            from ffbayes.utils.config_loader import get_config
            config = get_config()
            ppr = ppr or config.get_vor_setting('ppr')
            top_n = top_n or config.get_vor_setting('top_rank')
        except ImportError:
            ppr = ppr or 0.5
            top_n = top_n or 120
    
    return f"vor_strategy_ppr{ppr}_top{top_n}.csv"


def get_output_directory(
    base_type: str,
    current_year: int,
    is_quick_test: bool = False
) -> Path:
    """
    Generate consistent output directory structure.
    
    Args:
        base_type: Type of output (e.g., "montecarlo_results", "baseline_results")
        current_year: The year being analyzed
        is_quick_test: Whether running in quick test mode
    
    Returns:
        Path to output directory
    """
    if is_quick_test:
        # Test runs go to test_runs directory for easy cleanup
        from ffbayes.utils.path_constants import get_plots_dir
        return get_plots_dir(current_year) / "test_runs"
    else:
        # Production runs go to organized year-based directories
        from ffbayes.utils.path_constants import get_results_dir
        return get_results_dir(current_year) / base_type


def get_latest_file_by_pattern(
    directory: Path,
    filename_pattern: str,
    file_extension: str = ".tsv"
) -> Optional[Path]:
    """
    Find the most recent file matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        filename_pattern: Pattern to match (e.g., "mc_projections_2025")
        file_extension: File extension to filter by
    
    Returns:
        Path to most recent matching file, or None if not found
    """
    if not directory.exists():
        return None
    
    # Find all files matching the pattern
    matching_files = list(directory.glob(f"{filename_pattern}*{file_extension}"))
    
    if not matching_files:
        return None
    
    # Return the most recent file
    return max(matching_files, key=lambda f: f.stat().st_mtime)


def get_monte_carlo_file_path(
    current_year: int,
    training_years: List[int],
    is_quick_test: bool = False
) -> Path:
    """
    Get the full path for Monte Carlo results using consistent naming.
    
    Args:
        current_year: The year being projected
        training_years: List of years used for training
        is_quick_test: Whether running in quick test mode
    
    Returns:
        Full path to Monte Carlo results file
    """
    output_dir = get_output_directory("montecarlo_results", current_year, is_quick_test)
    filename = get_monte_carlo_filename(current_year, training_years)
    return output_dir / filename


def get_baseline_file_path(current_year: int) -> Path:
    """
    Get the full path for baseline model results.
    
    Args:
        current_year: The year being projected
    
    Returns:
        Full path to baseline results file
    """
    output_dir = get_output_directory("baseline_results", current_year)
    filename = get_baseline_filename(current_year)
    return output_dir / filename


def get_hybrid_mc_file_path(current_year: int) -> Path:
    """
    Get the full path for hybrid MC model results.
    
    Args:
        current_year: The year being projected
    
    Returns:
        Full path to hybrid MC results file
    """
    output_dir = get_output_directory("hybrid_mc_bayesian", current_year)
    filename = get_hybrid_mc_filename(current_year)
    return output_dir / filename


def get_team_aggregation_file_path(
    current_year: int,
    team_name: Optional[str] = None,
    is_quick_test: bool = False
) -> Path:
    """
    Get the full path for team aggregation results.
    
    Args:
        current_year: The year being analyzed
        team_name: Optional team identifier
        is_quick_test: Whether running in quick test mode
    
    Returns:
        Full path to team aggregation results file
    """
    output_dir = get_output_directory("team_aggregation", current_year, is_quick_test)
    filename = get_team_aggregation_filename(current_year, team_name)
    return output_dir / filename


# Legacy filename patterns for backward compatibility
LEGACY_MONTE_CARLO_PATTERNS = [
    "{year}_projections_from_years{years}.tsv",
    "monte_carlo_results_{year}.tsv",
    "mc_results_{year}.tsv"
]


def find_monte_carlo_file_legacy(
    current_year: int,
    training_years: List[int]
) -> Optional[Path]:
    """
    Find Monte Carlo results using legacy filename patterns.
    
    This function helps with backward compatibility while transitioning
    to the new naming convention.
    
    Args:
        current_year: The year being projected
        training_years: List of years used for training
    
    Returns:
        Path to legacy Monte Carlo file if found, None otherwise
    """
    # Try the new naming convention first
    new_path = get_monte_carlo_file_path(current_year, training_years)
    if new_path.exists():
        return new_path
    
    # Try legacy patterns
    from ffbayes.utils.path_constants import get_monte_carlo_dir
    output_dir = get_monte_carlo_dir(current_year)
    if not output_dir.exists():
        return None
    
    # Legacy pattern: 2025_projections_from_years[2023, 2024].tsv
    legacy_pattern = f"{current_year}_projections_from_years{training_years}.tsv"
    legacy_path = output_dir / legacy_pattern
    
    if legacy_path.exists():
        return legacy_path
    
    # Try other legacy patterns
    for pattern in LEGACY_MONTE_CARLO_PATTERNS:
        try:
            legacy_filename = pattern.format(year=current_year, years=training_years)
            legacy_path = output_dir / legacy_filename
            if legacy_path.exists():
                return legacy_path
        except (KeyError, ValueError):
            continue
    
    return None
