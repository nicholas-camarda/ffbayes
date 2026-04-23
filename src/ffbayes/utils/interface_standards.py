#!/usr/bin/env python3
"""Shared CLI and path helpers for pipeline-facing scripts.

These helpers keep public script behavior aligned with the canonical runtime
layout: local working inputs under `inputs/`, season-scoped outputs under
`seasons/<year>/`, and derived plots/results surfaces created by the scripts
that explicitly need them.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger from the `LOG_LEVEL` environment variable."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using common truthy strings."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


def get_env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back to `default`."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class StandardPaths:
    """Common script-facing path bundle derived from the canonical runtime tree."""

    plots_root: Path
    results_root: Path
    inputs_root: Path
    monte_carlo_results: Path
    bayesian_results: Path
    team_aggregation_plots: Path
    test_runs_plots: Path


def get_standard_paths(project_root: Optional[Path] = None) -> StandardPaths:
    """Return standard directories used across the current runtime layout.

    Directories are described here but not created automatically; callers that
    write outputs remain responsible for creating the paths they need.
    """
    if project_root:
        root = Path(project_root)
        plots_root = root / 'plots'
        results_root = root / 'results'
        inputs_root = root / 'inputs'
    else:
        from ffbayes.utils.path_constants import (
            INPUTS_DIR,
            get_plots_dir,
            get_results_dir,
        )

        plots_root = get_plots_dir(None)
        results_root = get_results_dir(None)
        inputs_root = INPUTS_DIR
    return StandardPaths(
        plots_root=plots_root,
        results_root=results_root,
        inputs_root=inputs_root,
        monte_carlo_results=results_root / 'montecarlo_results',
        bayesian_results=results_root / 'bayesian-hierarchical-results',
        team_aggregation_plots=plots_root / 'team_aggregation',
        test_runs_plots=plots_root / 'test_runs',
    )


def handle_exception(error: Exception, context: str = "") -> str:
    """Format an exception consistently for logs and CLI output."""
    prefix = f'[{context}] ' if context else ''
    return f'{prefix}{type(error).__name__}: {error}'
