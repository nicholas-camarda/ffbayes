#!/usr/bin/env python3
"""
interface_standards.py

Lightweight utilities to standardize script interfaces across the pipeline
without requiring invasive refactors. Scripts can gradually adopt these helpers.

Provides:
- setup_logger(name): standardized logging configuration using LOG_LEVEL env
- get_env_bool/get_env_int: helper accessors for common env vars (e.g., QUICK_TEST)
- get_standard_paths(): central place for standard output/input directories
- handle_exception(e, context): consistent error formatting for exceptions
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def setup_logger(name: str) -> logging.Logger:
	"""Create and configure a logger with level from LOG_LEVEL env (default INFO)."""
	logger = logging.getLogger(name)
	if not logger.handlers:
		log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
		level = getattr(logging, log_level_str, logging.INFO)
		logger.setLevel(level)
		ch = logging.StreamHandler()
		ch.setLevel(level)
		formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
		ch.setFormatter(formatter)
		logger.addHandler(ch)
		logger.propagate = False
	return logger


def get_env_bool(name: str, default: bool = False) -> bool:
	"""Get a boolean environment variable with common truthy/falsey parsing."""
	val = os.getenv(name)
	if val is None:
		return default
	return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def get_env_int(name: str, default: int) -> int:
	"""Get an integer environment variable with fallback to default on error."""
	try:
		return int(os.getenv(name, str(default)))
	except (TypeError, ValueError):
		return default


@dataclass(frozen=True)
class StandardPaths:
	plots_root: Path
	results_root: Path
	datasets_root: Path
	monte_carlo_results: Path
	bayesian_results: Path
	team_aggregation_plots: Path
	test_runs_plots: Path


def get_standard_paths(project_root: Optional[Path] = None) -> StandardPaths:
	"""Return standard directories used across the project.
	
	Directories are not created here; run_pipeline.py is responsible for creation.
	"""
	root = Path(project_root) if project_root else Path.cwd()
	plots_root = root / "plots"
	results_root = root / "results"
	datasets_root = root / "datasets"
	return StandardPaths(
		plots_root=plots_root,
		results_root=results_root,
		datasets_root=datasets_root,
		monte_carlo_results=results_root / "montecarlo_results",
		bayesian_results=results_root / "bayesian-hierarchical-results",
		team_aggregation_plots=plots_root / "team_aggregation",
		test_runs_plots=plots_root / "test_runs",
	)


def handle_exception(error: Exception, context: str = "") -> str:
	"""Format exceptions consistently for logs and user output."""
	prefix = f"[{context}] " if context else ""
	return f"{prefix}{type(error).__name__}: {error}"
