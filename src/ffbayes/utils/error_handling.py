#!/usr/bin/env python3
"""
Enhanced Error Handling and Failure Detection for ffbayes Pipeline.

This module provides comprehensive error handling that addresses critical issues:
- Eliminate silent failures with immediate failure on loading problems
- Prevent cached data contamination with freshness validation
- Implement comprehensive data quality checks
- Provide detailed error diagnostics and recovery guidance
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .interface_standards import handle_exception, setup_logger


class PipelineError(Exception):
    """Base exception for pipeline errors with enhanced diagnostics."""
    
    def __init__(self, message: str, context: str = "", diagnostics: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context
        self.diagnostics = diagnostics or {}
        self.timestamp = datetime.now()
    
    def __str__(self):
        base_msg = f"[{self.context}] {super().__str__()}" if self.context else super().__str__()
        if self.diagnostics:
            diag_str = "; ".join([f"{k}: {v}" for k, v in self.diagnostics.items()])
            return f"{base_msg} (Diagnostics: {diag_str})"
        return base_msg


class DataValidationError(PipelineError):
    """Exception for data validation failures."""
    pass


class CacheContaminationError(PipelineError):
    """Exception for cached data contamination issues."""
    pass


class SilentFailureError(PipelineError):
    """Exception for silent failures that should be detected."""
    pass


class ErrorHandler:
    """Enhanced error handler with comprehensive failure detection."""
    
    def __init__(self, logger_name: str = "error_handler"):
        self.logger = setup_logger(logger_name)
        self.error_count = 0
        self.warning_count = 0
        self.critical_errors = []
    
    def validate_file_exists(self, file_path: Union[str, Path], context: str = "") -> None:
        """Validate that a file exists and is accessible.
        
        Args:
            file_path: Path to the file to validate
            context: Context for error reporting
            
        Raises:
            PipelineError: If file doesn't exist or is not accessible
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PipelineError(
                f"Required file not found: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "current_dir": str(Path.cwd()),
                    "file_exists": False
                }
            )
        
        if not file_path.is_file():
            raise PipelineError(
                f"Path exists but is not a file: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "is_file": file_path.is_file(),
                    "is_dir": file_path.is_dir()
                }
            )
        
        if not os.access(file_path, os.R_OK):
            raise PipelineError(
                f"File exists but is not readable: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "readable": os.access(file_path, os.R_OK),
                    "writable": os.access(file_path, os.W_OK)
                }
            )
    
    def validate_data_freshness(self, file_path: Union[str, Path], max_age_hours: int = 24, context: str = "") -> None:
        """Validate that data file is fresh (not stale cached data).
        
        Args:
            file_path: Path to the data file
            max_age_hours: Maximum age in hours before file is considered stale
            context: Context for error reporting
            
        Raises:
            CacheContaminationError: If file is too old (potential cached contamination)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise CacheContaminationError(
                f"Data file not found: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "max_age_hours": max_age_hours
                }
            )
        
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        max_age = timedelta(hours=max_age_hours)
        
        if file_age > max_age:
            raise CacheContaminationError(
                f"Data file is too old (potential cached contamination): {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "file_age_hours": file_age.total_seconds() / 3600,
                    "max_age_hours": max_age_hours,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            )
    
    def validate_json_structure(self, file_path: Union[str, Path], required_keys: List[str], context: str = "") -> Dict[str, Any]:
        """Validate JSON file structure and load data.
        
        Args:
            file_path: Path to JSON file
            required_keys: List of required top-level keys
            context: Context for error reporting
            
        Returns:
            Loaded JSON data
            
        Raises:
            DataValidationError: If JSON is invalid or missing required keys
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataValidationError(
                f"Invalid JSON in file: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "json_error": str(e),
                    "line": getattr(e, 'lineno', 'unknown'),
                    "column": getattr(e, 'colno', 'unknown')
                }
            )
        except Exception as e:
            raise DataValidationError(
                f"Failed to read JSON file: {file_path}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
        
        # Check for required keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise DataValidationError(
                f"JSON file missing required keys: {missing_keys}",
                context=context,
                diagnostics={
                    "file_path": str(file_path),
                    "missing_keys": missing_keys,
                    "available_keys": list(data.keys()),
                    "required_keys": required_keys
                }
            )
        
        return data
    
    def validate_dataframe_quality(self, df: pd.DataFrame, required_columns: List[str], min_rows: int = 1, context: str = "") -> None:
        """Validate DataFrame quality and structure.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns
            min_rows: Minimum number of rows required
            context: Context for error reporting
            
        Raises:
            DataValidationError: If DataFrame doesn't meet quality requirements
        """
        if df is None:
            raise DataValidationError(
                "DataFrame is None",
                context=context,
                diagnostics={
                    "dataframe_type": type(df).__name__,
                    "required_columns": required_columns,
                    "min_rows": min_rows
                }
            )
        
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Expected DataFrame, got {type(df).__name__}",
                context=context,
                diagnostics={
                    "actual_type": type(df).__name__,
                    "required_columns": required_columns,
                    "min_rows": min_rows
                }
            )
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(
                f"DataFrame missing required columns: {missing_columns}",
                context=context,
                diagnostics={
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                    "required_columns": required_columns,
                    "shape": df.shape
                }
            )
        
        # Check minimum rows
        if len(df) < min_rows:
            raise DataValidationError(
                f"DataFrame has insufficient rows: {len(df)} < {min_rows}",
                context=context,
                diagnostics={
                    "actual_rows": len(df),
                    "min_rows": min_rows,
                    "shape": df.shape,
                    "columns": list(df.columns)
                }
            )
        
        # Check for completely empty DataFrame
        if df.empty:
            raise DataValidationError(
                "DataFrame is completely empty",
                context=context,
                diagnostics={
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "min_rows": min_rows
                }
            )
    
    def validate_bayesian_results(self, results_file: Union[str, Path], context: str = "") -> Dict[str, Any]:
        """Validate Bayesian analysis results file.
        
        Args:
            results_file: Path to Bayesian results JSON file
            context: Context for error reporting
            
        Returns:
            Loaded Bayesian results
            
        Raises:
            DataValidationError: If results are invalid or incomplete
        """
        results_file = Path(results_file)
        
        # Validate file exists and is fresh
        self.validate_file_exists(results_file, context)
        self.validate_data_freshness(results_file, max_age_hours=6, context=context)  # Bayesian results should be recent
        
        # Validate JSON structure
        required_keys = ["model_evaluation", "player_predictions"]
        data = self.validate_json_structure(results_file, required_keys, context)
        
        # Validate player predictions
        player_predictions = data.get("player_predictions", {})
        if not player_predictions:
            raise DataValidationError(
                "Bayesian results contain no player predictions",
                context=context,
                diagnostics={
                    "file_path": str(results_file),
                    "player_predictions_count": len(player_predictions),
                    "available_keys": list(data.keys())
                }
            )
        
        # Validate individual player prediction structure
        for player_name, prediction in player_predictions.items():
            required_prediction_keys = ["mean", "std", "ci_lower", "ci_upper"]
            missing_keys = [key for key in required_prediction_keys if key not in prediction]
            if missing_keys:
                raise DataValidationError(
                    f"Player prediction missing required keys: {missing_keys}",
                    context=context,
                    diagnostics={
                        "player_name": player_name,
                        "missing_keys": missing_keys,
                        "available_keys": list(prediction.keys()),
                        "required_keys": required_prediction_keys
                    }
                )
        
        self.logger.info(f"✅ Bayesian results validated: {len(player_predictions)} player predictions")
        return data
    
    def validate_monte_carlo_results(self, results_file: Union[str, Path], context: str = "") -> pd.DataFrame:
        """Validate Monte Carlo simulation results.
        
        Args:
            results_file: Path to Monte Carlo results TSV file
            context: Context for error reporting
            
        Returns:
            Loaded Monte Carlo results DataFrame
            
        Raises:
            DataValidationError: If results are invalid or incomplete
        """
        results_file = Path(results_file)
        
        # Validate file exists and is fresh
        self.validate_file_exists(results_file, context)
        self.validate_data_freshness(results_file, max_age_hours=6, context=context)
        
        try:
            df = pd.read_csv(results_file, sep='\t')
        except Exception as e:
            raise DataValidationError(
                f"Failed to read Monte Carlo results: {results_file}",
                context=context,
                diagnostics={
                    "file_path": str(results_file),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
        
        # Validate DataFrame quality
        required_columns = ["projected_total"]  # Minimum required column
        self.validate_dataframe_quality(df, required_columns, min_rows=100, context=context)
        
        # Check for reasonable variance in projections (not all identical)
        if "projected_total" in df.columns:
            variance = df["projected_total"].var()
            if variance < 1.0:  # Very low variance suggests fake data
                raise DataValidationError(
                    f"Monte Carlo results show suspiciously low variance: {variance:.2f}",
                    context=context,
                    diagnostics={
                        "file_path": str(results_file),
                        "variance": variance,
                        "mean": df["projected_total"].mean(),
                        "std": df["projected_total"].std(),
                        "min": df["projected_total"].min(),
                        "max": df["projected_total"].max()
                    }
                )
        
        self.logger.info(f"✅ Monte Carlo results validated: {len(df)} simulations")
        return df
    
    def detect_silent_failures(self, step_name: str, expected_outputs: List[str], context: str = "") -> None:
        """Detect silent failures by checking for expected outputs.
        
        Args:
            step_name: Name of the pipeline step
            expected_outputs: List of expected output file patterns
            context: Context for error reporting
            
        Raises:
            SilentFailureError: If expected outputs are missing (silent failure)
        """
        missing_outputs = []
        
        for pattern in expected_outputs:
            matching_files = list(Path(".").glob(pattern))
            if not matching_files:
                missing_outputs.append(pattern)
        
        if missing_outputs:
            raise SilentFailureError(
                f"Step '{step_name}' completed but expected outputs are missing",
                context=context,
                diagnostics={
                    "step_name": step_name,
                    "missing_outputs": missing_outputs,
                    "expected_outputs": expected_outputs,
                    "current_directory": str(Path.cwd())
                }
            )
    
    def log_error(self, error: Exception, context: str = "", critical: bool = False) -> None:
        """Log error with enhanced diagnostics.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            critical: Whether this is a critical error
        """
        self.error_count += 1
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "critical": critical
        }
        
        if hasattr(error, 'diagnostics'):
            error_info["diagnostics"] = error.diagnostics
        
        if critical:
            self.critical_errors.append(error_info)
            self.logger.error(f"🚨 CRITICAL ERROR: {error}")
        else:
            self.logger.error(f"❌ ERROR: {error}")
        
        self.logger.debug(f"Error details: {json.dumps(error_info, indent=2)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered.
        
        Returns:
            Dictionary with error summary statistics
        """
        return {
            "total_errors": self.error_count,
            "total_warnings": self.warning_count,
            "critical_errors": len(self.critical_errors),
            "critical_error_details": self.critical_errors,
            "timestamp": datetime.now().isoformat()
        }


# Global error handler instance
error_handler = ErrorHandler()


def validate_pipeline_step(step_name: str, step_outputs: List[str], context: str = "") -> None:
    """Validate a pipeline step's outputs and detect silent failures.
    
    Args:
        step_name: Name of the pipeline step
        step_outputs: List of expected output file patterns
        context: Context for error reporting
        
    Raises:
        SilentFailureError: If step failed silently (missing outputs)
    """
    error_handler.detect_silent_failures(step_name, step_outputs, context)


def validate_bayesian_predictions(results_file: Union[str, Path], context: str = "") -> Dict[str, Any]:
    """Validate Bayesian predictions with comprehensive checks.
    
    Args:
        results_file: Path to Bayesian results file
        context: Context for error reporting
        
    Returns:
        Validated Bayesian results
        
    Raises:
        DataValidationError: If predictions are invalid
    """
    return error_handler.validate_bayesian_results(results_file, context)


def validate_monte_carlo_simulation(results_file: Union[str, Path], context: str = "") -> pd.DataFrame:
    """Validate Monte Carlo simulation results with comprehensive checks.
    
    Args:
        results_file: Path to Monte Carlo results file
        context: Context for error reporting
        
    Returns:
        Validated Monte Carlo results DataFrame
        
    Raises:
        DataValidationError: If results are invalid
    """
    return error_handler.validate_monte_carlo_results(results_file, context)
