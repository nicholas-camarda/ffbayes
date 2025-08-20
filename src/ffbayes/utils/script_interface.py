#!/usr/bin/env python3
"""
Standardized script interface for ffbayes package.
Provides consistent argument parsing, error handling, logging, and progress monitoring.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interface_standards import get_env_int, get_standard_paths, handle_exception, setup_logger
from .model_validation import (
    validate_bayesian_model,
    validate_model_outputs,
    validate_monte_carlo_model,
)


class StandardizedScriptInterface:
    """Standardized interface for all ffbayes scripts."""
    
    def __init__(self, script_name: str, description: str = ""):
        """Initialize standardized script interface.
        
        Args:
            script_name: Name of the script for logging and error messages
            description: Description of the script for help text
        """
        self.script_name = script_name
        self.description = description
        self.logger = None
        self.args = None
        
        # Standard exit codes
        self.EXIT_SUCCESS = 0
        self.EXIT_ERROR = 1
        self.EXIT_INVALID_ARGS = 2
        self.EXIT_CONFIG_ERROR = 3
        self.EXIT_DATA_ERROR = 4
        
    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Set up standardized argument parser with common options.
        
        Returns:
            Configured argument parser with standard options
        """
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Standard options for all scripts
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-error output'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level (default: INFO)'
        )
        
        parser.add_argument(
            '--quick-test',
            action='store_true',
            help='Run in quick test mode (reduced iterations/data)'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        
        parser.add_argument(
            '--output-dir',
            type=str,
            help='Output directory (overrides default)'
        )
        
        return parser
    
    def add_model_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model-specific arguments to parser.
        
        Args:
            parser: Argument parser to add model arguments to
            
        Returns:
            Parser with model arguments added
        """
        parser.add_argument(
            '--draws',
            type=int,
            default=get_env_int('DRAWS', 1000),
            help='Number of draws for Bayesian models (default: 1000)'
        )
        
        parser.add_argument(
            '--tune',
            type=int,
            default=get_env_int('TUNE', 1000),
            help='Number of tuning steps for Bayesian models (default: 1000)'
        )
        
        parser.add_argument(
            '--chains',
            type=int,
            default=get_env_int('CHAINS', 4),
            help='Number of chains for Bayesian models (default: 4)'
        )
        
        parser.add_argument(
            '--cores',
            type=int,
            default=get_env_int('MAX_CORES', 4),
            help='Number of CPU cores to use (default: 4)'
        )
        
        return parser
    
    def add_data_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add data-specific arguments to parser.
        
        Args:
            parser: Argument parser to add data arguments to
            
        Returns:
            Parser with data arguments added
        """
        parser.add_argument(
            '--years',
            type=str,
            help='Comma-separated list of years to process (e.g., "2020,2021,2022")'
        )
        
        parser.add_argument(
            '--data-dir',
            type=str,
            help='Data directory (overrides default)'
        )
        
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Force refresh of existing data'
        )
        
        return parser
    
    def setup_logging(self, args: argparse.Namespace) -> logging.Logger:
        """Set up standardized logging.
        
        Args:
            args: Parsed arguments containing logging configuration
            
        Returns:
            Configured logger
        """
        # Determine log level
        if args.quiet:
            log_level = 'ERROR'
        elif args.verbose:
            log_level = 'DEBUG'
        else:
            log_level = args.log_level
        
        # Set up logger
        self.logger = setup_logger(self.script_name)
        
        # Log script start
        self.logger.info(f"Starting {self.script_name}")
        if args.quick_test:
            self.logger.info("Running in QUICK_TEST mode")
        
        return self.logger
    
    def parse_arguments(self, argv: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments with standardized interface.
        
        Args:
            argv: Command line arguments (defaults to sys.argv[1:])
            
        Returns:
            Parsed arguments
        """
        if argv is None:
            argv = sys.argv[1:]
        
        parser = self.setup_argument_parser()
        self.args = parser.parse_args(argv)
        
        # Set up logging
        self.logger = self.setup_logging(self.args)
        
        return self.args
    
    def handle_errors(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with standardized error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            SystemExit: If function fails with appropriate exit code
        """
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            sys.exit(self.EXIT_SUCCESS)
        except ValueError as e:
            error_msg = handle_exception(e, f"{self.script_name} validation error")
            if self.logger:
                self.logger.error(error_msg)
            sys.exit(self.EXIT_INVALID_ARGS)
        except FileNotFoundError as e:
            error_msg = handle_exception(e, f"{self.script_name} file not found")
            if self.logger:
                self.logger.error(error_msg)
            sys.exit(self.EXIT_DATA_ERROR)
        except PermissionError as e:
            error_msg = handle_exception(e, f"{self.script_name} permission error")
            if self.logger:
                self.logger.error(error_msg)
            sys.exit(self.EXIT_ERROR)
        except Exception as e:
            error_msg = handle_exception(e, f"{self.script_name} unexpected error")
            if self.logger:
                self.logger.error(error_msg)
            sys.exit(self.EXIT_ERROR)
    
    def get_output_directory(self, args: argparse.Namespace) -> Path:
        """Get standardized output directory.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Output directory path
        """
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            paths = get_standard_paths()
            output_dir = paths.plots_root
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def get_data_directory(self, args: argparse.Namespace) -> Path:
        """Get standardized data directory.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Data directory path
        """
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            paths = get_standard_paths()
            data_dir = paths.datasets_root
        
        return data_dir
    
    def parse_years(self, years_str: Optional[str]) -> List[int]:
        """Parse years string into list of integers.
        
        Args:
            years_str: Comma-separated string of years
            
        Returns:
            List of years as integers
        """
        if not years_str:
            return []
        
        try:
            years = [int(year.strip()) for year in years_str.split(',')]
            return years
        except ValueError as e:
            raise ValueError(f"Invalid years format: {years_str}. Expected comma-separated integers.")
    
    def log_progress(self, message: str, level: str = 'INFO'):
        """Log progress message with standardized format.
        
        Args:
            message: Progress message
            level: Log level
        """
        if self.logger:
            getattr(self.logger, level.lower())(message)
    
    def log_completion(self, message: str = "Completed successfully"):
        """Log completion message.
        
        Args:
            message: Completion message
        """
        if self.logger:
            self.logger.info(f"{self.script_name}: {message}")
    
    def log_error(self, message: str, exit_code: int = None):
        """Log error message and optionally exit.
        
        Args:
            message: Error message
            exit_code: Exit code (if None, don't exit)
        """
        if self.logger:
            self.logger.error(f"{self.script_name}: {message}")
        
        if exit_code is not None:
            sys.exit(exit_code)
    
    def validate_bayesian_model(self, trace, model_name: str = None) -> Dict[str, Any]:
        """Validate Bayesian model convergence.
        
        Args:
            trace: PyMC trace object
            model_name: Name of the model (defaults to script name)
            
        Returns:
            Validation results dictionary
        """
        if model_name is None:
            model_name = self.script_name
        
        return validate_bayesian_model(trace, model_name)
    
    def validate_monte_carlo_model(self, results_df, model_name: str = None) -> Dict[str, Any]:
        """Validate Monte Carlo simulation results.
        
        Args:
            results_df: DataFrame with simulation results
            model_name: Name of the model (defaults to script name)
            
        Returns:
            Validation results dictionary
        """
        if model_name is None:
            model_name = self.script_name
        
        return validate_monte_carlo_model(results_df, model_name)
    
    def validate_model_outputs(self, outputs: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """Validate general model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            model_name: Name of the model (defaults to script name)
            
        Returns:
            Validation results dictionary
        """
        if model_name is None:
            model_name = self.script_name
        
        return validate_model_outputs(outputs, model_name)


def create_standardized_interface(script_name: str, description: str = "") -> StandardizedScriptInterface:
    """Create a standardized script interface.
    
    Args:
        script_name: Name of the script
        description: Description of the script
        
    Returns:
        StandardizedScriptInterface instance
    """
    return StandardizedScriptInterface(script_name, description)


def run_with_standardized_interface(script_name: str, main_func: callable, description: str = ""):
    """Run a script with standardized interface.
    
    Args:
        script_name: Name of the script
        main_func: Main function to execute
        description: Description of the script
    """
    interface = create_standardized_interface(script_name, description)
    
    try:
        # Parse arguments
        args = interface.parse_arguments()
        
        # Execute main function with error handling
        result = interface.handle_errors(main_func, args)
        
        # Log completion
        interface.log_completion()
        
        return result
        
    except SystemExit:
        # Re-raise system exit
        raise
    except Exception as e:
        # Handle any unexpected errors
        interface.log_error(f"Unexpected error: {e}", interface.EXIT_ERROR)
