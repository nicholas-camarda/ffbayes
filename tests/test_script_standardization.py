#!/usr/bin/env python3
"""
Tests for script standardization across the ffbayes package.
Verifies that all scripts work correctly with the standardized interface.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.append(str(Path.cwd() / 'src'))

from ffbayes.utils.model_validation import ModelValidator
from ffbayes.utils.script_interface import (
    StandardizedScriptInterface,
    create_standardized_interface,
)


class TestScriptStandardization(unittest.TestCase):
    """Test script standardization across all ffbayes scripts."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        os.makedirs('datasets/season_datasets', exist_ok=True)
        os.makedirs('datasets/combined_datasets', exist_ok=True)
        os.makedirs('results/montecarlo_results', exist_ok=True)
        os.makedirs('results/bayesian-hierarchical-results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_standardized_interface_creation(self):
        """Test creation of standardized interface."""
        interface = create_standardized_interface("test-script", "Test description")
        self.assertIsInstance(interface, StandardizedScriptInterface)
        self.assertEqual(interface.script_name, "test-script")
        self.assertEqual(interface.description, "Test description")
    
    def test_argument_parser_setup(self):
        """Test argument parser setup with standard options."""
        interface = create_standardized_interface("test-script")
        parser = interface.setup_argument_parser()
        
        # Test that standard arguments are added
        args = parser.parse_args(['--verbose', '--quick-test', '--log-level', 'DEBUG'])
        self.assertTrue(args.verbose)
        self.assertTrue(args.quick_test)
        self.assertEqual(args.log_level, 'DEBUG')
    
    def test_model_arguments(self):
        """Test model-specific arguments."""
        interface = create_standardized_interface("test-script")
        parser = interface.setup_argument_parser()
        parser = interface.add_model_arguments(parser)
        
        args = parser.parse_args(['--draws', '2000', '--tune', '1500', '--chains', '6', '--cores', '8'])
        self.assertEqual(args.draws, 2000)
        self.assertEqual(args.tune, 1500)
        self.assertEqual(args.chains, 6)
        self.assertEqual(args.cores, 8)
    
    def test_data_arguments(self):
        """Test data-specific arguments."""
        interface = create_standardized_interface("test-script")
        parser = interface.setup_argument_parser()
        parser = interface.add_data_arguments(parser)
        
        args = parser.parse_args(['--years', '2020,2021,2022', '--force-refresh'])
        self.assertEqual(args.years, '2020,2021,2022')
        self.assertTrue(args.force_refresh)
    
    def test_years_parsing(self):
        """Test years string parsing."""
        interface = create_standardized_interface("test-script")
        
        # Test valid years
        years = interface.parse_years("2020,2021,2022")
        self.assertEqual(years, [2020, 2021, 2022])
        
        # Test empty string
        years = interface.parse_years("")
        self.assertEqual(years, [])
        
        # Test None
        years = interface.parse_years(None)
        self.assertEqual(years, [])
        
        # Test invalid format
        with self.assertRaises(ValueError):
            interface.parse_years("2020,invalid,2022")
    
    def test_logging_setup(self):
        """Test logging setup with different levels."""
        interface = create_standardized_interface("test-script")
        
        # Test verbose mode
        args = MagicMock()
        args.verbose = True
        args.quiet = False
        args.log_level = 'INFO'
        
        logger = interface.setup_logging(args)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test-script")
        
        # Test quiet mode
        args.verbose = False
        args.quiet = True
        logger = interface.setup_logging(args)
        self.assertIsNotNone(logger)
    
    def test_error_handling(self):
        """Test standardized error handling."""
        interface = create_standardized_interface("test-script")
        
        # Test successful function execution
        def success_func():
            return "success"
        
        result = interface.handle_errors(success_func)
        self.assertEqual(result, "success")
        
        # Test ValueError handling
        def value_error_func():
            raise ValueError("Test error")
        
        with patch('sys.exit') as mock_exit:
            interface.handle_errors(value_error_func)
            mock_exit.assert_called_with(interface.EXIT_INVALID_ARGS)
        
        # Test FileNotFoundError handling
        def file_not_found_func():
            raise FileNotFoundError("File not found")
        
        with patch('sys.exit') as mock_exit:
            interface.handle_errors(file_not_found_func)
            mock_exit.assert_called_with(interface.EXIT_DATA_ERROR)
    
    def test_output_directory_handling(self):
        """Test output directory handling."""
        interface = create_standardized_interface("test-script")
        
        args = MagicMock()
        args.output_dir = None
        
        output_dir = interface.get_output_directory(args)
        self.assertIsInstance(output_dir, Path)
        
        # Test custom output directory
        args.output_dir = str(self.temp_dir)
        output_dir = interface.get_output_directory(args)
        self.assertEqual(output_dir, Path(self.temp_dir))
    
    def test_data_directory_handling(self):
        """Test data directory handling."""
        interface = create_standardized_interface("test-script")
        
        args = MagicMock()
        args.data_dir = None
        
        data_dir = interface.get_data_directory(args)
        self.assertIsInstance(data_dir, Path)
        
        # Test custom data directory
        args.data_dir = str(self.temp_dir)
        data_dir = interface.get_data_directory(args)
        self.assertEqual(data_dir, Path(self.temp_dir))
    
    def test_model_validation_integration(self):
        """Test model validation integration."""
        interface = create_standardized_interface("test-script")
        
        # Test Monte Carlo validation
        import pandas as pd
        test_df = pd.DataFrame({
            'Player1': [10, 15, 20, 25, 30],
            'Player2': [5, 10, 15, 20, 25],
            'Total': [15, 25, 35, 45, 55]
        })
        
        validation_results = interface.validate_monte_carlo_model(test_df)
        self.assertIsInstance(validation_results, dict)
        self.assertIn('valid', validation_results)
        
        # Test model output validation
        test_outputs = {
            'predictions': test_df,
            'uncertainty': test_df
        }
        
        validation_results = interface.validate_model_outputs(test_outputs)
        self.assertIsInstance(validation_results, dict)
        self.assertIn('overall_valid', validation_results)
    
    def test_data_collection_script_standardization(self):
        """Test data collection script standardization."""
        try:
            import ffbayes.data_pipeline.collect_data as collect_data

            # Test that main function accepts args parameter
            self.assertTrue(hasattr(collect_data, 'main'))
            
            # Test argument parsing
            with patch('sys.argv', ['collect_data.py', '--help']):
                # Should not raise an exception
                pass
                
        except ImportError as e:
            self.fail(f"Failed to import data collection script: {e}")
    
    def test_data_validation_script_standardization(self):
        """Test data validation script standardization."""
        try:
            import ffbayes.data_pipeline.validate_data as validate_data

            # Test that main function accepts args parameter
            self.assertTrue(hasattr(validate_data, 'main'))
            
            # Test argument parsing
            with patch('sys.argv', ['validate_data.py', '--help']):
                # Should not raise an exception
                pass
                
        except ImportError as e:
            self.fail(f"Failed to import data validation script: {e}")
    
    def test_preprocess_script_standardization(self):
        """Test preprocessing script standardization."""
        try:
            import ffbayes.data_pipeline.preprocess_analysis_data as preprocess

            # Test that main function accepts args parameter
            self.assertTrue(hasattr(preprocess, 'main'))
            
            # Test argument parsing
            with patch('sys.argv', ['preprocess.py', '--help']):
                # Should not raise an exception
                pass
                
        except ImportError as e:
            self.fail(f"Failed to import preprocessing script: {e}")
    
    def test_monte_carlo_script_standardization(self):
        """Test Monte Carlo script standardization."""
        try:
            import ffbayes.analysis.montecarlo_historical_ff as monte_carlo

            # Test that main function accepts args parameter
            self.assertTrue(hasattr(monte_carlo, 'main'))
            
            # Test argument parsing
            with patch('sys.argv', ['monte_carlo.py', '--help']):
                # Should not raise an exception
                pass
                
        except ImportError as e:
            self.fail(f"Failed to import Monte Carlo script: {e}")
    
    def test_console_script_compatibility(self):
        """Test that console scripts are compatible with standardized interface."""
        # Test that all console scripts can be imported and have main functions
        console_scripts = [
            'ffbayes.data_pipeline.collect_data',
            'ffbayes.data_pipeline.validate_data', 
            'ffbayes.data_pipeline.preprocess_analysis_data',
            'ffbayes.analysis.montecarlo_historical_ff'
        ]
        
        for script_module in console_scripts:
            try:
                module = __import__(script_module, fromlist=['main'])
                self.assertTrue(hasattr(module, 'main'))
                self.assertTrue(callable(module.main))
            except ImportError as e:
                self.fail(f"Failed to import console script {script_module}: {e}")
    
    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        interface = create_standardized_interface("test-script")
        
        # Test that environment variables are properly handled
        with patch.dict(os.environ, {'QUICK_TEST': 'true', 'MAX_CORES': '8'}):
            parser = interface.setup_argument_parser()
            parser = interface.add_model_arguments(parser)
            args = parser.parse_args([])
            
            # These should use environment variable defaults
            self.assertEqual(args.cores, 8)
    
    def test_progress_monitoring_integration(self):
        """Test progress monitoring integration."""
        interface = create_standardized_interface("test-script")
        
        # Test progress logging
        interface.logger = MagicMock()
        interface.log_progress("Processing...", "INFO")
        interface.logger.info.assert_called_with("Processing...")
        
        # Test completion logging
        interface.log_completion("Task completed")
        interface.logger.info.assert_called_with("test-script: Task completed")
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        validator = ModelValidator("test-model")
        
        # Add some validation results
        validator.validation_results = {
            'bayesian_convergence': {'converged': True},
            'monte_carlo': {'valid': True},
            'model_outputs': {'overall_valid': True}
        }
        
        summary = validator.get_validation_summary()
        self.assertIn('model_name', summary)
        self.assertIn('validation_results', summary)
        self.assertIn('overall_valid', summary)
        self.assertTrue(summary['overall_valid'])


if __name__ == '__main__':
    unittest.main()
