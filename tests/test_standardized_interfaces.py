#!/usr/bin/env python3
"""
Tests for standardized script interfaces across the ffbayes package.
Ensures all scripts have consistent argument parsing, error handling, logging, and progress monitoring.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.append(str(Path.cwd() / 'src'))

import ffbayes.utils.interface_standards as interface_standards
from ffbayes.utils.interface_standards import (
    get_env_bool,
    get_env_int,
    get_standard_paths,
    handle_exception,
    setup_logger,
)


class TestStandardizedInterfaces(unittest.TestCase):
    """Test standardized script interfaces across all ffbayes scripts."""
    
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
    
    def test_interface_standards_imports(self):
        """Test that interface standards module can be imported."""
        self.assertIsNotNone(interface_standards)
        self.assertTrue(hasattr(interface_standards, 'setup_logger'))
        self.assertTrue(hasattr(interface_standards, 'get_env_bool'))
        self.assertTrue(hasattr(interface_standards, 'get_env_int'))
        self.assertTrue(hasattr(interface_standards, 'get_standard_paths'))
        self.assertTrue(hasattr(interface_standards, 'handle_exception'))
    
    def test_setup_logger(self):
        """Test standardized logger setup."""
        logger = setup_logger('test_script')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_script')
        # Level depends on LOG_LEVEL env var, so just check it's set
        self.assertIsInstance(logger.level, int)
    
    def test_get_env_bool(self):
        """Test environment variable boolean parsing."""
        # Test with unset variable
        self.assertFalse(get_env_bool('NONEXISTENT_VAR'))
        
        # Test with explicit False
        with patch.dict(os.environ, {'TEST_VAR': 'false'}):
            self.assertFalse(get_env_bool('TEST_VAR'))
        
        # Test with explicit True
        with patch.dict(os.environ, {'TEST_VAR': 'true'}):
            self.assertTrue(get_env_bool('TEST_VAR'))
        
        # Test with default value
        self.assertTrue(get_env_bool('NONEXISTENT_VAR', default=True))
    
    def test_get_env_int(self):
        """Test environment variable integer parsing."""
        # Test with unset variable
        self.assertEqual(get_env_int('NONEXISTENT_VAR', default=0), 0)
        
        # Test with valid integer
        with patch.dict(os.environ, {'TEST_VAR': '42'}):
            self.assertEqual(get_env_int('TEST_VAR', default=0), 42)
        
        # Test with default value
        self.assertEqual(get_env_int('NONEXISTENT_VAR', default=100), 100)
        
        # Test with invalid integer
        with patch.dict(os.environ, {'TEST_VAR': 'invalid'}):
            self.assertEqual(get_env_int('TEST_VAR', default=0), 0)
    
    def test_get_standard_paths(self):
        """Test standard path resolution."""
        paths = get_standard_paths()
        self.assertIsNotNone(paths)
        self.assertTrue(hasattr(paths, 'datasets_root'))
        self.assertTrue(hasattr(paths, 'results_root'))
        self.assertTrue(hasattr(paths, 'plots_root'))
    
    def test_handle_exception(self):
        """Test standardized exception handling."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_msg = handle_exception(e, "Test operation")
            self.assertIn("Test operation", error_msg)
            self.assertIn("Test error", error_msg)
    
    def test_data_collection_script_interface(self):
        """Test data collection script has standardized interface."""
        try:
            import ffbayes.data_pipeline.collect_data as collect_data
            self.assertTrue(hasattr(collect_data, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(collect_data.main))
            self.assertTrue(hasattr(collect_data, 'collect_nfl_data'))
            self.assertTrue(hasattr(collect_data, 'collect_data_by_year'))
            
        except ImportError as e:
            self.fail(f"Failed to import data collection script: {e}")
    
    def test_data_validation_script_interface(self):
        """Test data validation script has standardized interface."""
        try:
            import ffbayes.data_pipeline.validate_data as validate_data
            self.assertTrue(hasattr(validate_data, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(validate_data.main))
            self.assertTrue(hasattr(validate_data, 'validate_data_quality'))
            
        except ImportError as e:
            self.fail(f"Failed to import data validation script: {e}")
    
    def test_preprocess_script_interface(self):
        """Test data preprocessing script has standardized interface."""
        try:
            import ffbayes.data_pipeline.preprocess_analysis_data as preprocess
            self.assertTrue(hasattr(preprocess, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(preprocess.main))
            self.assertTrue(hasattr(preprocess, 'create_analysis_dataset'))
            
        except ImportError as e:
            self.fail(f"Failed to import preprocessing script: {e}")
    
    def test_monte_carlo_script_interface(self):
        """Test Monte Carlo script has standardized interface."""
        try:
            import ffbayes.analysis.montecarlo_historical_ff as monte_carlo
            self.assertTrue(hasattr(monte_carlo, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(monte_carlo.main))
            self.assertTrue(hasattr(monte_carlo, 'simulate'))
            self.assertTrue(hasattr(monte_carlo, 'get_combined_data'))
            
        except ImportError as e:
            self.fail(f"Failed to import Monte Carlo script: {e}")
    
    def test_bayesian_script_interface(self):
        """Test Bayesian script has standardized interface."""
        try:
            import ffbayes.analysis.bayesian_hierarchical_ff_modern as bayesian
            self.assertTrue(hasattr(bayesian, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(bayesian.main))
            self.assertTrue(hasattr(bayesian, 'bayesian_hierarchical_ff_modern'))
            
        except ImportError as e:
            self.fail(f"Failed to import Bayesian script: {e}")
    
    def test_team_aggregation_script_interface(self):
        """Test team aggregation script has standardized interface."""
        try:
            import ffbayes.analysis.bayesian_team_aggregation as team_agg
            self.assertTrue(hasattr(team_agg, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(team_agg.main))
            self.assertTrue(hasattr(team_agg, 'aggregate_individual_to_team_projections'))
            
        except ImportError as e:
            self.fail(f"Failed to import team aggregation script: {e}")
    
    def test_model_comparison_script_interface(self):
        """Test model comparison script has standardized interface."""
        try:
            import ffbayes.analysis.model_comparison_framework as model_compare
            self.assertTrue(hasattr(model_compare, 'ModelComparisonFramework'))
            
            # Test that script can be imported and has expected classes
            self.assertTrue(hasattr(model_compare.ModelComparisonFramework, 'load_monte_carlo_results'))
            
        except ImportError as e:
            self.fail(f"Failed to import model comparison script: {e}")
    
    def test_visualization_script_interface(self):
        """Test visualization script has standardized interface."""
        try:
            import ffbayes.analysis.create_team_aggregation_visualizations as viz
            self.assertTrue(hasattr(viz, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(viz.main))
            self.assertTrue(hasattr(viz, 'create_team_score_breakdown_chart'))
            
        except ImportError as e:
            self.fail(f"Failed to import visualization script: {e}")
    
    def test_pipeline_script_interface(self):
        """Test main pipeline script has standardized interface."""
        try:
            import ffbayes.run_pipeline as run_pipeline
            self.assertTrue(hasattr(run_pipeline, 'main'))
            
            # Test that script can be imported and has expected functions
            self.assertTrue(callable(run_pipeline.main))
            self.assertTrue(hasattr(run_pipeline, 'create_required_directories'))
            
        except ImportError as e:
            self.fail(f"Failed to import pipeline script: {e}")
    
    def test_console_script_availability(self):
        """Test that all console scripts are properly configured."""
        import ffbayes.utils.interface_standards as interface_standards

        # Test that interface standards can be imported
        self.assertIsNotNone(interface_standards)
        
        # Test that all expected utility functions are available
        expected_functions = [
            'setup_logger',
            'get_env_bool', 
            'get_env_int',
            'get_standard_paths',
            'handle_exception'
        ]
        
        for func_name in expected_functions:
            self.assertTrue(hasattr(interface_standards, func_name))
            self.assertTrue(callable(getattr(interface_standards, func_name)))
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across scripts."""
        # Test that handle_exception function works correctly
        test_exception = ValueError("Test error message")
        error_msg = handle_exception(test_exception, "Test operation")
        
        self.assertIn("Test operation", error_msg)
        self.assertIn("Test error message", error_msg)
        self.assertIn("ValueError", error_msg)
    
    def test_logging_consistency(self):
        """Test that logging setup is consistent."""
        # Test logger creation
        logger = setup_logger('test_logger')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')
    
    def test_environment_variable_handling(self):
        """Test environment variable handling consistency."""
        # Test boolean environment variables
        with patch.dict(os.environ, {'QUICK_TEST': 'true'}):
            self.assertTrue(get_env_bool('QUICK_TEST'))
        
        with patch.dict(os.environ, {'QUICK_TEST': 'false'}):
            self.assertFalse(get_env_bool('QUICK_TEST'))
        
        # Test integer environment variables
        with patch.dict(os.environ, {'MAX_CORES': '4'}):
            self.assertEqual(get_env_int('MAX_CORES', default=0), 4)
        
        with patch.dict(os.environ, {'DRAWS': '1000'}):
            self.assertEqual(get_env_int('DRAWS', default=0), 1000)


if __name__ == '__main__':
    unittest.main()
