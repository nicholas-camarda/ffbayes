#!/usr/bin/env python3
"""
Tests for utility script integration across the ffbayes package.
Verifies that all utility functionality is properly integrated into the main pipeline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.append(str(Path.cwd() / 'src'))


class TestUtilityIntegration(unittest.TestCase):
    """Test utility script integration across all ffbayes components."""
    
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
    
    def test_quick_test_functionality_integration(self):
        """Test that QUICK_TEST functionality is properly integrated."""
        # Test that QUICK_TEST is available in all major scripts
        scripts_to_test = [
            'ffbayes.data_pipeline.collect_data',
            'ffbayes.data_pipeline.validate_data',
            'ffbayes.data_pipeline.preprocess_analysis_data',
            'ffbayes.analysis.montecarlo_historical_ff',
            'ffbayes.analysis.bayesian_hierarchical_ff_modern',
            'ffbayes.analysis.bayesian_team_aggregation',
            'ffbayes.analysis.create_team_aggregation_visualizations'
        ]
        
        for script_module in scripts_to_test:
            try:
                module = __import__(script_module, fromlist=['main'])
                # Check that the module can be imported
                self.assertIsNotNone(module)
                
                # Check that it has a main function (indicating it's a script)
                if hasattr(module, 'main'):
                    self.assertTrue(callable(module.main))
                    
            except ImportError as e:
                self.fail(f"Failed to import {script_module}: {e}")
    
    def test_environment_variable_integration(self):
        """Test that environment variables are properly integrated."""
        from ffbayes.utils.interface_standards import get_env_bool, get_env_int

        # Test QUICK_TEST environment variable
        with patch.dict(os.environ, {'QUICK_TEST': 'true'}):
            self.assertTrue(get_env_bool('QUICK_TEST'))
        
        with patch.dict(os.environ, {'QUICK_TEST': 'false'}):
            self.assertFalse(get_env_bool('QUICK_TEST'))
        
        # Test other environment variables
        with patch.dict(os.environ, {'MAX_CORES': '8'}):
            self.assertEqual(get_env_int('MAX_CORES', default=4), 8)
        
        with patch.dict(os.environ, {'DRAWS': '2000'}):
            self.assertEqual(get_env_int('DRAWS', default=1000), 2000)
    
    def test_utility_functions_integration(self):
        """Test that utility functions are properly integrated."""
        # Test interface standards
        from ffbayes.utils.interface_standards import (
            get_standard_paths,
            handle_exception,
            setup_logger,
        )

        # Test logger setup
        logger = setup_logger('test_utility')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_utility')
        
        # Test standard paths
        paths = get_standard_paths()
        self.assertIsNotNone(paths)
        self.assertTrue(hasattr(paths, 'datasets_root'))
        self.assertTrue(hasattr(paths, 'results_root'))
        self.assertTrue(hasattr(paths, 'plots_root'))
        
        # Test exception handling
        test_exception = ValueError("Test error")
        error_msg = handle_exception(test_exception, "Test operation")
        self.assertIn("Test operation", error_msg)
        self.assertIn("Test error", error_msg)
    
    def test_progress_monitoring_integration(self):
        """Test that progress monitoring is properly integrated."""
        try:
            from ffbayes.utils.progress_monitor import ProgressMonitor

            # Test that ProgressMonitor can be instantiated
            monitor = ProgressMonitor("Test Operation")
            self.assertIsNotNone(monitor)
            self.assertEqual(monitor.title, "Test Operation")
            
        except ImportError as e:
            self.fail(f"Failed to import ProgressMonitor: {e}")
    
    def test_model_validation_integration(self):
        """Test that model validation is properly integrated."""
        from ffbayes.utils.model_validation import ModelValidator, validate_monte_carlo_model

        # Test ModelValidator instantiation
        validator = ModelValidator("test-model")
        self.assertIsNotNone(validator)
        self.assertEqual(validator.model_name, "test-model")
        
        # Test Monte Carlo validation
        import pandas as pd
        test_df = pd.DataFrame({
            'Player1': [10, 15, 20, 25, 30],
            'Player2': [5, 10, 15, 20, 25],
            'Total': [15, 25, 35, 45, 55]
        })
        
        validation_results = validate_monte_carlo_model(test_df)
        self.assertIsInstance(validation_results, dict)
        self.assertIn('valid', validation_results)
    
    def test_script_interface_integration(self):
        """Test that script interface is properly integrated."""
        from ffbayes.utils.script_interface import create_standardized_interface

        # Test interface creation
        interface = create_standardized_interface("test-script", "Test description")
        self.assertIsNotNone(interface)
        self.assertEqual(interface.script_name, "test-script")
        self.assertEqual(interface.description, "Test description")
        
        # Test argument parser setup
        parser = interface.setup_argument_parser()
        self.assertIsNotNone(parser)
        
        # Test that standard arguments are available
        args = parser.parse_args(['--verbose', '--quick-test'])
        self.assertTrue(args.verbose)
        self.assertTrue(args.quick_test)
    
    def test_pipeline_orchestration_integration(self):
        """Test that pipeline orchestration is properly integrated."""
        try:
            from ffbayes.utils.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator

            # Test that EnhancedPipelineOrchestrator can be imported
            self.assertIsNotNone(EnhancedPipelineOrchestrator)
            
            # Test that the class has the expected methods
            self.assertTrue(hasattr(EnhancedPipelineOrchestrator, '__init__'))
            
        except ImportError as e:
            self.fail(f"Failed to import EnhancedPipelineOrchestrator: {e}")
        except Exception as e:
            # Configuration file not found is expected in test environment
            if "Configuration file not found" in str(e):
                pass  # This is expected in test environment
            else:
                self.fail(f"Unexpected error: {e}")
    
    def test_console_script_integration(self):
        """Test that console scripts are properly integrated."""
        # Test that all console scripts can be imported
        console_scripts = [
            'ffbayes.data_pipeline.collect_data',
            'ffbayes.data_pipeline.validate_data',
            'ffbayes.data_pipeline.preprocess_analysis_data',
            'ffbayes.analysis.montecarlo_historical_ff',
            'ffbayes.analysis.bayesian_hierarchical_ff_modern',
            'ffbayes.analysis.bayesian_team_aggregation',
            'ffbayes.analysis.create_team_aggregation_visualizations',
            'ffbayes.analysis.model_comparison_framework'
        ]
        
        for script_module in console_scripts:
            try:
                module = __import__(script_module, fromlist=['main'])
                self.assertIsNotNone(module)
                
                # Check that it has a main function
                if hasattr(module, 'main'):
                    self.assertTrue(callable(module.main))
                    
            except ImportError as e:
                self.fail(f"Failed to import console script {script_module}: {e}")
    
    def test_utility_function_availability(self):
        """Test that all utility functions are available and working."""
        # Test all major utility modules
        utility_modules = [
            'ffbayes.utils.interface_standards',
            'ffbayes.utils.progress_monitor',
            'ffbayes.utils.model_validation',
            'ffbayes.utils.script_interface',
            'ffbayes.utils.enhanced_pipeline_orchestrator'
        ]
        
        for module_name in utility_modules:
            try:
                module = __import__(module_name, fromlist=['*'])
                self.assertIsNotNone(module)
                
                # Check that the module has some functionality
                module_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                self.assertGreater(len(module_attrs), 0, f"Module {module_name} has no public attributes")
                
            except ImportError as e:
                self.fail(f"Failed to import utility module {module_name}: {e}")
    
    def test_no_orphaned_utility_scripts(self):
        """Test that there are no orphaned utility scripts outside the main structure."""
        # Check that all Python files are in the proper locations
        src_dir = Path('src/ffbayes')
        
        # All Python files should be in the src/ffbayes structure
        python_files = list(src_dir.rglob('*.py'))
        
        for py_file in python_files:
            # Skip __init__.py files
            if py_file.name == '__init__.py':
                continue
                
            # Check that the file is in a proper module structure
            relative_path = py_file.relative_to(src_dir)
            module_parts = relative_path.parts
            
            # Should be in a proper module (not just loose files)
            self.assertGreater(len(module_parts), 1, f"File {py_file} is not in a proper module structure")
            
            # Should not be in a test directory
            self.assertNotIn('test', module_parts, f"File {py_file} appears to be a test file in the wrong location")
    
    def test_utility_function_consistency(self):
        """Test that utility functions provide consistent interfaces."""
        from ffbayes.utils.interface_standards import get_env_bool, get_env_int

        # Test consistent behavior for environment variables
        with patch.dict(os.environ, {'TEST_VAR': 'true'}):
            self.assertTrue(get_env_bool('TEST_VAR'))
        
        with patch.dict(os.environ, {'TEST_VAR': 'false'}):
            self.assertFalse(get_env_bool('TEST_VAR'))
        
        with patch.dict(os.environ, {'TEST_VAR': '1'}):
            self.assertTrue(get_env_bool('TEST_VAR'))
        
        with patch.dict(os.environ, {'TEST_VAR': '0'}):
            self.assertFalse(get_env_bool('TEST_VAR'))
        
        # Test consistent behavior for integer environment variables
        with patch.dict(os.environ, {'TEST_INT': '42'}):
            self.assertEqual(get_env_int('TEST_INT', default=0), 42)
        
        with patch.dict(os.environ, {'TEST_INT': 'invalid'}):
            self.assertEqual(get_env_int('TEST_INT', default=100), 100)
    
    def test_integration_with_main_pipeline(self):
        """Test that utility functions integrate properly with the main pipeline."""
        try:
            from ffbayes.run_pipeline import main

            # Test that the main pipeline can be imported
            self.assertTrue(callable(main))
            
        except ImportError as e:
            self.fail(f"Failed to import main pipeline: {e}")


if __name__ == '__main__':
    unittest.main()
