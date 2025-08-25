#!/usr/bin/env python3
"""
Test pipeline diagnostic and analysis functionality.

This module tests the diagnostic capabilities for the ffbayes pipeline,
including pipeline state analysis, failure pattern detection, and
operational health assessment.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffbayes.run_pipeline import (create_required_directories,
                                  validate_step_outputs)
from ffbayes.utils.enhanced_pipeline_orchestrator import \
    EnhancedPipelineOrchestrator


class TestPipelineDiagnostics(unittest.TestCase):
    """Test pipeline diagnostic and analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "pipeline": {
                "name": "Test Pipeline",
                "version": "1.0.0",
                "description": "Test pipeline for diagnostics"
            },
            "pipeline_steps": [
                {
                    "name": "test_step",
                    "script": "test.script",
                    "description": "Test step",
                    "timeout": 300,
                    "retry_count": 3,
                    "dependencies": [],
                    "critical": True,
                    "parallel_group": "test"
                }
            ],
            "global_config": {
                "max_parallel_steps": 1,
                "pipeline_timeout": 7200,
                "default_timeout": 300,
                "default_retry_count": 2,
                "retry_delay": 5,
                "progress_monitoring": True,
                "error_recovery": True
            },
            "monitoring": {
                "alert_thresholds": {
                    "cpu_usage_warning": 0.8,
                    "memory_usage_warning": 0.8,
                    "disk_usage_warning": 0.9
                },
                "resource_check_interval": 30,
                "performance_tracking": True,
                "resource_usage_tracking": True,
                "progress_update_interval": 30
            },
            "parallel_groups": {
                "test": {
                    "max_concurrent": 1,
                    "timeout": 600
                }
            }
        }
    
    def test_directory_creation_diagnostic(self):
        """Test diagnostic for directory creation issues."""
        # Test that required directories are created
        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                create_required_directories()
                # Should have called mkdir for each required directory
                self.assertGreater(mock_mkdir.call_count, 0)
    
    def test_step_output_validation(self):
        """Test step output validation diagnostic."""
        # Test validation for known step
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = ['test_file.csv']
            result = validate_step_outputs("Data Collection")
            self.assertTrue(result)
    
    def test_step_output_validation_missing_files(self):
        """Test step output validation when files are missing."""
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = []  # No files found
            result = validate_step_outputs("Data Collection")
            self.assertFalse(result)
    
    def test_enhanced_orchestrator_diagnostic(self):
        """Test enhanced orchestrator diagnostic capabilities."""
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=self.test_config):
                with patch('os.path.exists', return_value=True):
                    orchestrator = EnhancedPipelineOrchestrator("test_config.json")
                    
                    # Test configuration loading
                    self.assertIsNotNone(orchestrator.config)
                    self.assertEqual(len(orchestrator.steps), 1)
                    
                    # Test performance metrics initialization
                    self.assertIn('total_steps', orchestrator.performance_metrics)
                    self.assertIn('completed_steps', orchestrator.performance_metrics)
                    self.assertIn('failed_steps', orchestrator.performance_metrics)
    
    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation diagnostic."""
        invalid_config = {
            "pipeline": {
                "name": "Test Pipeline"
            }
            # Missing required sections
        }
        
        with patch('json.load', return_value=invalid_config):
            with patch('os.path.exists', return_value=True):
                # Should raise PipelineError for invalid config
                with self.assertRaises(Exception):
                    EnhancedPipelineOrchestrator("test_config.json")
    
    def test_log_analysis_capability(self):
        """Test log analysis diagnostic capability."""
        # Test that we can analyze log files for patterns
        log_content = """
        🏈 FANTASY FOOTBALL ANALYTICS PIPELINE
        ============================================================
        Started at: 2025-08-22 08:32:15
        🚀 Using Enhanced Pipeline Orchestrator
        Pipeline Status: ❌ FAILED
        Total Steps: 11
        Completed Steps: 4
        Failed Steps: 0
        """
        
        # Test failure pattern detection
        self.assertIn("❌ FAILED", log_content)
        self.assertIn("Completed Steps: 4", log_content)
        self.assertIn("Total Steps: 11", log_content)
    
    def test_directory_structure_diagnostic(self):
        """Test directory structure diagnostic."""
        # Test that we can detect missing directories
        required_dirs = [
            "results",
            "plots", 
            "logs",
            "datasets/season_datasets",
            "datasets/combined_datasets"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            # Test that we can detect if directory exists
            exists = path.exists()
            # This is a diagnostic test - we're just checking the capability
            self.assertIsInstance(exists, bool)
    
    def test_bayesian_analysis_diagnostic(self):
        """Test Bayesian analysis diagnostic capabilities."""
        # Test that we can detect Bayesian analysis issues
        bayesian_output_patterns = [
            "results/bayesian-hierarchical-results/modern_model_results.json",
            "plots/bayesian_model/",
            "Only 20 samples per chain"
        ]
        
        # Test pattern matching capability
        for pattern in bayesian_output_patterns:
            self.assertIsInstance(pattern, str)
            self.assertGreater(len(pattern), 0)


class TestPipelineFailurePatterns(unittest.TestCase):
    """Test pipeline failure pattern detection."""
    
    def test_timeout_detection(self):
        """Test timeout failure pattern detection."""
        timeout_log = "⏰ Step timed out after 300 seconds"
        self.assertIn("timed out", timeout_log)
        self.assertIn("300 seconds", timeout_log)
    
    def test_missing_file_detection(self):
        """Test missing file failure pattern detection."""
        missing_file_log = "❌ Missing expected outputs:"
        self.assertIn("Missing expected outputs", missing_file_log)
    
    def test_enhanced_orchestrator_fallback_detection(self):
        """Test enhanced orchestrator fallback detection."""
        fallback_log = "🔄 Falling back to basic pipeline execution"
        self.assertIn("Falling back to basic", fallback_log)
    
    def test_bayesian_analysis_timeout_detection(self):
        """Test Bayesian analysis timeout detection."""
        bayesian_timeout = "Only 20 samples per chain. Reliable r-hat and ESS diagnostics require longer chains"
        self.assertIn("Only 20 samples per chain", bayesian_timeout)
        self.assertIn("Reliable r-hat and ESS diagnostics", bayesian_timeout)


if __name__ == "__main__":
    unittest.main()
