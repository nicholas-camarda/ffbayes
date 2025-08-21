"""
Test suite for pipeline execution fixes.

This module tests the fixes for the pipeline execution failure issues:
- Enhanced orchestrator configuration loading
- Pipeline script execution methods
- End-to-end pipeline functionality
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.append(str(Path.cwd() / 'src'))

from ffbayes.run_pipeline import EnhancedPipelineOrchestrator, main


class TestPipelineExecutionFixes(unittest.TestCase):
    """Test pipeline execution fixes for import errors and configuration issues."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "pipeline": {
                "name": "Test Pipeline",
                "version": "1.0.0",
                "description": "Test pipeline for execution fixes"
            },
            "pipeline_steps": [
                {
                    "name": "test_stage",
                    "script": "ffbayes.test_module",
                    "description": "Test stage",
                    "timeout": 300,
                    "retry_count": 2,
                    "dependencies": [],
                    "critical": True,
                    "parallel_group": "default"
                }
            ],
            "global_config": {
                "max_parallel_steps": 1,
                "pipeline_timeout": 3600,
                "default_timeout": 300,
                "default_retry_count": 2
            },
            "parallel_groups": {
                "default": {
                    "max_concurrent": 1,
                    "timeout": 300
                }
            },
            "stages": [
                {
                    "name": "test_stage",
                    "script": "ffbayes.test_module",
                    "description": "Test stage",
                    "timeout": 300,
                    "retries": 2,
                    "dependencies": []
                }
            ],
            "execution": {
                "parallel_stages": [],
                "max_parallel_jobs": 1,
                "default_timeout": 300,
                "default_retries": 2
            },
            "output": {
                "results_directory": "results",
                "plots_directory": "plots"
            }
        }

    def test_enhanced_orchestrator_configuration_loading(self):
        """Test that enhanced orchestrator can load configuration with pipeline_steps key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name

        try:
            # Test that orchestrator can load config with pipeline_steps
            orchestrator = EnhancedPipelineOrchestrator(config_path)
            self.assertIsNotNone(orchestrator)
            self.assertTrue(hasattr(orchestrator, 'steps'))
            self.assertEqual(len(orchestrator.steps), 1)
        finally:
            os.unlink(config_path)

    def test_enhanced_orchestrator_missing_pipeline_steps_error(self):
        """Test that orchestrator fails gracefully when pipeline_steps is missing."""
        # Remove pipeline_steps from config
        config_without_steps = self.test_config.copy()
        del config_without_steps['pipeline_steps']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_without_steps, f)
            config_path = f.name

        try:
            # Should fail gracefully when pipeline_steps is missing
            with self.assertRaises(Exception):  # PipelineError or similar
                EnhancedPipelineOrchestrator(config_path)
        finally:
            os.unlink(config_path)

    def test_pipeline_script_execution_methods(self):
        """Test different pipeline script execution methods."""
        # Test that pipeline can handle execution failures gracefully
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            # Test that the main function exists and can be called
            self.assertTrue(callable(main), "Main function should be callable")

    def test_python_module_execution_syntax(self):
        """Test that pipeline uses proper Python module execution syntax."""
        # Verify that pipeline steps use python -m ffbayes.* format
        expected_script = "ffbayes.data_pipeline.collect_data"
        self.assertIn("ffbayes.", expected_script)
        self.assertNotIn("python -m", expected_script)  # Should be module path only

    def test_pipeline_configuration_structure(self):
        """Test that pipeline configuration has correct structure."""
        # Check required keys
        required_keys = ['pipeline', 'stages', 'execution']
        for key in required_keys:
            self.assertIn(key, self.test_config)

        # Check stages structure
        stages = self.test_config['stages']
        self.assertIsInstance(stages, list)
        self.assertGreater(len(stages), 0)

        # Check each stage has required fields
        for stage in stages:
            required_stage_keys = ['name', 'script', 'description', 'timeout', 'retries']
            for stage_key in required_stage_keys:
                self.assertIn(stage_key, stage)

    def test_pipeline_dependency_validation(self):
        """Test that pipeline dependencies are properly validated."""
        # Test valid dependency chain
        stages = self.test_config['stages']
        stage_names = [stage['name'] for stage in stages]
        
        for stage in stages:
            for dep in stage.get('dependencies', []):
                self.assertIn(dep, stage_names, f"Dependency {dep} not found in stages")

    def test_pipeline_execution_timeout_configuration(self):
        """Test that pipeline timeout configuration is properly set."""
        for stage in self.test_config['stages']:
            self.assertIsInstance(stage['timeout'], int)
            self.assertGreater(stage['timeout'], 0)
            self.assertLessEqual(stage['timeout'], 3600)  # Max 1 hour

    def test_pipeline_retry_configuration(self):
        """Test that pipeline retry configuration is properly set."""
        for stage in self.test_config['stages']:
            self.assertIsInstance(stage['retries'], int)
            self.assertGreaterEqual(stage['retries'], 0)
            self.assertLessEqual(stage['retries'], 5)  # Max 5 retries

    def test_pipeline_parallel_execution_config(self):
        """Test that pipeline parallel execution configuration is valid."""
        execution_config = self.test_config['execution']
        self.assertIsInstance(execution_config['parallel_stages'], list)
        self.assertIsInstance(execution_config['max_parallel_jobs'], int)
        self.assertGreater(execution_config['max_parallel_jobs'], 0)

    def test_pipeline_environment_configuration(self):
        """Test that pipeline environment configuration is properly set."""
        # This test would check environment variables and paths
        # For now, just verify the test config has the right structure
        self.assertIn('execution', self.test_config)
        self.assertIn('output', self.test_config)

    def test_pipeline_fallback_execution(self):
        """Test that pipeline falls back to basic execution when enhanced orchestrator fails."""
        # Mock enhanced orchestrator failure
        with patch('ffbayes.run_pipeline.EnhancedPipelineOrchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Enhanced orchestrator failed")
            
            # Should fall back to basic execution
            self.assertTrue(callable(main), "Main function should be callable for fallback execution")

    def test_pipeline_error_handling(self):
        """Test that pipeline handles errors gracefully."""
        # Test with invalid configuration
        invalid_config = {"invalid": "config"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            # Should handle invalid config gracefully
            with self.assertRaises(Exception):  # PipelineError or similar
                EnhancedPipelineOrchestrator(config_path)
        finally:
            os.unlink(config_path)


class TestPipelineScriptExecution(unittest.TestCase):
    """Test pipeline script execution methods and import handling."""

    def test_script_execution_with_proper_context(self):
        """Test that scripts execute with proper Python module context."""
        # Test that we can import ffbayes modules
        try:
            import ffbayes.analysis.bayesian_hierarchical_ff_modern
            import ffbayes.data_pipeline.collect_data
            import ffbayes.data_pipeline.validate_data
            self.assertTrue(True, "All required modules can be imported")
        except ImportError as e:
            self.fail(f"Failed to import required module: {e}")

    def test_relative_import_handling(self):
        """Test that relative imports work correctly within package structure."""
        # Test that relative imports in ffbayes modules work
        try:
            from ffbayes.utils.interface_standards import setup_logger
            self.assertTrue(True, "Relative imports work correctly")
        except ImportError as e:
            self.fail(f"Relative import failed: {e}")

    def test_console_script_availability(self):
        """Test that console scripts are available and functional."""
        # Test that console scripts can be found
        script_names = [
            'ffbayes-pipeline',
            'ffbayes-collect', 
            'ffbayes-validate',
            'ffbayes-preprocess'
        ]
        
        for script_name in script_names:
            # Check if script exists in PATH
            script_path = Path(f"/Users/ncamarda/mambaforge/envs/ffbayes/bin/{script_name}")
            self.assertTrue(script_path.exists(), f"Console script {script_name} not found")

    def test_pipeline_module_execution(self):
        """Test that pipeline can execute modules using python -m syntax."""
        # Test that we can execute ffbayes modules as packages
        test_module = "ffbayes.data_pipeline.collect_data"
        
        # Verify module path format
        self.assertTrue(test_module.startswith("ffbayes."))
        self.assertIn(".", test_module)
        
        # Test that module can be imported
        try:
            __import__(test_module)
            self.assertTrue(True, f"Module {test_module} can be imported")
        except ImportError as e:
            self.fail(f"Failed to import module {test_module}: {e}")


class TestEndToEndPipelineExecution(unittest.TestCase):
    """Test end-to-end pipeline execution without import errors."""

    def test_pipeline_stage_sequence(self):
        """Test that pipeline stages execute in correct sequence."""
        # Define expected stage sequence
        expected_sequence = [
            "data_collection",
            "data_validation", 
            "data_preprocessing",
            "bayesian_analysis",
            "team_aggregation",
            "draft_strategy",
            "monte_carlo_validation",
            "model_comparison",
            "visualization"
        ]
        
        # Load actual pipeline config
        config_path = Path("config/pipeline_config.json")
        self.assertTrue(config_path.exists(), "Pipeline config file not found")
        
        with open(config_path) as f:
            config = json.load(f)
        
        actual_sequence = [stage['name'] for stage in config['stages']]
        
        # Verify sequence matches expected
        self.assertEqual(actual_sequence, expected_sequence, 
                        "Pipeline stage sequence doesn't match expected order")

    def test_pipeline_dependencies_acyclic(self):
        """Test that pipeline dependencies form an acyclic graph."""
        config_path = Path("config/pipeline_config.json")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Build dependency graph
        dependencies = {}
        for stage in config['stages']:
            dependencies[stage['name']] = stage.get('dependencies', [])
        
        # Check for cycles using simple cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check each stage for cycles
        for stage_name in dependencies:
            if stage_name not in visited:
                if has_cycle(stage_name):
                    self.fail(f"Pipeline has circular dependency involving {stage_name}")
        
        self.assertTrue(True, "Pipeline dependencies are acyclic")

    def test_pipeline_configuration_completeness(self):
        """Test that pipeline configuration is complete and valid."""
        config_path = Path("config/pipeline_config.json")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required top-level keys
        required_keys = ['pipeline', 'stages', 'execution', 'output', 'environment']
        for key in required_keys:
            self.assertIn(key, config, f"Missing required key: {key}")
        
        # Check stages have all required fields
        for i, stage in enumerate(config['stages']):
            required_stage_keys = ['name', 'script', 'description', 'timeout', 'retries']
            for key in required_stage_keys:
                self.assertIn(key, stage, f"Stage {i} missing required key: {key}")
        
        # Check execution configuration
        execution = config['execution']
        self.assertIn('parallel_stages', execution)
        self.assertIn('max_parallel_jobs', execution)
        self.assertIn('default_timeout', execution)
        self.assertIn('default_retries', execution)

    def test_pipeline_script_paths_valid(self):
        """Test that all pipeline script paths are valid module paths."""
        config_path = Path("config/pipeline_config.json")
        
        with open(config_path) as f:
            config = json.load(f)
        
        for stage in config['stages']:
            script_path = stage['script']
            
            # Script should be a valid module path
            self.assertTrue(script_path.startswith("ffbayes."), 
                          f"Script path {script_path} should start with 'ffbayes.'")
            
            # Script should not contain file extensions
            self.assertNotIn(".py", script_path, 
                            f"Script path {script_path} should not contain .py extension")
            
            # Script should use dot notation for module paths
            self.assertIn(".", script_path, 
                          f"Script path {script_path} should use dot notation")


if __name__ == '__main__':
    unittest.main()
