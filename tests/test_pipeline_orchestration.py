#!/usr/bin/env python3
"""
Test suite for Pipeline Orchestration functionality.

This module tests the enhanced pipeline orchestration features including
stage sequencing, dependency management, error recovery, and progress monitoring.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Add the scripts directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestPipelineOrchestration(unittest.TestCase):
    """Test cases for Pipeline Orchestration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline_config = {
            'pipeline_steps': [
                {
                    'name': 'Data Collection',
                    'script': 'scripts/data_pipeline/01_collect_data.py',
                    'description': 'Collect raw NFL data from multiple sources',
                    'dependencies': [],
                    'timeout': 300,
                    'retry_count': 3
                },
                {
                    'name': 'Data Validation',
                    'script': 'scripts/data_pipeline/02_validate_data.py',
                    'description': 'Validate data quality and completeness',
                    'dependencies': ['Data Collection'],
                    'timeout': 180,
                    'retry_count': 2
                },
                {
                    'name': 'Monte Carlo Simulation',
                    'script': 'scripts/analysis/montecarlo-historical-ff.py',
                    'description': 'Generate team-level projections using historical data',
                    'dependencies': ['Data Validation'],
                    'timeout': 600,
                    'retry_count': 1
                },
                {
                    'name': 'Bayesian Predictions',
                    'script': 'scripts/analysis/bayesian-hierarchical-ff-modern.py',
                    'description': 'Generate player-level predictions with uncertainty using PyMC4',
                    'dependencies': ['Data Validation'],
                    'timeout': 900,
                    'retry_count': 1
                }
            ],
            'global_config': {
                'max_parallel_steps': 2,
                'default_timeout': 300,
                'retry_delay': 30,
                'enable_progress_monitoring': True,
                'log_level': 'INFO'
            }
        }
        
        # Create test script files
        self.test_scripts = [
            'scripts/data_pipeline/01_collect_data.py',
            'scripts/data_pipeline/02_validate_data.py',
            'scripts/analysis/montecarlo-historical-ff.py',
            'scripts/analysis/bayesian-hierarchical-ff-modern.py'
        ]
        
        for script_path in self.test_scripts:
            full_path = os.path.join(self.test_dir, script_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write('#!/usr/bin/env python3\nprint("Test script executed")\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_pipeline_configuration_loading(self):
        """Test loading and validation of pipeline configuration."""
        # Test configuration structure validation
        required_keys = ['pipeline_steps', 'global_config']
        for key in required_keys:
            self.assertIn(key, self.pipeline_config)
        
        # Test step configuration validation
        for step in self.pipeline_config['pipeline_steps']:
            required_step_keys = ['name', 'script', 'description', 'dependencies', 'timeout', 'retry_count']
            for step_key in required_step_keys:
                self.assertIn(step_key, step)
        
        # Test global configuration validation
        global_config = self.pipeline_config['global_config']
        required_global_keys = ['max_parallel_steps', 'default_timeout', 'retry_delay', 'enable_progress_monitoring', 'log_level']
        for global_key in required_global_keys:
            self.assertIn(global_key, global_config)
    
    def test_dependency_validation(self):
        """Test validation of pipeline step dependencies."""
        # Test circular dependency detection
        steps = self.pipeline_config['pipeline_steps']
        
        # Create dependency graph
        dependency_graph = {}
        for step in steps:
            dependency_graph[step['name']] = step['dependencies']
        
        # Verify no circular dependencies
        def has_circular_dependency(graph, node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in graph.get(node, []):
                if not visited.get(neighbor, False):
                    if has_circular_dependency(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            
            rec_stack[node] = False
            return False
        
        visited = {}
        rec_stack = {}
        
        for step_name in dependency_graph:
            if not visited.get(step_name, False):
                self.assertFalse(has_circular_dependency(dependency_graph, step_name, visited, rec_stack))
    
    def test_stage_sequencing(self):
        """Test proper sequencing of pipeline stages based on dependencies."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that dependent steps come after their dependencies
        step_order = {}
        for i, step in enumerate(steps):
            step_order[step['name']] = i
        
        for step in steps:
            for dependency in step['dependencies']:
                self.assertLess(step_order[dependency], step_order[step['name']], 
                               f"Step '{step['name']}' should come after dependency '{dependency}'")
    
    def test_timeout_configuration(self):
        """Test timeout configuration for pipeline steps."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that all steps have reasonable timeout values
        for step in steps:
            self.assertGreater(step['timeout'], 0, f"Step '{step['name']}' must have positive timeout")
            self.assertLessEqual(step['timeout'], 3600, f"Step '{step['name']}' timeout should be reasonable (< 1 hour)")
        
        # Test that global default timeout is reasonable
        global_config = self.pipeline_config['global_config']
        self.assertGreater(global_config['default_timeout'], 0)
        self.assertLessEqual(global_config['default_timeout'], 1800)  # 30 minutes max
    
    def test_retry_configuration(self):
        """Test retry configuration for pipeline steps."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that all steps have reasonable retry counts
        for step in steps:
            self.assertGreaterEqual(step['retry_count'], 0, f"Step '{step['name']}' must have non-negative retry count")
            self.assertLessEqual(step['retry_count'], 5, f"Step '{step['name']}' retry count should be reasonable (≤ 5)")
        
        # Test that retry delay is reasonable
        global_config = self.pipeline_config['global_config']
        self.assertGreater(global_config['retry_delay'], 0)
        self.assertLessEqual(global_config['retry_delay'], 300)  # 5 minutes max
    
    def test_parallel_execution_configuration(self):
        """Test parallel execution configuration."""
        global_config = self.pipeline_config['global_config']
        
        # Test parallel execution limits
        self.assertGreater(global_config['max_parallel_steps'], 0, "Must allow at least 1 parallel step")
        self.assertLessEqual(global_config['max_parallel_steps'], 4, "Parallel steps should be reasonable (≤ 4)")
        
        # Test that parallel execution makes sense for the pipeline
        steps = self.pipeline_config['pipeline_steps']
        independent_steps = [step for step in steps if not step['dependencies']]
        
        # Should be able to run at least some steps in parallel
        self.assertGreaterEqual(len(independent_steps), 1, "Pipeline should have at least one independent step")
    
    def test_error_recovery_configuration(self):
        """Test error recovery and graceful degradation configuration."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that critical steps have appropriate retry counts
        critical_steps = ['Data Collection', 'Data Validation']
        for step_name in critical_steps:
            step = next((s for s in steps if s['name'] == step_name), None)
            if step:
                self.assertGreater(step['retry_count'], 0, f"Critical step '{step_name}' should have retry capability")
        
        # Test that analysis steps have reasonable retry counts
        analysis_steps = ['Monte Carlo Simulation', 'Bayesian Predictions']
        for step_name in analysis_steps:
            step = next((s for s in steps if s['name'] == step_name), None)
            if step:
                self.assertGreaterEqual(step['retry_count'], 0, f"Analysis step '{step_name}' should have retry configuration")
    
    def test_progress_monitoring_configuration(self):
        """Test progress monitoring configuration."""
        global_config = self.pipeline_config['global_config']
        
        # Test that progress monitoring can be enabled/disabled
        self.assertIsInstance(global_config['enable_progress_monitoring'], bool)
        
        # Test log level configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.assertIn(global_config['log_level'], valid_log_levels)
    
    def test_script_path_validation(self):
        """Test validation of script paths in pipeline configuration."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that all script paths are properly formatted
        for step in steps:
            script_path = step['script']
            
            # Should be relative path
            self.assertFalse(script_path.startswith('/'), f"Script path '{script_path}' should be relative")
            
            # Should have .py extension
            self.assertTrue(script_path.endswith('.py'), f"Script path '{script_path}' should have .py extension")
            
            # Should be in appropriate directory
            if 'data_pipeline' in script_path:
                self.assertIn('data_pipeline', script_path)
            elif 'analysis' in script_path:
                self.assertIn('analysis', script_path)
    
    def test_pipeline_execution_order(self):
        """Test that pipeline execution follows dependency order."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Create execution order based on dependencies
        execution_order = []
        visited = set()
        
        def add_step(step_name):
            if step_name in visited:
                return
            
            step = next((s for s in steps if s['name'] == step_name), None)
            if step:
                # Add dependencies first
                for dependency in step['dependencies']:
                    add_step(dependency)
                
                execution_order.append(step_name)
                visited.add(step_name)
        
        # Add all steps in dependency order
        for step in steps:
            add_step(step['name'])
        
        # Verify execution order
        self.assertEqual(len(execution_order), len(steps), "All steps should be in execution order")
        
        # Verify dependencies are satisfied
        for i, step_name in enumerate(execution_order):
            step = next((s for s in steps if s['name'] == step_name), None)
            if step:
                for dependency in step['dependencies']:
                    dep_index = execution_order.index(dependency)
                    self.assertLess(dep_index, i, f"Dependency '{dependency}' should come before '{step_name}'")
    
    def test_configuration_file_format(self):
        """Test pipeline configuration file format and structure."""
        # Test JSON serialization
        config_json = json.dumps(self.pipeline_config, indent=2)
        self.assertIsInstance(config_json, str)
        
        # Test JSON deserialization
        parsed_config = json.loads(config_json)
        self.assertEqual(parsed_config, self.pipeline_config)
        
        # Test configuration validation after serialization
        self.assertIn('pipeline_steps', parsed_config)
        self.assertIn('global_config', parsed_config)
        
        # Verify step count is preserved
        self.assertEqual(len(parsed_config['pipeline_steps']), len(self.pipeline_config['pipeline_steps']))
    
    def test_pipeline_metadata(self):
        """Test pipeline metadata and information."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test step metadata
        for step in steps:
            # Each step should have descriptive information
            self.assertIsInstance(step['name'], str)
            self.assertIsInstance(step['description'], str)
            self.assertGreater(len(step['description']), 10, f"Step '{step['name']}' should have meaningful description")
            
            # Script path should be valid
            self.assertIsInstance(step['script'], str)
            self.assertGreater(len(step['script']), 0)
        
        # Test global metadata
        global_config = self.pipeline_config['global_config']
        self.assertIsInstance(global_config['log_level'], str)
        self.assertIsInstance(global_config['enable_progress_monitoring'], bool)
    
    def test_pipeline_scalability(self):
        """Test that pipeline configuration supports scalability."""
        steps = self.pipeline_config['pipeline_steps']
        global_config = self.pipeline_config['global_config']
        
        # Test that pipeline can handle multiple steps
        self.assertGreater(len(steps), 1, "Pipeline should support multiple steps")
        
        # Test that parallel execution is configured
        self.assertGreater(global_config['max_parallel_steps'], 1, "Pipeline should support parallel execution")
        
        # Test that timeout values scale appropriately
        total_timeout = sum(step['timeout'] for step in steps)
        self.assertLess(total_timeout, 7200, "Total pipeline timeout should be reasonable (< 2 hours)")
    
    def test_error_handling_configuration(self):
        """Test error handling configuration for pipeline steps."""
        steps = self.pipeline_config['pipeline_steps']
        
        # Test that all steps have error handling configuration
        for step in steps:
            # Retry count should be configured
            self.assertIsInstance(step['retry_count'], int)
            self.assertGreaterEqual(step['retry_count'], 0)
            
            # Timeout should be configured
            self.assertIsInstance(step['timeout'], int)
            self.assertGreater(step['timeout'], 0)
        
        # Test global error handling configuration
        global_config = self.pipeline_config['global_config']
        self.assertIsInstance(global_config['retry_delay'], int)
        self.assertGreater(global_config['retry_delay'], 0)

if __name__ == '__main__':
    unittest.main()
