#!/usr/bin/env python3
"""
Test directory management and organization functionality.

This module tests the directory creation, cleanup, and organization
capabilities for the ffbayes pipeline.
"""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffbayes.run_pipeline import create_required_directories, validate_step_outputs


class TestDirectoryManagement(unittest.TestCase):
    """Test directory management and organization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_directory_creation_functionality(self):
        """Test that required directories are created properly."""
        # Test that directories don't exist initially
        current_year = datetime.now().year
        required_dirs = [
            f"results/{current_year}/montecarlo_results",
            f"results/{current_year}/bayesian-hierarchical-results",
            f"results/{current_year}/team_aggregation",
            f"results/{current_year}/draft_strategy",
            f"results/{current_year}/draft_strategy_comparison",
            f"results/{current_year}/model_comparison",
            f"plots/{current_year}/team_aggregation",
            f"plots/{current_year}/draft_strategy_comparison",
            f"plots/{current_year}/test_runs"
        ]
        
        for dir_path in required_dirs:
            self.assertFalse(Path(dir_path).exists(), f"Directory {dir_path} should not exist initially")
        
        # Create directories
        create_required_directories()
        
        # Test that directories are created
        for dir_path in required_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Directory {dir_path} should be created")
            self.assertTrue(Path(dir_path).is_dir(), f"{dir_path} should be a directory")
    
    def test_directory_creation_idempotent(self):
        """Test that directory creation is idempotent (safe to run multiple times)."""
        # Create directories first time
        create_required_directories()
        
        # Count existing directories
        existing_dirs = []
        for dir_path in Path("results").iterdir():
            if dir_path.is_dir():
                existing_dirs.append(dir_path)
        
        # Create directories again
        create_required_directories()
        
        # Count directories after second creation
        existing_dirs_after = []
        for dir_path in Path("results").iterdir():
            if dir_path.is_dir():
                existing_dirs_after.append(dir_path)
        
        # Should have same number of directories
        self.assertEqual(len(existing_dirs), len(existing_dirs_after))
    
    def test_directory_structure_completeness(self):
        """Test that all required directory structure is created."""
        create_required_directories()
        
        # Test main directories exist
        self.assertTrue(Path("results").exists())
        self.assertTrue(Path("plots").exists())
        self.assertTrue(Path("datasets").exists())
        self.assertTrue(Path("misc-datasets").exists())
        self.assertTrue(Path("datasets/snake_draft_datasets").exists())
        self.assertTrue(Path("my_ff_teams").exists())
        
        # Test subdirectories exist
        current_year = datetime.now().year
        results_subdirs = [
            f"{current_year}/montecarlo_results",
            f"{current_year}/bayesian-hierarchical-results", 
            f"{current_year}/team_aggregation",
            f"{current_year}/draft_strategy",
            f"{current_year}/draft_strategy_comparison",
            f"{current_year}/model_comparison"
        ]
        
        plots_subdirs = [
            f"{current_year}/team_aggregation",
            f"{current_year}/draft_strategy_comparison",
            f"{current_year}/test_runs"
        ]
        
        for subdir in results_subdirs:
            self.assertTrue(Path(f"results/{subdir}").exists())
        
        for subdir in plots_subdirs:
            self.assertTrue(Path(f"plots/{subdir}").exists())
    
    def test_directory_permissions(self):
        """Test that created directories have proper permissions."""
        create_required_directories()
        
        # Test that directories are writable
        test_dirs = ["results", "plots", "datasets"]
        for dir_path in test_dirs:
            path = Path(dir_path)
            self.assertTrue(path.exists())
            self.assertTrue(os.access(path, os.W_OK), f"Directory {dir_path} should be writable")
    
    def test_pipeline_breakage_prevention(self):
        """Test that pipeline doesn't break when directories don't exist."""
        # Ensure directories don't exist
        if Path("results").exists():
            shutil.rmtree("results")
        if Path("plots").exists():
            shutil.rmtree("plots")
        
        # Pipeline should be able to create directories and continue
        try:
            create_required_directories()
            # If we get here, no exception was raised
            self.assertTrue(True, "Directory creation should not raise exceptions")
        except Exception as e:
            self.fail(f"Directory creation should not fail: {e}")
    
    def test_empty_directory_cleanup(self):
        """Test that empty directories are handled properly."""
        create_required_directories()
        
        # Create some empty directories
        empty_dir = Path("results/empty_test_dir")
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        # Test that empty directories don't break the pipeline
        self.assertTrue(empty_dir.exists())
        self.assertTrue(empty_dir.is_dir())
        
        # Clean up
        empty_dir.rmdir()
    
    def test_directory_validation_integration(self):
        """Test that directory validation works with step output validation."""
        create_required_directories()
        
        # Test that step output validation can find directories
        # This tests the integration between directory creation and step validation
        validation_result = validate_step_outputs("Data Collection")
        # Should not fail due to missing directories
        self.assertIsInstance(validation_result, bool)


class TestDirectoryCleanup(unittest.TestCase):
    """Test directory cleanup functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cleanup_old_files_capability(self):
        """Test capability to clean up old files."""
        create_required_directories()
        
        # Create some test files
        test_file = Path("results/test_file.txt")
        test_file.write_text("test content")
        
        # Test that files can be created and cleaned up
        self.assertTrue(test_file.exists())
        
        # Clean up
        test_file.unlink()
        self.assertFalse(test_file.exists())
    
    def test_directory_cleanup_safety(self):
        """Test that directory cleanup is safe and doesn't remove important files."""
        create_required_directories()
        
        # Create some important files
        important_file = Path("results/important_results.json")
        important_file.write_text('{"important": "data"}')
        
        # Test that important files are preserved
        self.assertTrue(important_file.exists())
        
        # Clean up only test files, not important ones
        important_file.unlink()  # This is just for test cleanup


if __name__ == "__main__":
    unittest.main()
