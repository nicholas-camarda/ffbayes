#!/usr/bin/env python3
"""
Tests for enhanced data validation functionality.
Tests the data validation pipeline with comprehensive checks.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add scripts to path for testing
sys.path.append('scripts')
sys.path.append('scripts/utils')

try:
    from data_pipeline.validate_data import check_data_completeness, validate_data_quality
    from utils.progress_monitor import ProgressMonitor
    VALIDATE_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validate_data module: {e}")
    VALIDATE_DATA_AVAILABLE = False


class TestEnhancedDataValidation(unittest.TestCase):
    """Test suite for enhanced data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_datasets_dir = Path(self.temp_dir) / 'datasets'
        self.test_season_datasets_dir = self.test_datasets_dir / 'season_datasets'
        self.test_season_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample test data that matches the expected structure
        self.sample_data_2022 = pd.DataFrame({
            'G#': [1, 2, 3, 4, 5],
            'Date': ['2022-09-08', '2022-09-11', '2022-09-11', '2022-09-11', '2022-09-11'],
            'Tm': ['BUF', 'BUF', 'BUF', 'BUF', 'BUF'],
            'Away': ['LAR', 'TEN', 'TEN', 'TEN', 'TEN'],
            'Opp': ['LAR', 'TEN', 'TEN', 'TEN', 'TEN'],
            'FantPt': [18.5, 12.3, 22.1, 15.7, 8.0],
            'FantPtPPR': [18.5, 12.3, 22.1, 15.7, 8.0],
            'Name': ['Josh Allen', 'Devin Singletary', 'Stefon Diggs', 'Dawson Knox', 'Tyler Bass'],
            'PlayerID': [1, 2, 3, 4, 5],
            'Position': ['QB', 'RB', 'WR', 'TE', 'K'],
            'Season': [2022, 2022, 2022, 2022, 2022],
            'is_home': [1, 1, 1, 1, 1],
            'GameInjuryStatus': [None, None, None, None, None],
            'PracticeInjuryStatus': [None, None, None, None, None]
        })
        
        self.sample_data_2023 = pd.DataFrame({
            'G#': [1, 2, 3, 4, 5],
            'Date': ['2023-09-07', '2023-09-10', '2023-09-10', '2023-09-10', '2023-09-10'],
            'Tm': ['KC', 'KC', 'KC', 'KC', 'KC'],
            'Away': ['DET', 'JAX', 'JAX', 'JAX', 'JAX'],
            'Opp': ['DET', 'JAX', 'JAX', 'JAX', 'JAX'],
            'FantPt': [20.1, 14.8, 19.2, 12.5, 9.0],
            'FantPtPPR': [20.1, 14.8, 19.2, 12.5, 9.0],
            'Name': ['Patrick Mahomes', 'Isiah Pacheco', 'Travis Kelce', 'Rashee Rice', 'Harrison Butker'],
            'PlayerID': [6, 7, 8, 9, 10],
            'Position': ['QB', 'RB', 'TE', 'WR', 'K'],
            'Season': [2023, 2023, 2023, 2023, 2023],
            'is_home': [1, 1, 1, 1, 1],
            'GameInjuryStatus': [None, None, None, None, None],
            'PracticeInjuryStatus': [None, None, None, None, None]
        })
        
        # Create data with missing values to test validation
        self.sample_data_missing = pd.DataFrame({
            'G#': [1, 2, 3, 4, 5],
            'Date': ['2023-09-07', '2023-09-10', '2023-09-10', '2023-09-10', '2023-09-10'],
            'Tm': ['KC', 'KC', 'KC', 'KC', 'KC'],
            'Away': ['DET', 'JAX', 'JAX', 'JAX', 'JAX'],
            'Opp': ['DET', 'JAX', 'JAX', 'JAX', 'JAX'],
            'FantPt': [20.1, 14.8, np.nan, 12.5, 9.0],  # Missing value
            'FantPtPPR': [20.1, 14.8, 19.2, 12.5, 9.0],
            'Name': ['Patrick Mahomes', 'Isiah Pacheco', 'Travis Kelce', 'Rashee Rice', 'Harrison Butker'],
            'PlayerID': [6, 7, 8, 9, 10],
            'Position': ['QB', 'RB', 'TE', 'WR', 'K'],
            'Season': [2023, 2023, 2023, 2023, 2023],
            'is_home': [1, 1, 1, 1, 1],
            'GameInjuryStatus': [None, None, None, None, None],
            'PracticeInjuryStatus': [None, None, None, None, None]
        })
        
        # Create data with missing columns to test validation
        self.sample_data_missing_columns = pd.DataFrame({
            'G#': [1, 2, 3, 4, 5],
            'Date': ['2023-09-07', '2023-09-10', '2023-09-10', '2023-09-10', '2023-09-10'],
            'Tm': ['KC', 'KC', 'KC', 'KC', 'KC'],
            'FantPt': [20.1, 14.8, 19.2, 12.5, 9.0],
            'Name': ['Patrick Mahomes', 'Isiah Pacheco', 'Travis Kelce', 'Rashee Rice', 'Harrison Butker'],
            'PlayerID': [6, 7, 8, 9, 10],
            'Position': ['QB', 'RB', 'TE', 'WR', 'K'],
            'Season': [2023, 2023, 2023, 2023, 2023]
            # Missing several core columns
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_function_exists(self):
        """Test that validate_data_quality function exists and is callable."""
        self.assertTrue(callable(validate_data_quality))
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_check_data_completeness_function_exists(self):
        """Test that check_data_completeness function exists and is callable."""
        self.assertTrue(callable(check_data_completeness))
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_with_good_data(self):
        """Test validate_data_quality with good quality data."""
        # Create test files
        file_2022 = self.test_season_datasets_dir / '2022season.csv'
        file_2023 = self.test_season_datasets_dir / '2023season.csv'
        
        self.sample_data_2022.to_csv(file_2022, index=False)
        self.sample_data_2023.to_csv(file_2023, index=False)
        
        # Mock the glob.glob to return our test files
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_2022), str(file_2023)]
            
            result = validate_data_quality()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['season_files'], 2)
            self.assertEqual(result['total_rows'], 10)  # 5 rows per file
            self.assertEqual(result['quality_score'], 100.0)
            self.assertEqual(len(result['errors']), 0)
            self.assertEqual(len(result['warnings']), 0)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_with_missing_data(self):
        """Test validate_data_quality with data containing missing values."""
        # Create test file with missing data
        file_2023_missing = self.test_season_datasets_dir / '2023season.csv'
        self.sample_data_missing.to_csv(file_2023_missing, index=False)
        
        # Mock the glob.glob to return our test file
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_2023_missing)]
            
            result = validate_data_quality()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['season_files'], 1)
            self.assertEqual(result['total_rows'], 5)
            self.assertEqual(result['quality_score'], 100.0)  # Still 100% because missing is < 10%
            self.assertEqual(len(result['errors']), 0)
            self.assertEqual(len(result['warnings']), 0)  # Missing data is within acceptable threshold
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_with_missing_columns(self):
        """Test validate_data_quality with data missing core columns."""
        # Create test file with missing columns
        file_2023_missing_cols = self.test_season_datasets_dir / '2023season.csv'
        self.sample_data_missing_columns.to_csv(file_2023_missing_cols, index=False)
        
        # Mock the glob.glob to return our test file
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_2023_missing_cols)]
            
            result = validate_data_quality()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['season_files'], 1)
            self.assertEqual(result['total_rows'], 5)
            self.assertEqual(result['quality_score'], 100.0)
            self.assertGreater(len(result['errors']), 0)  # Should have errors for missing columns
            self.assertEqual(len(result['warnings']), 0)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_with_no_files(self):
        """Test validate_data_quality when no season files exist."""
        # Mock the glob.glob to return no files
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = []
            
            result = validate_data_quality()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['season_files'], 0)
            self.assertEqual(result['total_rows'], 0)
            self.assertEqual(result['quality_score'], 100)
            self.assertGreater(len(result['errors']), 0)  # Should have error for no files
            self.assertEqual(len(result['warnings']), 0)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_with_file_read_error(self):
        """Test validate_data_quality when file reading fails."""
        # Create a test file
        file_2023 = self.test_season_datasets_dir / '2023season.csv'
        self.sample_data_2023.to_csv(file_2023, index=False)
        
        # Mock the glob.glob to return our test file
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_2023)]
            
            # Mock pd.read_csv to raise an exception
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = Exception("File read error")
                
                result = validate_data_quality()
                
                self.assertIsInstance(result, dict)
                self.assertEqual(result['season_files'], 1)
                self.assertEqual(result['total_rows'], 0)
                self.assertGreater(len(result['errors']), 0)  # Should have error for file read failure
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_check_data_completeness_with_all_years(self):
        """Test check_data_completeness when all expected years are available."""
        # Create test files for multiple years
        for year in [2020, 2021, 2022, 2023]:
            file_path = self.test_season_datasets_dir / f'{year}season.csv'
            sample_data = pd.DataFrame({
                'Season': [year] * 5,
                'G#': [1, 2, 3, 4, 5],
                'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']
            })
            sample_data.to_csv(file_path, index=False)
        
        # Mock the os.path.exists to return True for our test files
        with patch('data_pipeline.validate_data.os.path.exists') as mock_exists:
            def mock_exists_side_effect(path):
                if 'season_datasets' in str(path) and 'season.csv' in str(path):
                    return True
                return False
            mock_exists.side_effect = mock_exists_side_effect
            
            result = check_data_completeness()
            
            self.assertTrue(result)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_check_data_completeness_with_missing_years(self):
        """Test check_data_completeness when some years are missing."""
        # Create test files for only some years
        for year in [2021, 2023]:  # Missing 2020, 2022
            file_path = self.test_season_datasets_dir / f'{year}season.csv'
            sample_data = pd.DataFrame({
                'Season': [year] * 5,
                'G#': [1, 2, 3, 4, 5],
                'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']
            })
            sample_data.to_csv(file_path, index=False)
        
        # Mock the os.path.exists to return True only for existing files
        with patch('data_pipeline.validate_data.os.path.exists') as mock_exists:
            def mock_exists_side_effect(path):
                if 'season_datasets' in str(path) and 'season.csv' in str(path):
                    year = str(path).split('season.csv')[0].split('/')[-1]
                    return year in ['2021', '2023']
                return False
            mock_exists.side_effect = mock_exists_side_effect
            
            result = check_data_completeness()
            
            self.assertFalse(result)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_core_columns_validation(self):
        """Test that core columns are properly validated."""
        # Create test data with all core columns
        core_columns = ['G#', 'Date', 'Tm', 'Away', 'Opp', 'FantPt', 'FantPtPPR', 'Name', 'PlayerID', 'Position', 'Season', 'is_home']
        
        test_data = pd.DataFrame({
            'G#': [1, 2, 3],
            'Date': ['2023-09-07', '2023-09-10', '2023-09-10'],
            'Tm': ['KC', 'KC', 'KC'],
            'Away': ['DET', 'JAX', 'JAX'],
            'Opp': ['DET', 'JAX', 'JAX'],
            'FantPt': [20.1, 14.8, 19.2],
            'FantPtPPR': [20.1, 14.8, 19.2],
            'Name': ['Patrick Mahomes', 'Isiah Pacheco', 'Travis Kelce'],
            'PlayerID': [6, 7, 8],
            'Position': ['QB', 'RB', 'TE'],
            'Season': [2023, 2023, 2023],
            'is_home': [1, 1, 1]
        })
        
        file_path = self.test_season_datasets_dir / '2023season.csv'
        test_data.to_csv(file_path, index=False)
        
        # Mock the glob.glob to return our test file
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_path)]
            
            result = validate_data_quality()
            
            self.assertEqual(result['season_files'], 1)
            self.assertEqual(result['total_rows'], 3)
            self.assertEqual(len(result['errors']), 0)  # All core columns present
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_validate_data_quality_missing_data_threshold(self):
        """Test that missing data threshold is properly enforced."""
        # Create test data with high missing data (>10%)
        test_data = pd.DataFrame({
            'G#': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Date': ['2023-09-07'] * 10,
            'Tm': ['KC'] * 10,
            'Away': ['DET'] * 10,
            'Opp': ['DET'] * 10,
            'FantPt': [20.1, 14.8, np.nan, 12.5, 9.0, 18.2, np.nan, 15.7, 11.3, np.nan],  # 30% missing
            'FantPtPPR': [20.1, 14.8, 19.2, 12.5, 9.0, 18.2, 16.8, 15.7, 11.3, 13.1],
            'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
            'PlayerID': list(range(1, 11)),
            'Position': ['QB', 'RB', 'TE', 'WR', 'K', 'QB', 'RB', 'TE', 'WR', 'K'],
            'Season': [2023] * 10,
            'is_home': [1] * 10
        })
        
        file_path = self.test_season_datasets_dir / '2023season.csv'
        test_data.to_csv(file_path, index=False)
        
        # Mock the glob.glob to return our test file
        with patch('data_pipeline.validate_data.glob.glob') as mock_glob:
            mock_glob.return_value = [str(file_path)]
            
            result = validate_data_quality()
            
            self.assertEqual(result['season_files'], 1)
            self.assertEqual(result['total_rows'], 10)
            self.assertEqual(len(result['warnings']), 1)  # Should have warning for high missing data
            self.assertEqual(result['missing_data'], 1)
    
    @unittest.skipUnless(VALIDATE_DATA_AVAILABLE, "validate_data module not available")
    def test_progress_monitoring_integration(self):
        """Test that progress monitoring is properly integrated."""
        # Test that ProgressMonitor can be imported and used
        self.assertTrue(hasattr(ProgressMonitor, 'monitor'))
        self.assertTrue(hasattr(ProgressMonitor, 'start_timer'))
        self.assertTrue(hasattr(ProgressMonitor, 'elapsed_time'))
    
    def test_directory_structure_requirements(self):
        """Test that required directory structure exists."""
        required_dirs = [
            'scripts/data_pipeline',
            'scripts/utils',
            'datasets',
            'tests'
        ]
        
        for dir_path in required_dirs:
            with self.subTest(dir_path=dir_path):
                self.assertTrue(
                    os.path.exists(dir_path),
                    f"Required directory {dir_path} should exist"
                )
    
    def test_file_requirements(self):
        """Test that required files exist."""
        required_files = [
            'scripts/data_pipeline/02_validate_data.py',
            'scripts/utils/progress_monitor.py',
            'tests/test_enhanced_data_validation.py'
        ]
        
        for file_path in required_files:
            with self.subTest(file_path=file_path):
                self.assertTrue(
                    os.path.exists(file_path),
                    f"Required file {file_path} should exist"
                )


if __name__ == '__main__':
    unittest.main()
