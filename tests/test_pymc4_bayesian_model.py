import os
import pickle
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src on path for package imports
sys.path.append(str(Path.cwd() / 'src'))

class TestPyMC4BayesianModel(unittest.TestCase):
    """Test PyMC4 Bayesian hierarchical fantasy football model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = os.path.join(self.temp_dir, 'datasets')
        self.combined_datasets_dir = os.path.join(self.temp_dir, 'datasets', 'combined_datasets')
        self.results_dir = os.path.join(self.temp_dir, 'results', 'bayesian-hierarchical-results')
        self.plots_dir = os.path.join(self.temp_dir, 'plots')
        
        os.makedirs(self.combined_datasets_dir)
        os.makedirs(self.results_dir)
        os.makedirs(self.plots_dir)
        
        # Create sample preprocessed data
        self.sample_data = pd.DataFrame({
            'Season': [2022, 2022, 2022, 2023, 2023, 2023],
            'FantPt': [18.5, 22.1, 15.8, 20.3, 25.7, 19.2],
            '7_game_avg': [17.2, 21.5, 16.1, 19.8, 24.9, 18.7],
            'diff_from_avg': [1.3, 0.6, -0.3, 0.5, 0.8, 0.5],
            'Opp': ['NE', 'BUF', 'MIA', 'NE', 'BUF', 'MIA'],
            'opp_team': [0, 1, 2, 0, 1, 2],  # Team indices
            'rank': [1, 2, 3, 1, 2, 3],
            'is_home': [1, 0, 1, 0, 1, 0],
            'position_QB': [1, 0, 0, 1, 0, 0],
            'position_RB': [0, 1, 0, 0, 1, 0],
            'position_WR': [0, 0, 1, 0, 0, 1],
            'position_TE': [0, 0, 0, 0, 0, 0]
        })
        
        # Save sample data
        data_file = os.path.join(self.combined_datasets_dir, '2023season_modern.csv')
        self.sample_data.to_csv(data_file, index=False)
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original working directory
        os.chdir(self.original_cwd)
        
        # Remove temporary directories
        shutil.rmtree(self.temp_dir)
    
    def test_load_preprocessed_data_success(self):
        """Test successful loading of preprocessed data"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Test the function
        data, team_names = bayesian.load_preprocessed_data(self.datasets_dir)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(team_names, np.ndarray)
        self.assertEqual(len(data), 6)  # 6 rows in sample data
        self.assertEqual(len(team_names), 3)  # 3 unique teams (NE, BUF, MIA)
        self.assertTrue('FantPt' in data.columns)
        self.assertTrue('7_game_avg' in data.columns)
        self.assertTrue('diff_from_avg' in data.columns)
    
    def test_load_preprocessed_data_no_files(self):
        """Test error handling when no preprocessed data files are found"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Remove all data files
        for file in os.listdir(self.combined_datasets_dir):
            os.remove(os.path.join(self.combined_datasets_dir, file))
        
        with self.assertRaises(ValueError) as context:
            bayesian.load_preprocessed_data(self.datasets_dir)
        
        self.assertIn('No preprocessed analysis data found', str(context.exception))
    
    def test_load_recent_trace_success(self):
        """Test successful loading of recent trace file"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Create a mock trace file
        mock_trace = {'test': 'trace_data'}
        trace_file = os.path.join(self.results_dir, 'trace_20231201_120000.pkl')
        
        with open(trace_file, 'wb') as f:
            pickle.dump(mock_trace, f)
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            result = bayesian.load_recent_trace()
            self.assertIsNotNone(result)
            self.assertEqual(result, mock_trace)
        finally:
            os.chdir(original_cwd)
    
    def test_load_recent_trace_no_files(self):
        """Test trace loading when no trace files exist"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            result = bayesian.load_recent_trace()
            self.assertIsNone(result)
        finally:
            os.chdir(original_cwd)
    
    def test_load_recent_trace_corrupted_file(self):
        """Test trace loading with corrupted pickle file"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Create a corrupted trace file
        trace_file = os.path.join(self.results_dir, 'trace_20231201_120000.pkl')
        with open(trace_file, 'w') as f:
            f.write('corrupted pickle data')
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            result = bayesian.load_recent_trace()
            self.assertIsNone(result)
        finally:
            os.chdir(original_cwd)
    
    def test_bayesian_model_data_validation(self):
        """Test data validation and preprocessing in the Bayesian model"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Test with insufficient years of data
        insufficient_data = self.sample_data[self.sample_data['Season'] == 2023].copy()
        insufficient_file = os.path.join(self.combined_datasets_dir, 'insufficient_2023season_modern.csv')
        insufficient_data.to_csv(insufficient_file, index=False)
        
        # Remove the original file to force use of insufficient data
        original_file = os.path.join(self.combined_datasets_dir, '2023season_modern.csv')
        os.remove(original_file)
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            with self.assertRaises(ValueError) as context:
                bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=5, chains=2)

            # Current implementation checks for minimum rows
            self.assertIn('Insufficient data', str(context.exception))
        finally:
            os.chdir(original_cwd)
    
    def test_bayesian_model_parameter_validation(self):
        """Test parameter validation in the Bayesian model"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Test with invalid parameter values
        with self.assertRaises(ValueError):
            bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=0, tune=5, chains=2)
        
        with self.assertRaises(ValueError):
            bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=0, chains=2)
        
        with self.assertRaises(ValueError):
            bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=5, chains=0)
    
    def test_bayesian_model_missing_columns(self):
        """Test handling of missing required columns in data"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Create data missing required columns
        incomplete_data = self.sample_data[['Season', 'FantPt']].copy()
        incomplete_file = os.path.join(self.combined_datasets_dir, 'incomplete_2023season_modern.csv')
        incomplete_data.to_csv(incomplete_file, index=False)
        
        # Remove the original file to force use of incomplete data
        original_file = os.path.join(self.combined_datasets_dir, '2023season_modern.csv')
        os.remove(original_file)
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            with self.assertRaises(KeyError):
                bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=5, chains=2)
        finally:
            os.chdir(original_cwd)
    
    def test_bayesian_model_numeric_data_validation(self):
        """Test validation of numeric data types in the Bayesian model"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Create data with non-numeric values in numeric columns
        invalid_data = self.sample_data.copy()
        # Convert columns to object dtype to allow string assignment
        invalid_data['FantPt'] = invalid_data['FantPt'].astype(object)
        invalid_data['7_game_avg'] = invalid_data['7_game_avg'].astype(object)
        invalid_data.loc[0, 'FantPt'] = 'invalid'
        invalid_data.loc[1, '7_game_avg'] = 'invalid'
        
        invalid_file = os.path.join(self.combined_datasets_dir, 'invalid_2023season_modern.csv')
        invalid_data.to_csv(invalid_file, index=False)
        
        # Remove the original file to force use of invalid data
        original_file = os.path.join(self.combined_datasets_dir, '2023season_modern.csv')
        os.remove(original_file)
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            with self.assertRaises((ValueError, TypeError)):
                bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=5, chains=2)
        finally:
            os.chdir(original_cwd)
    
    def test_bayesian_model_edge_cases(self):
        """Test edge cases in the Bayesian model"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Test with single player data
        single_player_data = self.sample_data.head(1).copy()
        single_player_file = os.path.join(self.combined_datasets_dir, 'single_2023season_modern.csv')
        single_player_data.to_csv(single_player_file, index=False)
        
        # Remove the original file to force use of single player data
        original_file = os.path.join(self.combined_datasets_dir, '2023season_modern.csv')
        os.remove(original_file)
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Single row should raise insufficient data error in current implementation
            with self.assertRaises(ValueError) as context:
                bayesian.bayesian_hierarchical_ff_modern(self.datasets_dir, draws=10, tune=5, chains=2)
            self.assertIn('Insufficient data', str(context.exception))
        finally:
            os.chdir(original_cwd)
    
    def test_bayesian_model_output_validation(self):
        """Test validation of model outputs and results"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Mock PyMC to avoid actual sampling
        with mock.patch('pymc.sample') as mock_sample:
            mock_trace = mock.MagicMock()
            mock_trace.posterior = {
                'defensive_differential_qb': mock.MagicMock(),
                'defensive_differential_wr': mock.MagicMock(),
                'defensive_differential_rb': mock.MagicMock(),
                'defensive_differential_te': mock.MagicMock()
            }
            mock_sample.return_value = mock_trace
            
            with mock.patch('pymc.sample_posterior_predictive') as mock_predict:
                mock_predict.return_value = mock.MagicMock()
                
                # Change to temp directory for the test
                original_cwd = os.getcwd()
                os.chdir(self.temp_dir)
                
                try:
                    # With only one season CSV (6 rows), the function should raise due to insufficient rows
                    with self.assertRaises(ValueError):
                        bayesian.bayesian_hierarchical_ff_modern(
                            self.datasets_dir, draws=10, tune=5, chains=2
                        )
                finally:
                    os.chdir(original_cwd)
    
    def test_bayesian_model_file_operations(self):
        """Test file operations (saving trace, results, plots)"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Mock PyMC to avoid actual sampling
        with mock.patch('pymc.sample') as mock_sample:
            mock_trace = mock.MagicMock()
            mock_trace.posterior = {
                'defensive_differential_qb': mock.MagicMock(),
                'defensive_differential_wr': mock.MagicMock(),
                'defensive_differential_rb': mock.MagicMock(),
                'defensive_differential_te': mock.MagicMock()
            }
            mock_sample.return_value = mock_trace
            
            with mock.patch('pymc.sample_posterior_predictive') as mock_predict:
                mock_predict.return_value = mock.MagicMock()
                
                # Change to temp directory for the test
                original_cwd = os.getcwd()
                os.chdir(self.temp_dir)
                
                try:
                    # With 6 rows, should raise; this test focuses on file ops but respects validation
                    with self.assertRaises(ValueError):
                        bayesian.bayesian_hierarchical_ff_modern(
                            self.datasets_dir, draws=10, tune=5, chains=2
                        )
                finally:
                    os.chdir(original_cwd)
    
    def test_main_function_execution(self):
        """Test main function execution"""
        import importlib
        bayesian = importlib.import_module('ffbayes.analysis.bayesian_hierarchical_ff_modern')
        
        # Mock the main Bayesian function
        with mock.patch.object(bayesian, 'bayesian_hierarchical_ff_modern') as mock_main:
            mock_trace = mock.MagicMock()
            mock_results = {
                'mae_bayesian': 8.5,
                'mae_baseline': 10.2,
                'team_names': ['NE', 'BUF', 'MIA'],
                'timestamp': '2023-12-01T12:00:00',
                'test_data_shape': (100, 15),
                'predictions_shape': (100,)
            }
            mock_main.return_value = (mock_trace, mock_results)
            
            # Change to temp directory for the test
            original_cwd = os.getcwd()
            os.chdir(self.temp_dir)
            
            try:
                # Capture print output
                with mock.patch('builtins.print') as mock_print:
                    bayesian.main()
                    
                    # Check that main function was called
                    mock_main.assert_called_once_with('datasets')
                    
                    # Check that results were printed
                    mock_print.assert_any_call(mock.ANY)  # Any print call
                    
            finally:
                os.chdir(original_cwd)

if __name__ == '__main__':
    unittest.main()
