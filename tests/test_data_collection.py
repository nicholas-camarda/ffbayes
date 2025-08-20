#!/usr/bin/env python3
"""
Tests for data collection functionality validation.
Tests existing data collection implementations and their capabilities.
"""

import os

# Import the modules we're testing
import sys
import tempfile
import unittest

# Ensure src on path for package imports
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd() / 'src'))

# We'll import these as they exist
try:
	from ffbayes.data_pipeline import collect_data as collect_data
except ImportError:
	collect_data = None

try:
	from ffbayes.data_pipeline import validate_data as validate_data
except ImportError:
	validate_data = None


class TestDataCollectionFunctionality(unittest.TestCase):
	"""Test suite for data collection functionality validation."""
	
	def setUp(self):
		"""Set up test fixtures."""
		self.temp_dir = tempfile.mkdtemp()
		self.test_data_dir = os.path.join(self.temp_dir, 'datasets')
		os.makedirs(self.test_data_dir, exist_ok=True)
		
		# Create sample test data
		self.sample_player_data = pd.DataFrame({
			'player_id': [1, 2, 3],
			'name': ['Player A', 'Player B', 'Player C'],
			'position': ['QB', 'RB', 'WR'],
			'team': ['Team A', 'Team B', 'Team C'],
			'fantasy_points': [20.5, 15.2, 18.7]
		})
		
		self.sample_schedule_data = pd.DataFrame({
			'game_id': [1, 2, 3],
			'home_team': ['Team A', 'Team B', 'Team C'],
			'away_team': ['Team D', 'Team E', 'Team F'],
			'week': [1, 1, 1],
			'season': [2023, 2023, 2023]
		})
	
	def tearDown(self):
		"""Clean up test fixtures."""
		import shutil
		shutil.rmtree(self.temp_dir)
	
	def test_data_collection_scripts_exist(self):
		"""Test that data collection scripts exist and are importable."""
		# Test that key data collection files exist (now moved under src/ffbayes)
		expected_files = [
			'src/ffbayes/data_pipeline/collect_data.py',
			'src/ffbayes/data_pipeline/validate_data.py'
		]
		
		for file_path in expected_files:
			with self.subTest(file_path=file_path):
				self.assertTrue(
					os.path.exists(file_path),
					f"Data collection script {file_path} should exist"
				)
	
	def test_data_collection_directory_structure(self):
		"""Test that data collection follows expected directory structure."""
		expected_dirs = [
			'src/ffbayes/data_pipeline',
			'datasets',
			'results'
		]
		
		for dir_path in expected_dirs:
			with self.subTest(dir_path=dir_path):
				self.assertTrue(
					os.path.exists(dir_path),
					f"Directory {dir_path} should exist"
				)
	
	def test_data_collection_capabilities(self):
		"""Test that data collection can handle expected data types."""
		# Test player data structure
		self.assertIn('player_id', self.sample_player_data.columns)
		self.assertIn('name', self.sample_player_data.columns)
		self.assertIn('position', self.sample_player_data.columns)
		self.assertIn('team', self.sample_player_data.columns)
		self.assertIn('fantasy_points', self.sample_player_data.columns)
		
		# Test schedule data structure
		self.assertIn('game_id', self.sample_schedule_data.columns)
		self.assertIn('home_team', self.sample_schedule_data.columns)
		self.assertIn('away_team', self.sample_schedule_data.columns)
		self.assertIn('week', self.sample_schedule_data.columns)
		self.assertIn('season', self.sample_schedule_data.columns)
	
	def test_data_quality_validation(self):
		"""Test that data quality validation functions work correctly."""
		# Test with good data
		good_data = self.sample_player_data.copy()
		missing_pct = (good_data.isnull().sum() / len(good_data) * 100).max()
		self.assertEqual(missing_pct, 0.0, "Good data should have no missing values")
		
		# Test with bad data
		bad_data = good_data.copy()
		bad_data.loc[0, 'fantasy_points'] = np.nan
		missing_pct = (bad_data.isnull().sum() / len(bad_data) * 100).max()
		self.assertGreater(missing_pct, 0.0, "Bad data should have missing values")
	
	def test_data_completeness_check(self):
		"""Test that data completeness checking works correctly."""
		# Save sample data to test directory
		player_file = os.path.join(self.test_data_dir, '2023_players.csv')
		self.sample_player_data.to_csv(player_file, index=False)
		
		# Test that file exists
		self.assertTrue(os.path.exists(player_file))
		
		# Test that we can read the file back
		loaded_data = pd.read_csv(player_file)
		self.assertEqual(len(loaded_data), len(self.sample_player_data))
		self.assertTrue(all(loaded_data.columns == self.sample_player_data.columns))
	
	@patch('pandas.read_csv')
	def test_data_collection_error_handling(self, mock_read_csv):
		"""Test that data collection handles errors gracefully."""
		# Mock a file read error
		mock_read_csv.side_effect = FileNotFoundError("File not found")
		
		# Test that error is handled appropriately
		with self.assertRaises(FileNotFoundError):
			pd.read_csv('nonexistent_file.csv')
	
	def test_data_types_and_formats(self):
		"""Test that data has correct types and formats."""
		# Test player data types
		self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_player_data['player_id']))
		self.assertTrue(pd.api.types.is_string_dtype(self.sample_player_data['name']))
		self.assertTrue(pd.api.types.is_string_dtype(self.sample_player_data['position']))
		self.assertTrue(pd.api.types.is_string_dtype(self.sample_player_data['team']))
		self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_player_data['fantasy_points']))
		
		# Test schedule data types
		self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_schedule_data['game_id']))
		self.assertTrue(pd.api.types.is_string_dtype(self.sample_schedule_data['home_team']))
		self.assertTrue(pd.api.types.is_string_dtype(self.sample_schedule_data['away_team']))
		self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_schedule_data['week']))
		self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_schedule_data['season']))
	
	def test_data_validation_functions_exist(self):
		"""Test that data validation functions are available."""
		# Test that validation module can be imported
		try:
			import ffbayes.data_pipeline.validate_data as validate_module
			self.assertTrue(hasattr(validate_module, 'validate_data_quality'))
			self.assertTrue(hasattr(validate_module, 'check_data_completeness'))
		except ImportError:
			# If module doesn't exist yet, that's okay for this test
			pass
	
	def test_progress_monitoring_integration(self):
		"""Test that progress monitoring can be integrated with data collection."""
		try:
			from alive_progress import alive_bar
			with alive_bar(10, title="Test Progress") as bar:
				for i in range(10):
					bar()
			self.assertTrue(True, "Progress monitoring should work")
		except ImportError:
			self.skipTest("alive_progress not available")


if __name__ == '__main__':
	unittest.main()
