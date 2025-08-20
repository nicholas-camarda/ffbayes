#!/usr/bin/env python3
"""
Tests for scripts/utils/interface_standards.py utilities.
"""

import os

# Ensure utils on path
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from ffbayes.utils.interface_standards import (
	get_env_bool,
	get_env_int,
	get_standard_paths,
	handle_exception,
	setup_logger,
)


class TestInterfaceStandards(unittest.TestCase):
	def test_get_env_bool(self):
		os.environ['TEST_BOOL_TRUE'] = 'true'
		os.environ['TEST_BOOL_ONE'] = '1'
		os.environ['TEST_BOOL_NO'] = 'no'
		self.assertTrue(get_env_bool('TEST_BOOL_TRUE', False))
		self.assertTrue(get_env_bool('TEST_BOOL_ONE', False))
		self.assertFalse(get_env_bool('TEST_BOOL_NO', True))
		self.assertTrue(get_env_bool('TEST_BOOL_MISSING', True))

	def test_get_env_int(self):
		os.environ['TEST_INT_OK'] = '42'
		os.environ['TEST_INT_BAD'] = 'abc'
		self.assertEqual(get_env_int('TEST_INT_OK', 0), 42)
		self.assertEqual(get_env_int('TEST_INT_BAD', 7), 7)
		self.assertEqual(get_env_int('TEST_INT_MISSING', 9), 9)

	def test_setup_logger(self):
		os.environ['LOG_LEVEL'] = 'DEBUG'
		logger = setup_logger('test.logger')
		self.assertEqual(logger.level, 10)  # DEBUG
		logger.debug('debug message')
		self.assertTrue(len(logger.handlers) >= 1)

	def test_standard_paths(self):
		paths = get_standard_paths(Path.cwd())
		self.assertTrue(str(paths.plots_root).endswith('plots'))
		self.assertTrue(str(paths.results_root).endswith('results'))
		self.assertTrue(str(paths.datasets_root).endswith('datasets'))
		self.assertTrue(str(paths.monte_carlo_results).endswith('results/montecarlo_results'))
		self.assertTrue(str(paths.bayesian_results).endswith('results/bayesian-hierarchical-results'))
		self.assertTrue(str(paths.team_aggregation_plots).endswith('plots/team_aggregation'))
		self.assertTrue(str(paths.test_runs_plots).endswith('plots/test_runs'))

	def test_handle_exception(self):
		try:
			raise ValueError('bad value')
		except Exception as e:
			msg = handle_exception(e, context='UnitTest')
			self.assertIn('UnitTest', msg)
			self.assertIn('ValueError', msg)
			self.assertIn('bad value', msg)


if __name__ == '__main__':
	unittest.main()
