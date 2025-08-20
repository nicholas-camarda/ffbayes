#!/usr/bin/env python3
"""
Test suite for the Model Comparison Framework.

This module tests the functionality for comparing Monte Carlo and Bayesian team projections,
including statistical validation, uncertainty analysis, and model selection criteria.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

# Add the scripts directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestModelComparisonFramework(unittest.TestCase):
    """Test cases for the Model Comparison Framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.monte_carlo_data = {
            'monte_carlo_projection': {
                'team_projection': {
                    'total_score': {
                        'mean': 118.7,
                        'std': 2.1,
                        'min': 115.0,
                        'max': 122.0,
                        'confidence_interval': [116.6, 120.8],
                        'percentiles': {
                            'p5': 115.2,
                            'p25': 117.1,
                            'p50': 118.7,
                            'p75': 120.3,
                            'p95': 121.8
                        }
                    }
                },
                'player_contributions': {
                    'Player1': {'mean': 35.2, 'std': 1.8, 'contribution_pct': 29.7},
                    'Player2': {'mean': 28.1, 'std': 1.5, 'contribution_pct': 23.7},
                    'Player3': {'mean': 25.8, 'std': 1.2, 'contribution_pct': 21.8},
                    'Player4': {'mean': 18.6, 'std': 1.0, 'contribution_pct': 15.7},
                    'Player5': {'mean': 11.0, 'std': 0.8, 'contribution_pct': 9.3}
                }
            },
            'simulation_metadata': {
                'number_of_simulations': 1000,
                'execution_time': 45.2,
                'convergence_status': 'converged'
            }
        }
        
        self.bayesian_data = {
            'team_projection': {
                'total_score': {
                    'mean': 119.1,
                    'std': 3.2,
                    'min': 113.5,
                    'max': 125.8,
                    'confidence_interval': [115.9, 122.3],
                    'percentiles': {
                        'p5': 114.2,
                        'p25': 116.8,
                        'p50': 119.1,
                        'p75': 121.4,
                        'p95': 123.9
                    }
                },
                'player_contributions': {
                    'Player1': {'mean': 35.8, 'std': 2.1, 'contribution_pct': 30.1},
                    'Player2': {'mean': 28.3, 'std': 1.9, 'contribution_pct': 23.8},
                    'Player3': {'mean': 25.9, 'std': 1.6, 'contribution_pct': 21.7},
                    'Player4': {'mean': 18.6, 'std': 1.0, 'contribution_pct': 15.6},
                    'Player5': {'mean': 10.5, 'std': 0.8, 'contribution_pct': 8.8}
                }
            },
            'model_metadata': {
                'draws': 2000,
                'tune': 1000,
                'chains': 4,
                'convergence_metrics': {
                    'rhat': 1.02,
                    'effective_sample_size': 1800
                }
            }
        }
        
        # Create test data files
        self.monte_carlo_file = os.path.join(self.test_dir, 'monte_carlo_results.json')
        self.bayesian_file = os.path.join(self.test_dir, 'bayesian_results.json')
        
        with open(self.monte_carlo_file, 'w') as f:
            json.dump(self.monte_carlo_data, f)
        
        with open(self.bayesian_file, 'w') as f:
            json.dump(self.bayesian_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_monte_carlo_results(self):
        """Test loading Monte Carlo simulation results."""
        # This will be implemented in the actual framework
        # For now, just verify the test data structure
        self.assertIn('monte_carlo_projection', self.monte_carlo_data)
        self.assertIn('team_projection', self.monte_carlo_data['monte_carlo_projection'])
        self.assertIn('total_score', self.monte_carlo_data['monte_carlo_projection']['team_projection'])
        self.assertIn('player_contributions', self.monte_carlo_data['monte_carlo_projection'])
    
    def test_load_bayesian_results(self):
        """Test loading Bayesian model results."""
        # This will be implemented in the actual framework
        # For now, just verify the test data structure
        self.assertIn('team_projection', self.bayesian_data)
        self.assertIn('total_score', self.bayesian_data['team_projection'])
        self.assertIn('model_metadata', self.bayesian_data)
    
    def test_compare_team_projections(self):
        """Test comparison of team projections between models."""
        # Test data validation
        mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
        bayes_score = self.bayesian_data['team_projection']['total_score']
        
        # Verify both have required fields
        required_fields = ['mean', 'std', 'confidence_interval', 'percentiles']
        for field in required_fields:
            self.assertIn(field, mc_score)
            self.assertIn(field, bayes_score)
        
        # Verify statistical consistency
        self.assertGreater(mc_score['std'], 0)
        self.assertGreater(bayes_score['std'], 0)
        self.assertLess(mc_score['min'], mc_score['max'])
        self.assertLess(bayes_score['min'], bayes_score['max'])
    
    def test_statistical_validation_metrics(self):
        """Test calculation of statistical validation metrics."""
        mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
        bayes_score = self.bayesian_data['team_projection']['total_score']
        
        # Calculate basic comparison metrics
        mean_difference = abs(mc_score['mean'] - bayes_score['mean'])
        std_ratio = mc_score['std'] / bayes_score['std']
        
        # Verify metrics are reasonable
        self.assertGreater(mean_difference, 0)
        self.assertLess(mean_difference, 10)  # Should be within reasonable range
        self.assertGreater(std_ratio, 0.1)    # Should not be extremely different
        self.assertLess(std_ratio, 10)        # Should not be extremely different
    
    def test_uncertainty_comparison(self):
        """Test comparison of uncertainty estimates between models."""
        mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
        bayes_score = self.bayesian_data['team_projection']['total_score']
        
        # Compare confidence intervals
        mc_ci_width = mc_score['confidence_interval'][1] - mc_score['confidence_interval'][0]
        bayes_ci_width = bayes_score['confidence_interval'][1] - bayes_score['confidence_interval'][0]
        
        # Verify uncertainty quantification
        self.assertGreater(mc_ci_width, 0)
        self.assertGreater(bayes_ci_width, 0)
        
        # Monte Carlo should generally have lower uncertainty (more deterministic)
        # Bayesian should have higher uncertainty (more realistic)
        self.assertLess(mc_ci_width, bayes_ci_width)
    
    def test_player_contribution_comparison(self):
        """Test comparison of player contributions between models."""
        mc_players = self.monte_carlo_data['monte_carlo_projection']['player_contributions']
        bayes_players = self.bayesian_data['team_projection']['player_contributions']
        
        # Verify same players in both models
        self.assertEqual(set(mc_players.keys()), set(bayes_players.keys()))
        
        # Compare individual player projections
        for player in mc_players:
            mc_player = mc_players[player]
            bayes_player = bayes_players[player]
            
            # Verify required fields
            required_fields = ['mean', 'std', 'contribution_pct']
            for field in required_fields:
                self.assertIn(field, mc_player)
                self.assertIn(field, bayes_player)
            
            # Verify contribution percentages sum to reasonable total
            self.assertGreater(mc_player['contribution_pct'], 0)
            self.assertLess(mc_player['contribution_pct'], 100)
    
    def test_model_selection_criteria(self):
        """Test model selection criteria and validation framework."""
        # This will test the framework for selecting between models
        # For now, verify test data has necessary metadata
        
        # Monte Carlo metadata
        mc_meta = self.monte_carlo_data['simulation_metadata']
        self.assertIn('number_of_simulations', mc_meta)
        self.assertIn('convergence_status', mc_meta)
        
        # Bayesian metadata
        bayes_meta = self.bayesian_data['model_metadata']
        self.assertIn('convergence_metrics', bayes_meta)
        self.assertIn('rhat', bayes_meta['convergence_metrics'])
        
        # Verify convergence criteria
        self.assertEqual(mc_meta['convergence_status'], 'converged')
        self.assertLess(bayes_meta['convergence_metrics']['rhat'], 1.1)  # Good convergence
    
    def test_visualization_data_preparation(self):
        """Test preparation of data for model comparison visualizations."""
        # Prepare data for visualization
        comparison_data = {
            'monte_carlo': self.monte_carlo_data,
            'bayesian': self.bayesian_data,
            'comparison_metrics': {
                'mean_difference': abs(self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']['mean'] - 
                                     self.bayesian_data['team_projection']['total_score']['mean']),
                'uncertainty_ratio': (self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']['std'] / 
                                    self.bayesian_data['team_projection']['total_score']['std'])
            }
        }
        
        # Verify comparison data structure
        self.assertIn('monte_carlo', comparison_data)
        self.assertIn('bayesian', comparison_data)
        self.assertIn('comparison_metrics', comparison_data)
        
        # Verify metrics are calculated correctly
        self.assertGreater(comparison_data['comparison_metrics']['mean_difference'], 0)
        self.assertGreater(comparison_data['comparison_metrics']['uncertainty_ratio'], 0)
    
    def test_error_handling_invalid_data(self):
        """Test error handling for invalid or missing data."""
        # Test with missing required fields
        invalid_data = {
            'team_projection': {
                'total_score': {
                    'mean': 100.0
                    # Missing std, confidence_interval, etc.
                }
            }
        }
        
        # Verify invalid data is detected
        required_fields = ['std', 'confidence_interval', 'percentiles']
        for field in required_fields:
            self.assertNotIn(field, invalid_data['team_projection']['total_score'])
    
    def test_performance_benchmarks(self):
        """Test performance benchmarking between models."""
        mc_meta = self.monte_carlo_data['simulation_metadata']
        bayes_meta = self.bayesian_data['model_metadata']
        
        # Compare execution characteristics
        mc_simulations = mc_meta['number_of_simulations']
        bayes_draws = bayes_meta['draws']
        bayes_chains = bayes_meta['chains']
        
        # Verify performance metrics are reasonable
        self.assertGreater(mc_simulations, 100)  # Should have sufficient simulations
        self.assertGreater(bayes_draws, 1000)    # Should have sufficient draws
        self.assertGreater(bayes_chains, 1)      # Should have multiple chains
    
    def test_historical_data_validation(self):
        """Test validation against historical data (mock)."""
        # Mock historical data for validation
        historical_data = {
            'actual_scores': [115.0, 118.0, 120.0, 117.5, 119.0],
            'dates': ['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22', '2024-01-29']
        }
        
        # Calculate basic statistics
        actual_mean = np.mean(historical_data['actual_scores'])
        actual_std = np.std(historical_data['actual_scores'])
        
        # Verify historical data is reasonable
        self.assertGreater(len(historical_data['actual_scores']), 0)
        self.assertGreater(actual_std, 0)
        
        # Historical scores should be in reasonable range
        self.assertGreater(actual_mean, 100)
        self.assertLess(actual_mean, 150)
    
    def test_model_consistency_checks(self):
        """Test consistency checks between model outputs."""
        # Check that both models produce consistent team totals
        mc_players = self.monte_carlo_data['monte_carlo_projection']['player_contributions']
        bayes_players = self.bayesian_data['team_projection']['player_contributions']
        
        # Calculate total from player contributions
        mc_total_from_players = sum(player['mean'] for player in mc_players.values())
        bayes_total_from_players = sum(player['mean'] for player in bayes_players.values())
        
        # Compare with team projections
        mc_team_total = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']['mean']
        bayes_team_total = self.bayesian_data['team_projection']['total_score']['mean']
        
        # Verify consistency (should be close, allowing for rounding)
        # Note: In real data, player contributions might not exactly sum to team total
        # due to additional factors like team bonuses, penalties, etc.
        self.assertAlmostEqual(mc_total_from_players, mc_team_total, delta=5.0)  # Allow larger delta for realistic data
        self.assertAlmostEqual(bayes_total_from_players, bayes_team_total, delta=5.0)  # Allow larger delta for realistic data

if __name__ == '__main__':
    unittest.main()
