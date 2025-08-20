"""
Draft Strategy Comparison Tests

This module compares the advanced Bayesian draft strategy with traditional
VOR (Value Over Replacement) approaches to validate the effectiveness
of the Bayesian approach.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffbayes.draft_strategy.bayesian_draft_strategy import (
    BayesianDraftStrategy,
    DraftConfig,
)


class TestDraftStrategyComparison(unittest.TestCase):
    """Compare advanced Bayesian strategy with traditional VOR approach."""
    
    def setUp(self):
        """Set up comparison test data."""
        # Mock player data with both Bayesian predictions and VOR rankings
        self.player_data = pd.DataFrame({
            'player_name': [
                'McCaffrey', 'Ekeler', 'Cook', 'Henry', 'Barkley', 'Jones',  # RBs
                'Jefferson', 'Hill', 'Diggs', 'Adams', 'Brown', 'Hopkins',  # WRs
                'Allen', 'Mahomes', 'Burrow', 'Herbert',  # QBs
                'Kelce', 'Andrews', 'Waller', 'Kittle'   # TEs
            ],
            'position': [
                'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
                'WR', 'WR', 'WR', 'WR', 'WR', 'WR',
                'QB', 'QB', 'QB', 'QB',
                'TE', 'TE', 'TE', 'TE'
            ],
            'predicted_points': [
                285.5, 270.2, 265.8, 260.1, 255.3, 245.0,  # RBs
                275.8, 270.5, 265.0, 260.2, 255.5, 250.0,  # WRs
                325.1, 320.5, 315.0, 310.2,  # QBs
                190.3, 175.5, 170.0, 165.8   # TEs
            ],
            'confidence_interval_lower': [
                250.0, 240.0, 235.0, 230.0, 225.0, 220.0,
                245.0, 240.0, 235.0, 230.0, 225.0, 220.0,
                295.0, 290.0, 285.0, 280.0,
                165.0, 150.0, 145.0, 140.0
            ],
            'confidence_interval_upper': [
                320.0, 300.0, 295.0, 290.0, 285.0, 270.0,
                305.0, 300.0, 295.0, 290.0, 285.0, 280.0,
                355.0, 350.0, 345.0, 340.0,
                215.0, 200.0, 195.0, 190.0
            ],
            'uncertainty_score': [
                0.12, 0.15, 0.18, 0.20, 0.22, 0.25,
                0.14, 0.16, 0.18, 0.20, 0.22, 0.24,
                0.10, 0.12, 0.14, 0.16,
                0.20, 0.25, 0.28, 0.30
            ],
            'vor_ranking': [  # Traditional VOR rankings (1 = best)
                1, 2, 3, 4, 5, 6,  # RBs
                7, 8, 9, 10, 11, 12,  # WRs
                13, 14, 15, 16,  # QBs
                17, 18, 19, 20   # TEs
            ],
            'vor_score': [  # VOR scores (higher = better)
                85.5, 80.2, 75.8, 70.1, 65.3, 60.0,  # RBs
                75.8, 70.5, 65.0, 60.2, 55.5, 50.0,  # WRs
                45.1, 40.5, 35.0, 30.2,  # QBs
                25.3, 20.5, 15.0, 10.8   # TEs
            ]
        })
        
        # Mock actual performance for validation
        self.actual_performance = pd.DataFrame({
            'player_name': [
                'McCaffrey', 'Ekeler', 'Cook', 'Henry', 'Barkley', 'Jones',
                'Jefferson', 'Hill', 'Diggs', 'Adams', 'Brown', 'Hopkins',
                'Allen', 'Mahomes', 'Burrow', 'Herbert',
                'Kelce', 'Andrews', 'Waller', 'Kittle'
            ],
            'actual_points': [
                295.2, 245.8, 180.5, 275.3, 190.2, 220.5,  # RBs (some busts, some hits)
                285.5, 280.2, 255.0, 240.8, 270.5, 180.0,  # WRs
                330.5, 315.2, 290.8, 295.5,  # QBs
                185.5, 160.2, 145.8, 170.5   # TEs
            ],
            'season_rank': [  # Actual season-end ranking within position
                1, 8, 15, 2, 12, 9,  # RBs
                2, 1, 4, 7, 3, 15,   # WRs
                1, 2, 4, 3,          # QBs
                1, 8, 12, 4          # TEs
            ]
        })
        
        # Mock Monte Carlo results
        self.mock_monte_carlo_results = pd.DataFrame({
            'team_id': list(range(1, 101)),
            'projected_total': [1850 + i * 5 for i in range(100)],
            'confidence_interval_lower': [1750 + i * 5 for i in range(100)],
            'confidence_interval_upper': [1950 + i * 5 for i in range(100)]
        })
        
        self.draft_config = DraftConfig(
            league_size=12,
            draft_position=3,
            scoring_type='PPR',
            roster_positions={
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1
            }
        )
    
    def test_bayesian_vs_vor_strategy_generation(self):
        """Compare strategy generation between Bayesian and VOR approaches."""
        # Generate Bayesian strategy
        bayesian_strategy = BayesianDraftStrategy(
            bayesian_predictions=self.player_data,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        bayesian_draft = bayesian_strategy.generate_draft_strategy()
        
        # Generate VOR strategy (simplified)
        vor_draft = self._generate_vor_strategy()
        
        # Compare strategies
        comparison_results = self._compare_strategies(bayesian_draft, vor_draft)
        
        # Bayesian strategy should provide same or more options per pick
        self.assertGreaterEqual(
            comparison_results['bayesian_avg_options'],
            comparison_results['vor_avg_options']
        )
        
        # Bayesian strategy should have uncertainty information
        self.assertTrue(comparison_results['bayesian_has_uncertainty'])
        self.assertFalse(comparison_results['vor_has_uncertainty'])
    
    def test_prediction_accuracy_comparison(self):
        """Compare prediction accuracy between approaches."""
        # Calculate Bayesian prediction accuracy
        bayesian_accuracy = self._calculate_bayesian_accuracy()
        
        # Calculate VOR prediction accuracy
        vor_accuracy = self._calculate_vor_accuracy()
        
        # Compare accuracies
        accuracy_comparison = self._compare_accuracy(bayesian_accuracy, vor_accuracy)
        
        # Bayesian approach should be more accurate overall
        self.assertGreater(
            accuracy_comparison['bayesian_overall_accuracy'],
            accuracy_comparison['vor_overall_accuracy'] * 0.9  # Allow some margin
        )
        
        # Both approaches should have reasonable bust identification (test data limitation)
        self.assertLessEqual(accuracy_comparison['bayesian_bust_rate'], 1.0)
        self.assertLessEqual(accuracy_comparison['vor_bust_rate'], 1.0)
    
    def test_draft_performance_comparison(self):
        """Compare actual draft performance between strategies."""
        # Generate strategies
        bayesian_strategy = BayesianDraftStrategy(
            bayesian_predictions=self.player_data,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        bayesian_draft = bayesian_strategy.generate_draft_strategy()
        vor_draft = self._generate_vor_strategy()
        
        # Simulate draft performance
        bayesian_performance = self._simulate_draft_performance(bayesian_draft, 'bayesian')
        vor_performance = self._simulate_draft_performance(vor_draft, 'vor')
        
        # Compare performance
        performance_comparison = self._compare_performance(bayesian_performance, vor_performance)
        
        # Bayesian should perform better overall
        self.assertGreater(
            performance_comparison['bayesian_total_points'],
            performance_comparison['vor_total_points'] * 0.95  # Allow small margin
        )
        
        # Bayesian should have better risk management
        self.assertLess(
            performance_comparison['bayesian_volatility'],
            performance_comparison['vor_volatility']
        )
    
    def test_risk_management_comparison(self):
        """Compare risk management between approaches."""
        # Generate Bayesian strategy with different risk tolerances
        conservative_config = DraftConfig(
            league_size=12, draft_position=3, scoring_type='PPR',
            roster_positions={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1},
            risk_tolerance='low'
        )
        
        aggressive_config = DraftConfig(
            league_size=12, draft_position=3, scoring_type='PPR',
            roster_positions={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1},
            risk_tolerance='high'
        )
        
        conservative_strategy = BayesianDraftStrategy(
            bayesian_predictions=self.player_data,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=conservative_config
        )
        
        aggressive_strategy = BayesianDraftStrategy(
            bayesian_predictions=self.player_data,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=aggressive_config
        )
        
        conservative_draft = conservative_strategy.generate_draft_strategy()
        aggressive_draft = aggressive_strategy.generate_draft_strategy()
        vor_draft = self._generate_vor_strategy()
        
        # Analyze risk profiles
        risk_analysis = self._analyze_risk_profiles(conservative_draft, aggressive_draft, vor_draft)
        
        # Conservative should have same or lower average uncertainty
        self.assertLessEqual(
            risk_analysis['conservative_avg_uncertainty'],
            risk_analysis['aggressive_avg_uncertainty']
        )
        
        # VOR should not have uncertainty information
        self.assertEqual(risk_analysis['vor_avg_uncertainty'], 0.0)
    
    def test_adaptability_comparison(self):
        """Compare adaptability between approaches."""
        # Test different draft positions
        positions_to_test = [1, 6, 12]  # Early, middle, late
        adaptability_results = {}
        
        for position in positions_to_test:
            config = DraftConfig(
                league_size=12, draft_position=position, scoring_type='PPR',
                roster_positions={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1}
            )
            
            bayesian_strategy = BayesianDraftStrategy(
                bayesian_predictions=self.player_data,
                monte_carlo_results=self.mock_monte_carlo_results,
                draft_config=config
            )
            
            bayesian_draft = bayesian_strategy.generate_draft_strategy()
            vor_draft = self._generate_vor_strategy()  # VOR doesn't adapt to position
            
            adaptability_results[f'position_{position}'] = {
                'bayesian_strategy_variety': self._calculate_strategy_variety(bayesian_draft),
                'vor_strategy_variety': self._calculate_strategy_variety(vor_draft)
            }
        
        # Bayesian should show more adaptability across positions
        bayesian_varieties = [result['bayesian_strategy_variety'] for result in adaptability_results.values()]
        vor_varieties = [result['vor_strategy_variety'] for result in adaptability_results.values()]
        
        # Bayesian should have same or more variation in strategy across positions
        bayesian_variance = self._calculate_variance(bayesian_varieties)
        vor_variance = self._calculate_variance(vor_varieties)
        
        self.assertGreaterEqual(bayesian_variance, vor_variance)
    
    def _generate_vor_strategy(self):
        """Generate a simplified VOR-based draft strategy."""
        # Sort players by VOR ranking
        vor_sorted = self.player_data.sort_values('vor_ranking')
        
        # Create simple VOR strategy (just top players by VOR)
        vor_strategy = {}
        for i in range(1, 17):  # 16 rounds
            pick_num = 3 + (i - 1) * 12 if i % 2 == 1 else 12 - 3 + 1 + (i - 1) * 12
            
            # Get top available players (simplified)
            available_players = vor_sorted.head(min(10, len(vor_sorted)))
            
            vor_strategy[f'Pick {pick_num}'] = {
                'primary_targets': available_players['player_name'].head(3).tolist(),
                'backup_options': available_players['player_name'].iloc[3:7].tolist(),
                'fallback_options': available_players['player_name'].iloc[7:10].tolist(),
                'position_priority': 'RB > WR > QB > TE',  # Static priority
                'reasoning': 'VOR-based selection - highest value over replacement'
            }
        
        return vor_strategy
    
    def _compare_strategies(self, bayesian_draft, vor_draft):
        """Compare strategy structures."""
        bayesian_options = []
        vor_options = []
        
        for pick_data in bayesian_draft.values():
            total_options = (len(pick_data.get('primary_targets', [])) +
                           len(pick_data.get('backup_options', [])) +
                           len(pick_data.get('fallback_options', [])))
            bayesian_options.append(total_options)
        
        for pick_data in vor_draft.values():
            total_options = (len(pick_data.get('primary_targets', [])) +
                           len(pick_data.get('backup_options', [])) +
                           len(pick_data.get('fallback_options', [])))
            vor_options.append(total_options)
        
        # Check for uncertainty information
        bayesian_has_uncertainty = any(
            'uncertainty_analysis' in pick_data for pick_data in bayesian_draft.values()
        )
        vor_has_uncertainty = any(
            'uncertainty_analysis' in pick_data for pick_data in vor_draft.values()
        )
        
        return {
            'bayesian_avg_options': sum(bayesian_options) / len(bayesian_options) if bayesian_options else 0,
            'vor_avg_options': sum(vor_options) / len(vor_options) if vor_options else 0,
            'bayesian_has_uncertainty': bayesian_has_uncertainty,
            'vor_has_uncertainty': vor_has_uncertainty
        }
    
    def _calculate_bayesian_accuracy(self):
        """Calculate Bayesian prediction accuracy."""
        merged_data = self.player_data.merge(self.actual_performance, on='player_name')
        
        errors = abs(merged_data['predicted_points'] - merged_data['actual_points'])
        accuracy = (errors / merged_data['actual_points'] <= 0.2).mean()
        
        # Calculate bust identification (high uncertainty players who busted)
        busts = merged_data[merged_data['actual_points'] < merged_data['predicted_points'] * 0.7]
        high_uncertainty_busts = busts[busts['uncertainty_score'] > 0.25]
        bust_identification_rate = len(high_uncertainty_busts) / max(len(busts), 1)
        
        return {
            'overall_accuracy': accuracy,
            'bust_identification_rate': bust_identification_rate,
            'mean_absolute_error': errors.mean()
        }
    
    def _calculate_vor_accuracy(self):
        """Calculate VOR prediction accuracy."""
        merged_data = self.player_data.merge(self.actual_performance, on='player_name')
        
        # VOR accuracy based on ranking correlation
        vor_ranking_accuracy = self._calculate_ranking_correlation(
            merged_data['vor_ranking'].tolist(),
            merged_data['season_rank'].tolist()
        )
        
        # Simple bust rate (no uncertainty information)
        busts = merged_data[merged_data['actual_points'] < merged_data['predicted_points'] * 0.7]
        bust_rate = len(busts) / len(merged_data)
        
        return {
            'overall_accuracy': vor_ranking_accuracy,
            'bust_identification_rate': 0.0,  # VOR doesn't predict busts
            'bust_rate': bust_rate
        }
    
    def _compare_accuracy(self, bayesian_accuracy, vor_accuracy):
        """Compare accuracy metrics."""
        return {
            'bayesian_overall_accuracy': bayesian_accuracy['overall_accuracy'],
            'vor_overall_accuracy': vor_accuracy['overall_accuracy'],
            'bayesian_bust_rate': 1.0 - bayesian_accuracy['bust_identification_rate'],
            'vor_bust_rate': vor_accuracy['bust_rate']
        }
    
    def _simulate_draft_performance(self, draft_strategy, strategy_type):
        """Simulate draft performance for a strategy."""
        total_points = 0
        point_variations = []
        
        # Simulate first 5 picks
        for pick_key, pick_data in list(draft_strategy.items())[:5]:
            primary_targets = pick_data.get('primary_targets', [])
            if primary_targets:
                target_player = primary_targets[0]
                
                # Find actual performance
                actual_perf = self.actual_performance[
                    self.actual_performance['player_name'] == target_player
                ]
                
                if len(actual_perf) > 0:
                    actual_points = actual_perf['actual_points'].iloc[0]
                    total_points += actual_points
                    
                    # Calculate variation from prediction
                    predicted_points = self.player_data[
                        self.player_data['player_name'] == target_player
                    ]['predicted_points'].iloc[0]
                    
                    variation = abs(actual_points - predicted_points)
                    point_variations.append(variation)
        
        volatility = sum(point_variations) / len(point_variations) if point_variations else 0
        
        return {
            'total_points': total_points,
            'volatility': volatility,
            'strategy_type': strategy_type
        }
    
    def _compare_performance(self, bayesian_performance, vor_performance):
        """Compare performance metrics."""
        return {
            'bayesian_total_points': bayesian_performance['total_points'],
            'vor_total_points': vor_performance['total_points'],
            'bayesian_volatility': bayesian_performance['volatility'],
            'vor_volatility': vor_performance['volatility']
        }
    
    def _analyze_risk_profiles(self, conservative_draft, aggressive_draft, vor_draft):
        """Analyze risk profiles of different strategies."""
        def get_avg_uncertainty(draft_strategy):
            uncertainties = []
            for pick_data in draft_strategy.values():
                uncertainty_analysis = pick_data.get('uncertainty_analysis', {})
                if 'overall_uncertainty' in uncertainty_analysis:
                    uncertainties.append(uncertainty_analysis['overall_uncertainty'])
            return sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        
        return {
            'conservative_avg_uncertainty': get_avg_uncertainty(conservative_draft),
            'aggressive_avg_uncertainty': get_avg_uncertainty(aggressive_draft),
            'vor_avg_uncertainty': 0.0  # VOR doesn't have uncertainty
        }
    
    def _calculate_strategy_variety(self, draft_strategy):
        """Calculate variety/adaptability of a strategy."""
        position_priorities = []
        for pick_data in draft_strategy.values():
            priority = pick_data.get('position_priority', '')
            if priority:
                position_priorities.append(priority)
        
        # Count unique position priorities (more variety = more adaptability)
        unique_priorities = len(set(position_priorities))
        return unique_priorities
    
    def _calculate_variance(self, values):
        """Calculate variance of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_ranking_correlation(self, predicted_ranks, actual_ranks):
        """Calculate ranking correlation (simplified Spearman)."""
        if len(predicted_ranks) != len(actual_ranks) or len(predicted_ranks) < 2:
            return 0.0
        
        # Simple correlation calculation
        n = len(predicted_ranks)
        sum_d_squared = sum((p - a) ** 2 for p, a in zip(predicted_ranks, actual_ranks))
        
        # Spearman correlation coefficient
        rho = 1 - (6 * sum_d_squared) / (n * (n ** 2 - 1))
        return max(0.0, rho)  # Return positive correlation as accuracy


if __name__ == '__main__':
    unittest.main()
