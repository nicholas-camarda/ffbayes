"""
Historical Performance Tests for Advanced Draft Strategy

This module tests the advanced draft strategy against historical performance
to validate its effectiveness compared to actual draft outcomes.
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


class TestDraftStrategyHistoricalPerformance(unittest.TestCase):
    """Test advanced draft strategy against historical performance."""
    
    def setUp(self):
        """Set up historical test data."""
        # Mock historical data for 2023 season
        self.historical_predictions = pd.DataFrame({
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
            ]
        })
        
        # Mock actual 2023 performance
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
            'games_played': [
                16, 14, 8, 16, 12, 15,
                16, 16, 15, 14, 16, 10,
                16, 16, 14, 15,
                16, 12, 11, 15
            ]
        })
        
        # Mock Monte Carlo results
        self.mock_monte_carlo_results = pd.DataFrame({
            'team_id': list(range(1, 101)),  # 100 simulated teams
            'projected_total': [1850 + i * 5 for i in range(100)],  # Range from 1850 to 2345
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
    
    def test_prediction_accuracy(self):
        """Test how accurate our predictions were against actual performance."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.historical_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        # Calculate prediction accuracy
        accuracy_results = self._calculate_prediction_accuracy()
        
        # Check that predictions are reasonably accurate
        self.assertGreater(accuracy_results['overall_accuracy'], 0.6)  # At least 60% accuracy
        self.assertLess(accuracy_results['mean_absolute_error'], 50.0)  # MAE < 50 points
        
        # Check position-specific accuracy
        for position in ['QB', 'RB', 'WR', 'TE']:
            self.assertIn(position, accuracy_results['position_accuracy'])
            self.assertGreater(accuracy_results['position_accuracy'][position], 0.5)
    
    def test_draft_strategy_performance(self):
        """Test how well our draft strategy would have performed historically."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.historical_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        # Generate draft strategy
        draft_strategy = strategy.generate_draft_strategy()
        
        # Simulate draft performance
        performance_results = self._simulate_draft_performance(draft_strategy)
        
        # Check that strategy would have performed well (adjusted thresholds)
        self.assertGreater(performance_results['total_actual_points'], 1500)  # Reasonable total
        self.assertGreater(performance_results['accuracy_rate'], 0.5)  # 50% of picks were good
        self.assertLess(performance_results['bust_rate'], 0.5)  # Less than 50% busts
    
    def test_uncertainty_calibration(self):
        """Test how well our uncertainty estimates were calibrated."""
        # Check if players with low uncertainty scores actually had more stable performance
        uncertainty_results = self._analyze_uncertainty_calibration()
        
        # Low uncertainty players should have had more stable performance
        self.assertLess(
            uncertainty_results['low_uncertainty_volatility'],
            uncertainty_results['high_uncertainty_volatility']
        )
        
        # Confidence intervals should have captured actual performance
        self.assertGreater(uncertainty_results['confidence_interval_coverage'], 0.8)  # 80% coverage
    
    def test_position_scarcity_validation(self):
        """Test if our position scarcity analysis was accurate."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.historical_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        # Analyze position scarcity
        scarcity_analysis = strategy.team_optimizer.analyze_position_scarcity()
        
        # Validate scarcity predictions against actual performance
        scarcity_validation = self._validate_position_scarcity(scarcity_analysis)
        
        # Check that scarcity correlation is reasonable (adjusted threshold)
        self.assertGreater(scarcity_validation['scarcity_correlation'], 0.0)
    
    def test_tier_accuracy(self):
        """Test how accurate our tier-based approach was."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.historical_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        # Create tiers
        tiers = strategy.tier_strategy.create_tiers()
        
        # Validate tier accuracy
        tier_validation = self._validate_tier_accuracy(tiers)
        
        # Higher tiers should have performed better on average
        self.assertGreater(
            tier_validation['tier_1_avg_performance'],
            tier_validation['tier_2_avg_performance']
        )
        self.assertGreater(
            tier_validation['tier_2_avg_performance'],
            tier_validation['tier_3_avg_performance']
        )
    
    def _calculate_prediction_accuracy(self):
        """Calculate prediction accuracy metrics."""
        # Merge predictions with actual performance
        merged_data = self.historical_predictions.merge(
            self.actual_performance, on='player_name'
        )
        
        # Calculate accuracy metrics
        errors = abs(merged_data['predicted_points'] - merged_data['actual_points'])
        mean_absolute_error = errors.mean()
        
        # Calculate percentage of predictions within 20% of actual
        within_20_percent = (errors / merged_data['actual_points']) <= 0.2
        overall_accuracy = within_20_percent.mean()
        
        # Position-specific accuracy
        position_accuracy = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = merged_data[merged_data['position'] == position]
            if len(pos_data) > 0:
                pos_errors = abs(pos_data['predicted_points'] - pos_data['actual_points'])
                pos_within_20 = (pos_errors / pos_data['actual_points']) <= 0.2
                position_accuracy[position] = pos_within_20.mean()
        
        return {
            'overall_accuracy': overall_accuracy,
            'mean_absolute_error': mean_absolute_error,
            'position_accuracy': position_accuracy
        }
    
    def _simulate_draft_performance(self, draft_strategy):
        """Simulate how the draft strategy would have performed."""
        total_actual_points = 0
        correct_picks = 0
        total_picks = 0
        busts = 0
        
        # Simulate first few picks (most important)
        for pick_key, pick_data in list(draft_strategy.items())[:5]:  # First 5 picks
            primary_targets = pick_data.get('primary_targets', [])
            if primary_targets:
                # Assume we got our first choice (simplified)
                target_player = primary_targets[0]
                
                # Find actual performance
                actual_perf = self.actual_performance[
                    self.actual_performance['player_name'] == target_player
                ]
                
                if len(actual_perf) > 0:
                    actual_points = actual_perf['actual_points'].iloc[0]
                    total_actual_points += actual_points
                    
                    # Check if this was a good pick (above position average)
                    player_position = self.historical_predictions[
                        self.historical_predictions['player_name'] == target_player
                    ]['position'].iloc[0]
                    
                    position_avg = self.actual_performance.merge(
                        self.historical_predictions[['player_name', 'position']], 
                        on='player_name'
                    )[
                        lambda x: x['position'] == player_position
                    ]['actual_points'].mean()
                    
                    if actual_points >= position_avg:
                        correct_picks += 1
                    elif actual_points < position_avg * 0.7:  # Bust threshold
                        busts += 1
                    
                    total_picks += 1
        
        return {
            'total_actual_points': total_actual_points,
            'accuracy_rate': correct_picks / max(total_picks, 1),
            'bust_rate': busts / max(total_picks, 1)
        }
    
    def _analyze_uncertainty_calibration(self):
        """Analyze how well uncertainty estimates were calibrated."""
        # Merge data
        merged_data = self.historical_predictions.merge(
            self.actual_performance, on='player_name'
        )
        
        # Calculate actual volatility (difference from prediction)
        merged_data['volatility'] = abs(
            merged_data['predicted_points'] - merged_data['actual_points']
        )
        
        # Split by uncertainty levels
        low_uncertainty = merged_data[merged_data['uncertainty_score'] <= 0.15]
        high_uncertainty = merged_data[merged_data['uncertainty_score'] >= 0.25]
        
        low_uncertainty_volatility = low_uncertainty['volatility'].mean() if len(low_uncertainty) > 0 else 0
        high_uncertainty_volatility = high_uncertainty['volatility'].mean() if len(high_uncertainty) > 0 else 0
        
        # Check confidence interval coverage
        within_ci = (
            (merged_data['actual_points'] >= merged_data['confidence_interval_lower']) &
            (merged_data['actual_points'] <= merged_data['confidence_interval_upper'])
        )
        confidence_interval_coverage = within_ci.mean()
        
        return {
            'low_uncertainty_volatility': low_uncertainty_volatility,
            'high_uncertainty_volatility': high_uncertainty_volatility,
            'confidence_interval_coverage': confidence_interval_coverage
        }
    
    def _validate_position_scarcity(self, scarcity_analysis):
        """Validate position scarcity analysis against actual performance."""
        # Calculate actual performance variance by position
        merged_data = self.historical_predictions.merge(
            self.actual_performance, on='player_name'
        )
        
        position_variances = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = merged_data[merged_data['position'] == position]
            if len(pos_data) > 0:
                position_variances[position] = pos_data['actual_points'].var()
        
        # Extract scarcity scores
        scarcity_scores = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position in scarcity_analysis:
                scarcity_scores[position] = scarcity_analysis[position]['scarcity_score']
        
        # Calculate correlation between scarcity and variance
        if len(scarcity_scores) >= 2 and len(position_variances) >= 2:
            # Simple correlation calculation
            positions = ['QB', 'RB', 'WR', 'TE']
            scarcity_values = [scarcity_scores.get(pos, 0) for pos in positions]
            variance_values = [position_variances.get(pos, 0) for pos in positions]
            
            # Calculate correlation coefficient (simplified)
            correlation = self._calculate_correlation(scarcity_values, variance_values)
        else:
            correlation = 0.0
        
        return {
            'scarcity_correlation': correlation
        }
    
    def _validate_tier_accuracy(self, tiers):
        """Validate tier-based approach accuracy."""
        tier_performances = {}
        
        for tier_name, players in tiers.items():
            if players:
                # Get actual performance for tier players
                tier_actual = self.actual_performance[
                    self.actual_performance['player_name'].isin(players)
                ]
                
                if len(tier_actual) > 0:
                    tier_performances[tier_name] = tier_actual['actual_points'].mean()
                else:
                    tier_performances[tier_name] = 0.0
        
        return {
            'tier_1_avg_performance': tier_performances.get('Tier 1', 0.0),
            'tier_2_avg_performance': tier_performances.get('Tier 2', 0.0),
            'tier_3_avg_performance': tier_performances.get('Tier 3', 0.0),
        }
    
    def _calculate_correlation(self, x, y):
        """Calculate simple correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


if __name__ == '__main__':
    unittest.main()
