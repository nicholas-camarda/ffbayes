"""
Tests for advanced draft strategy functionality.

This module tests the tier-based Bayesian draft strategy implementation,
including strategy generation, team construction optimization, and
uncertainty-aware decision making.
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
    TeamConstructionOptimizer,
    TierBasedStrategy,
    UncertaintyAwareSelector,
)


class TestBayesianDraftStrategy(unittest.TestCase):
    """Test the main Bayesian draft strategy class."""
    
    def setUp(self):
        """Set up test data and mocks."""
        self.mock_bayesian_predictions = pd.DataFrame({
            'player_name': ['McCaffrey', 'Barkley', 'Hill', 'Allen', 'Kelce', 'Ekeler', 'Diggs', 'Mahomes', 'Henry', 'Adams'],
            'position': ['RB', 'RB', 'WR', 'QB', 'TE', 'RB', 'WR', 'QB', 'RB', 'WR'],
            'predicted_points': [280.5, 265.2, 245.8, 320.1, 185.3, 240.0, 235.0, 315.0, 230.0, 225.0],
            'confidence_interval_lower': [250.0, 240.0, 220.0, 290.0, 160.0, 210.0, 205.0, 285.0, 200.0, 195.0],
            'confidence_interval_upper': [310.0, 290.0, 270.0, 350.0, 210.0, 270.0, 265.0, 345.0, 260.0, 255.0],
            'uncertainty_score': [0.15, 0.18, 0.20, 0.12, 0.25, 0.22, 0.19, 0.14, 0.21, 0.23]
        })
        
        self.mock_monte_carlo_results = pd.DataFrame({
            'team_id': [1, 2, 3],
            'team_name': ['Team A', 'Team B', 'Team C'],
            'projected_total': [1850.5, 1780.2, 1920.8],
            'confidence_interval_lower': [1750.0, 1680.0, 1820.0],
            'confidence_interval_upper': [1950.0, 1880.0, 2020.0]
        })
        
        self.draft_config = DraftConfig(
            league_size=12,
            draft_position=3,
            scoring_type='PPR',
            roster_positions={
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1
            }
        )
    
    def test_bayesian_draft_strategy_initialization(self):
        """Test that BayesianDraftStrategy initializes correctly."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.mock_bayesian_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.league_size, 12)
        self.assertEqual(strategy.draft_position, 3)
        self.assertEqual(strategy.scoring_type, 'PPR')
    
    def test_bayesian_draft_strategy_with_invalid_config(self):
        """Test that BayesianDraftStrategy handles invalid config gracefully."""
        with self.assertRaises(ValueError):
            invalid_config = DraftConfig(
                league_size=0,  # Invalid league size
                draft_position=15,  # Invalid draft position
                scoring_type='INVALID',
                roster_positions={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
            )
    
    def test_generate_draft_strategy(self):
        """Test that draft strategy generation works correctly."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.mock_bayesian_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        draft_strategy = strategy.generate_draft_strategy()
        
        # Check that strategy is generated
        self.assertIsInstance(draft_strategy, dict)
        self.assertIn('Pick 3', draft_strategy)
        
        # Check structure of strategy for first pick
        pick_3_strategy = draft_strategy['Pick 3']
        self.assertIn('primary_targets', pick_3_strategy)
        self.assertIn('backup_options', pick_3_strategy)
        self.assertIn('fallback_options', pick_3_strategy)
        self.assertIn('position_priority', pick_3_strategy)
        self.assertIn('reasoning', pick_3_strategy)
        
        # Check that we have multiple options per pick
        self.assertGreaterEqual(len(pick_3_strategy['primary_targets']), 1)
        self.assertGreaterEqual(len(pick_3_strategy['backup_options']), 1)
        self.assertGreaterEqual(len(pick_3_strategy['fallback_options']), 1)


class TestTierBasedStrategy(unittest.TestCase):
    """Test the tier-based strategy generation."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_predictions = pd.DataFrame({
            'player_name': ['McCaffrey', 'Barkley', 'Hill', 'Allen', 'Kelce', 'Ekeler', 'Diggs', 'Mahomes'],
            'position': ['RB', 'RB', 'WR', 'QB', 'TE', 'RB', 'WR', 'QB'],
            'predicted_points': [280.5, 265.2, 245.8, 320.1, 185.3, 240.0, 235.0, 315.0],
            'uncertainty_score': [0.15, 0.18, 0.20, 0.12, 0.25, 0.22, 0.19, 0.14]
        })
    
    def test_create_tiers(self):
        """Test that tiers are created correctly."""
        tier_strategy = TierBasedStrategy(self.mock_predictions)
        tiers = tier_strategy.create_tiers()
        
        # Check that tiers are created
        self.assertIsInstance(tiers, dict)
        self.assertIn('Tier 1', tiers)
        self.assertIn('Tier 2', tiers)
        
        # Check that players are assigned to tiers
        tier_1_players = tiers['Tier 1']
        self.assertGreater(len(tier_1_players), 0)
        
        # Check that higher predicted points are in higher tiers
        tier_1_points = [self.mock_predictions[
            self.mock_predictions['player_name'] == player
        ]['predicted_points'].iloc[0] for player in tier_1_players]
        
        tier_2_points = [self.mock_predictions[
            self.mock_predictions['player_name'] == player
        ]['predicted_points'].iloc[0] for player in tiers['Tier 2']]
        
        # Tier 1 should have higher points than Tier 2
        self.assertGreater(min(tier_1_points), max(tier_2_points))
    
    def test_generate_pick_options(self):
        """Test that pick options are generated correctly."""
        tier_strategy = TierBasedStrategy(self.mock_predictions)
        config = DraftConfig(
            league_size=12,
            draft_position=3,
            scoring_type='PPR',
            roster_positions={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
        )
        pick_options = tier_strategy.generate_pick_options(draft_position=3, league_size=12, config=config)
        
        # Check structure
        self.assertIn('primary_targets', pick_options)
        self.assertIn('backup_options', pick_options)
        self.assertIn('fallback_options', pick_options)
        
        # Check that we have multiple options
        self.assertGreaterEqual(len(pick_options['primary_targets']), 1)
        self.assertGreaterEqual(len(pick_options['backup_options']), 1)
        self.assertGreaterEqual(len(pick_options['fallback_options']), 1)
        
        # Check that all options are valid players
        all_options = (pick_options['primary_targets'] + 
                      pick_options['backup_options'] + 
                      pick_options['fallback_options'])
        
        for player in all_options:
            self.assertIn(player, self.mock_predictions['player_name'].values)


class TestTeamConstructionOptimizer(unittest.TestCase):
    """Test the team construction optimization."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_predictions = pd.DataFrame({
            'player_name': ['McCaffrey', 'Barkley', 'Hill', 'Allen', 'Kelce', 'Diggs', 'Adams'],
            'position': ['RB', 'RB', 'WR', 'QB', 'TE', 'WR', 'WR'],
            'predicted_points': [280.5, 265.2, 245.8, 320.1, 185.3, 235.0, 225.0],
            'uncertainty_score': [0.15, 0.18, 0.20, 0.12, 0.25, 0.19, 0.23]
        })
        
        self.mock_monte_carlo_results = pd.DataFrame({
            'team_id': [1, 2, 3],
            'team_name': ['Team A', 'Team B', 'Team C'],
            'projected_total': [1850.5, 1780.2, 1920.8]
        })
        
        self.roster_requirements = {
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1
        }
    
    def test_optimize_team_construction(self):
        """Test that team construction optimization works."""
        optimizer = TeamConstructionOptimizer(
            predictions=self.mock_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            roster_requirements=self.roster_requirements
        )
        
        optimal_team = optimizer.optimize_team_construction(draft_position=3)
        
        # Check that optimal team is generated
        self.assertIsInstance(optimal_team, dict)
        self.assertIn('optimal_team', optimal_team)
        self.assertIn('team_projection', optimal_team)
        self.assertIn('uncertainty_analysis', optimal_team)
        
        team_data = optimal_team['optimal_team']
        self.assertIn('QB', team_data)
        self.assertIn('RB', team_data)
        self.assertIn('WR', team_data)
        self.assertIn('TE', team_data)
        self.assertIn('FLEX', team_data)
        
        # Check that roster requirements are met
        self.assertEqual(len(team_data['QB']), 1)
        self.assertEqual(len(team_data['RB']), 2)
        self.assertEqual(len(team_data['WR']), 2)
        self.assertEqual(len(team_data['TE']), 1)
        self.assertEqual(len(team_data['FLEX']), 1)
    
    def test_position_scarcity_analysis(self):
        """Test position scarcity analysis."""
        optimizer = TeamConstructionOptimizer(
            predictions=self.mock_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            roster_requirements=self.roster_requirements
        )
        
        scarcity_analysis = optimizer.analyze_position_scarcity()
        
        # Check that scarcity analysis is performed
        self.assertIsInstance(scarcity_analysis, dict)
        self.assertIn('RB', scarcity_analysis)
        self.assertIn('WR', scarcity_analysis)
        self.assertIn('QB', scarcity_analysis)
        self.assertIn('TE', scarcity_analysis)
        
        # Check that scarcity analysis is performed
        for position, analysis in scarcity_analysis.items():
            self.assertIsInstance(analysis, dict)
            self.assertIn('scarcity_score', analysis)
            self.assertIn('total_players', analysis)
            self.assertIn('avg_points', analysis)
            self.assertIn('hierarchical_insights', analysis)
            self.assertIn('quality_distribution', analysis)
            self.assertIn('uncertainty_profile', analysis)


class TestUncertaintyAwareSelector(unittest.TestCase):
    """Test the uncertainty-aware player selection."""
    
    def setUp(self):
        """Set up test data."""
        self.mock_predictions = pd.DataFrame({
            'player_name': ['McCaffrey', 'Barkley', 'Hill', 'Allen', 'Kelce'],
            'position': ['RB', 'RB', 'WR', 'QB', 'TE'],
            'predicted_points': [280.5, 265.2, 245.8, 320.1, 185.3],
            'confidence_interval_lower': [250.0, 240.0, 220.0, 290.0, 160.0],
            'confidence_interval_upper': [310.0, 290.0, 270.0, 350.0, 210.0],
            'uncertainty_score': [0.15, 0.18, 0.20, 0.12, 0.25]
        })
    
    def test_risk_adjusted_selection(self):
        """Test risk-adjusted player selection."""
        selector = UncertaintyAwareSelector(self.mock_predictions)
        
        # Test conservative selection (low risk tolerance)
        conservative_picks = selector.select_players(
            position='RB', 
            count=2, 
            risk_tolerance='low'
        )
        
        self.assertEqual(len(conservative_picks), 2)
        
        # Test aggressive selection (high risk tolerance)
        aggressive_picks = selector.select_players(
            position='RB', 
            count=2, 
            risk_tolerance='high'
        )
        
        self.assertEqual(len(aggressive_picks), 2)
        
        # Conservative picks should have lower uncertainty than aggressive picks
        conservative_uncertainty = self.mock_predictions[
            self.mock_predictions['player_name'].isin(conservative_picks)
        ]['uncertainty_score'].mean()
        
        aggressive_uncertainty = self.mock_predictions[
            self.mock_predictions['player_name'].isin(aggressive_picks)
        ]['uncertainty_score'].mean()
        
        self.assertLessEqual(conservative_uncertainty, aggressive_uncertainty)
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification methods."""
        selector = UncertaintyAwareSelector(self.mock_predictions)
        
        uncertainty_metrics = selector.calculate_uncertainty_metrics()
        
        # Check that uncertainty metrics are calculated
        self.assertIsInstance(uncertainty_metrics, dict)
        self.assertIn('overall_uncertainty', uncertainty_metrics)
        self.assertIn('position_uncertainty', uncertainty_metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(uncertainty_metrics['overall_uncertainty'], 0)
        self.assertLessEqual(uncertainty_metrics['overall_uncertainty'], 1)


class TestDraftStrategyIntegration(unittest.TestCase):
    """Test integration of all draft strategy components."""
    
    def setUp(self):
        """Set up comprehensive test data."""
        self.mock_bayesian_predictions = pd.DataFrame({
            'player_name': ['McCaffrey', 'Barkley', 'Hill', 'Allen', 'Kelce', 'Ekeler', 'Diggs', 'Mahomes', 'Henry', 'Adams'],
            'position': ['RB', 'RB', 'WR', 'QB', 'TE', 'RB', 'WR', 'QB', 'RB', 'WR'],
            'predicted_points': [280.5, 265.2, 245.8, 320.1, 185.3, 240.0, 235.0, 315.0, 230.0, 225.0],
            'confidence_interval_lower': [250.0, 240.0, 220.0, 290.0, 160.0, 210.0, 205.0, 285.0, 200.0, 195.0],
            'confidence_interval_upper': [310.0, 290.0, 270.0, 350.0, 210.0, 270.0, 265.0, 345.0, 260.0, 255.0],
            'uncertainty_score': [0.15, 0.18, 0.20, 0.12, 0.25, 0.22, 0.19, 0.14, 0.21, 0.23]
        })
        
        self.mock_monte_carlo_results = pd.DataFrame({
            'team_id': [1, 2, 3],
            'team_name': ['Team A', 'Team B', 'Team C'],
            'projected_total': [1850.5, 1780.2, 1920.8],
            'confidence_interval_lower': [1750.0, 1680.0, 1820.0],
            'confidence_interval_upper': [1950.0, 1880.0, 2020.0]
        })
        
        self.draft_config = DraftConfig(
            league_size=12,
            draft_position=3,
            scoring_type='PPR',
            roster_positions={
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1
            }
        )
    
    def test_end_to_end_strategy_generation(self):
        """Test complete end-to-end strategy generation."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.mock_bayesian_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        # Generate complete draft strategy
        complete_strategy = strategy.generate_complete_draft_strategy()
        
        # Check that complete strategy is generated
        self.assertIsInstance(complete_strategy, dict)
        
        # Check that strategy covers all picks
        strategy_data = complete_strategy['strategy']
        expected_picks = [f"Pick {i}" for i in range(3, 13, 10)]  # 3rd pick, 13th pick, etc.
        for pick in expected_picks:
            self.assertIn(pick, strategy_data)
        
        # Check that each pick has proper structure
        for pick_num, pick_strategy in strategy_data.items():
            self.assertIn('primary_targets', pick_strategy)
            self.assertIn('backup_options', pick_strategy)
            self.assertIn('fallback_options', pick_strategy)
            self.assertIn('position_priority', pick_strategy)
            self.assertIn('reasoning', pick_strategy)
            
            # Check that we have multiple options per pick
            total_options = (len(pick_strategy['primary_targets']) + 
                           len(pick_strategy['backup_options']) + 
                           len(pick_strategy['fallback_options']))
            self.assertGreaterEqual(total_options, 10)  # At least 10 options per pick
    
    def test_strategy_validation(self):
        """Test that generated strategy is valid."""
        strategy = BayesianDraftStrategy(
            bayesian_predictions=self.mock_bayesian_predictions,
            monte_carlo_results=self.mock_monte_carlo_results,
            draft_config=self.draft_config
        )
        
        draft_strategy = strategy.generate_draft_strategy()
        
        # Validate strategy
        validation_result = strategy.validate_strategy(draft_strategy)
        
        # Check that validation passes
        self.assertTrue(validation_result['is_valid'])
        self.assertIn('warnings', validation_result)
        self.assertIn('recommendations', validation_result)


if __name__ == '__main__':
    unittest.main()
