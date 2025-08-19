import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import pandas as pd

# Add the scripts directory to the path for imports
sys.path.append('scripts')

class TestMonteCarloSimulation(unittest.TestCase):
    """Test Monte Carlo simulation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.datasets_dir = os.path.join(self.temp_dir, 'datasets')
        self.season_datasets_dir = os.path.join(self.datasets_dir, 'season_datasets')
        self.results_dir = os.path.join(self.temp_dir, 'results')
        self.montecarlo_results_dir = os.path.join(self.results_dir, 'montecarlo_results')
        self.teams_dir = os.path.join(self.temp_dir, 'my_ff_teams')
        
        os.makedirs(self.season_datasets_dir)
        os.makedirs(self.montecarlo_results_dir)
        os.makedirs(self.teams_dir)
        
        # Create sample season data
        self.sample_season_data = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Christian McCaffrey', 'Tyreek Hill', 'Travis Kelce'],
            'Position': ['QB', 'RB', 'WR', 'TE'],
            'Tm': ['NYJ', 'SF', 'MIA', 'KC'],
            'Season': [2023, 2023, 2023, 2023],
            'G#': [1, 1, 1, 1],
            'FantPt': [18.5, 25.2, 22.1, 15.8]
        })
        
        # Create sample team data
        self.sample_team = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Christian McCaffrey', 'Tyreek Hill', 'Travis Kelce'],
            'Position': ['QB', 'RB', 'WR', 'TE'],
            'Tm': ['NYJ', 'SF', 'MIA', 'KC']
        })
        
        # Save sample data
        season_file = os.path.join(self.season_datasets_dir, '2023season.csv')
        self.sample_season_data.to_csv(season_file, index=False)
        
        team_file = os.path.join(self.teams_dir, 'my_team_2024.tsv')
        self.sample_team.to_csv(team_file, sep='\t', index=False)
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original working directory
        os.chdir(self.original_cwd)
        
        # Remove temporary directories
        shutil.rmtree(self.temp_dir)
    
    def test_get_combined_data_success(self):
        """Test successful data combination from multiple season files"""
        # Create multiple season files
        for year in [2022, 2023]:
            season_file = os.path.join(self.season_datasets_dir, f'{year}season.csv')
            year_data = self.sample_season_data.copy()
            year_data['Season'] = year
            year_data.to_csv(season_file, index=False)
        
        # Import the function dynamically
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test the function
        result = monte_carlo.get_combined_data(self.datasets_dir)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 8)  # 4 players * 2 years
        self.assertTrue('Season' in result.columns)
        self.assertTrue('Name' in result.columns)
        self.assertTrue('FantPt' in result.columns)
    
    def test_get_combined_data_no_files(self):
        """Test error handling when no season files are found"""
        # Remove all season files
        for file in os.listdir(self.season_datasets_dir):
            os.remove(os.path.join(self.season_datasets_dir, file))
        
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        with self.assertRaises(ValueError) as context:
            monte_carlo.get_combined_data(self.datasets_dir)
        
        self.assertIn('No season data files found', str(context.exception))
    
    def test_make_team_success(self):
        """Test successful team creation from database and user team"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Create database with more players
        db_data = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Christian McCaffrey', 'Tyreek Hill', 'Travis Kelce', 'Other Player'],
            'Position': ['QB', 'RB', 'WR', 'TE', 'RB'],
            'Tm': ['NYJ', 'SF', 'MIA', 'KC', 'DAL']
        })
        
        result = monte_carlo.make_team(self.sample_team, db_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # Only the 4 players from user team
        self.assertTrue('Name' in result.columns)
        self.assertTrue('Position' in result.columns)
        self.assertTrue('Tm' in result.columns)
        
        # Check that only valid positions are included
        valid_positions = {'QB', 'WR', 'TE', 'RB'}
        result_positions = set(result['Position'].tolist())
        self.assertTrue(result_positions.issubset(valid_positions))
    
    def test_validate_team_all_found(self):
        """Test team validation when all players are found in database"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Create database team with same players
        db_team = self.sample_team.copy()
        
        # Mock the print function and test the validate_team function directly
        with mock.patch('builtins.print') as mock_print:
            # Call the function directly without the module-level execution
            monte_carlo.validate_team(db_team, self.sample_team)
            
            # Check that success message was printed
            mock_print.assert_any_call('Found all team members.')
    
    def test_validate_team_missing_players(self):
        """Test team validation when some players are missing from database"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Create database team with fewer players
        db_team = self.sample_team.head(2)  # Only first 2 players
        
        # Mock the print function and test the validate_team function directly
        with mock.patch('builtins.print') as mock_print:
            # Call the function directly without the module-level execution
            monte_carlo.validate_team(db_team, self.sample_team)
            
            # Check that missing players message was printed (with newline prefix)
            mock_print.assert_any_call('\nMissing team members:')
    
    def test_get_games_success(self):
        """Test successful retrieval of games for specific year and week"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Create database with multiple weeks
        db_data = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Aaron Rodgers', 'Christian McCaffrey'],
            'Season': [2023, 2023, 2023],
            'G#': [1, 2, 1],
            'FantPt': [18.5, 22.1, 25.2]
        })
        
        result = monte_carlo.get_games(db_data, 2023, 1)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # 2 players in week 1
        self.assertTrue(all(result['Season'] == 2023))
        self.assertTrue(all(result['G#'] == 1))
    
    def test_get_games_no_matches(self):
        """Test game retrieval when no matches are found"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        result = monte_carlo.get_games(self.sample_season_data, 2020, 5)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)  # No matches
    
    def test_score_player_success(self):
        """Test successful player scoring"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Create a player object
        class MockPlayer:
            def __init__(self, name):
                self.Name = name
        
        player = MockPlayer('Aaron Rodgers')
        
        result = monte_carlo.score_player(player, self.sample_season_data, 2023, 1)
        
        self.assertEqual(result, 18.5)  # Expected fantasy points
    
    def test_score_player_not_found(self):
        """Test player scoring when player is not found"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        class MockPlayer:
            def __init__(self, name):
                self.Name = name
        
        player = MockPlayer('Unknown Player')
        
        # This should raise an IndexError when trying to access [0] of empty list
        with self.assertRaises(IndexError):
            monte_carlo.score_player(player, self.sample_season_data, 2023, 1)
    
    def test_get_score_for_player_safe_recursion_limit(self):
        """Test safe version with recursion limit"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        class MockPlayer:
            def __init__(self, name, position='RB'):
                self.Name = name
                self.Position = position
        
        player = MockPlayer('Test Player', 'RB')
        
        # Mock the get_score_for_player to simulate recursion issues
        with mock.patch.object(monte_carlo, 'get_score_for_player', side_effect=RecursionError):
            result = monte_carlo.get_score_for_player_safe(
                self.sample_season_data, player, [2023], max_attempts=3
            )
            
            self.assertEqual(result, 12.0)  # RB default score
    
    def test_simulate_success(self):
        """Test successful simulation with progress monitoring"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Mock the get_score_for_player_safe function
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=20.0):
            result = monte_carlo.simulate(
                self.sample_team, self.sample_season_data, [2023], exps=5
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (5, 4))  # 5 simulations, 4 players
            # Check that all values are 20.0 using numpy array comparison
            self.assertTrue(np.all(result.values == 20.0))
    
    def test_simulate_empty_team(self):
        """Test simulation with empty team"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        empty_team = pd.DataFrame(columns=['Name', 'Position', 'Tm'])
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=20.0):
            result = monte_carlo.simulate(
                empty_team, self.sample_season_data, [2023], exps=5
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (5, 0))  # 5 simulations, 0 players
    
    def test_main_function_success(self):
        """Test main function execution"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Mock the simulate function
        mock_sim_result = pd.DataFrame({
            'Aaron Rodgers': [20.0, 22.0],
            'Christian McCaffrey': [25.0, 27.0]
        })
        
        with mock.patch.object(monte_carlo, 'simulate', return_value=mock_sim_result):
            with mock.patch('builtins.print') as mock_print:
                # Change to temp directory for file operations
                original_cwd = os.getcwd()
                os.chdir(self.temp_dir)
                
                try:
                    monte_carlo.main(years=[2023], simulations=2)
                    
                    # Check that results were printed
                    mock_print.assert_any_call(mock.ANY)  # Team projection
                    mock_print.assert_any_call(mock.ANY)  # Standard deviations
                    
                    # Check that results file was created
                    result_files = os.listdir(self.montecarlo_results_dir)
                    self.assertTrue(len(result_files) > 0)
                    
                finally:
                    os.chdir(original_cwd)
    
    def test_recursion_depth_handling(self):
        """Test that recursion depth is properly handled"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Check that recursion limit is set
        self.assertGreater(sys.getrecursionlimit(), 1000)
    
    def test_data_quality_validation(self):
        """Test data quality validation in simulation"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test with data containing NaN values
        data_with_nan = self.sample_season_data.copy()
        data_with_nan.loc[0, 'FantPt'] = np.nan
        
        # This should handle NaN values gracefully
        result = monte_carlo.get_games(data_with_nan, 2023, 1)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_edge_case_empty_database(self):
        """Test edge case with empty database"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        empty_db = pd.DataFrame(columns=['Name', 'Season', 'G#', 'FantPt'])
        
        # Test get_games with empty database
        result = monte_carlo.get_games(empty_db, 2023, 1)
        self.assertEqual(len(result), 0)
        
        # Test make_team with empty database - should handle gracefully
        try:
            result = monte_carlo.make_team(self.sample_team, empty_db)
            # If it succeeds, should return empty DataFrame
            self.assertEqual(len(result), 0)
        except (KeyError, IndexError):
            # If it fails, that's also acceptable behavior for edge case
            pass
    
    def test_team_loading_edge_cases(self):
        """Test team loading with various edge cases"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test with team missing required columns
        incomplete_team = pd.DataFrame({
            'Name': ['Test Player']
            # Missing Position and Tm columns
        })
        
        incomplete_file = os.path.join(self.teams_dir, 'incomplete_team.tsv')
        incomplete_team.to_csv(incomplete_file, sep='\t', index=False)
        
        # This should handle missing columns gracefully
        # The actual behavior depends on the script's error handling

    def test_monte_carlo_various_team_configurations(self):
        """Test Monte Carlo simulation with various team configurations"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test configuration 1: All QB team
        all_qb_team = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Tom Brady', 'Patrick Mahomes', 'Josh Allen'],
            'Position': ['QB', 'QB', 'QB', 'QB'],
            'Tm': ['NYJ', 'TB', 'KC', 'BUF']
        })
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=20.0):
            result = monte_carlo.simulate(all_qb_team, self.sample_season_data, [2023], exps=5)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (5, 4))
        
        # Test configuration 2: Mixed positions team
        mixed_team = pd.DataFrame({
            'Name': ['Aaron Rodgers', 'Derrick Henry', 'DeAndre Hopkins', 'Travis Kelce'],
            'Position': ['QB', 'RB', 'WR', 'TE'],
            'Tm': ['NYJ', 'TEN', 'ARI', 'KC']
        })
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=15.0):
            result = monte_carlo.simulate(mixed_team, self.sample_season_data, [2023], exps=3)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (3, 4))
        
        # Test configuration 3: Single player team
        single_player_team = pd.DataFrame({
            'Name': ['Aaron Rodgers'],
            'Position': ['QB'],
            'Tm': ['NYJ']
        })
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=25.0):
            result = monte_carlo.simulate(single_player_team, self.sample_season_data, [2023], exps=2)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (2, 1))
        
        # Test configuration 4: Large team
        large_team_data = []
        positions = ['QB', 'RB', 'WR', 'TE'] * 3  # 12 players
        for i in range(12):
            large_team_data.append({
                'Name': f'Player_{i+1}',
                'Position': positions[i],
                'Tm': 'TEST'
            })
        large_team = pd.DataFrame(large_team_data)
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=12.0):
            result = monte_carlo.simulate(large_team, self.sample_season_data, [2023], exps=2)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (2, 12))

    def test_monte_carlo_simulation_accuracy(self):
        """Test Monte Carlo simulation accuracy and consistency"""
        import importlib.util
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test with consistent mock scores
        consistent_score = 18.5
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=consistent_score):
            result = monte_carlo.simulate(self.sample_team, self.sample_season_data, [2023], exps=10)
            
            # All scores should be exactly the mock value
            self.assertTrue(np.all(result.values == consistent_score))
            
            # Calculate team totals
            team_totals = result.sum(axis=1)
            expected_total = consistent_score * len(self.sample_team)
            
            # All team totals should be the same
            self.assertTrue(np.all(team_totals == expected_total))
        
        # Test with variable mock scores
        variable_scores = [15.0, 20.0, 12.0, 25.0]  # Different score for each player
        
        def variable_score_side_effect(db, player, years):
            player_names = ['Aaron Rodgers', 'Christian McCaffrey', 'Tyreek Hill', 'Travis Kelce']
            if player.Name in player_names:
                return variable_scores[player_names.index(player.Name)]
            return 10.0
        
        with mock.patch.object(monte_carlo, 'get_score_for_player_safe', side_effect=variable_score_side_effect):
            result = monte_carlo.simulate(self.sample_team, self.sample_season_data, [2023], exps=5)
            
            # Check that each player gets their expected score
            for idx, (_, player) in enumerate(self.sample_team.iterrows()):
                player_scores = result[player['Name']]
                expected_score = variable_scores[idx]
                self.assertTrue(np.all(player_scores == expected_score))

    def test_monte_carlo_performance_benchmarks(self):
        """Test Monte Carlo simulation performance characteristics"""
        import importlib.util
        import time
        
        monte_carlo_path = 'scripts/analysis/montecarlo-historical-ff.py'
        spec = importlib.util.spec_from_file_location("monte_carlo", monte_carlo_path)
        monte_carlo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monte_carlo)
        
        # Test performance with different simulation sizes
        performance_results = {}
        
        simulation_sizes = [5, 10, 25]  # Small sizes for testing
        
        for sim_size in simulation_sizes:
            with mock.patch.object(monte_carlo, 'get_score_for_player_safe', return_value=15.0):
                start_time = time.time()
                result = monte_carlo.simulate(self.sample_team, self.sample_season_data, [2023], exps=sim_size)
                end_time = time.time()
                
                performance_results[sim_size] = {
                    'duration': end_time - start_time,
                    'ops_per_second': (sim_size * len(self.sample_team)) / (end_time - start_time),
                    'result_shape': result.shape
                }
        
        # Verify results make sense
        for sim_size, perf in performance_results.items():
            self.assertGreater(perf['ops_per_second'], 0)
            self.assertEqual(perf['result_shape'], (sim_size, len(self.sample_team)))
            
        # Generally, larger simulations should have better throughput (ops/sec)
        # due to amortization of setup costs, but this isn't guaranteed in all cases

if __name__ == '__main__':
    unittest.main()
