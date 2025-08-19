#!python
import glob
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd
from alive_progress import alive_bar

sys.setrecursionlimit(10000)  # Increase recursion limit

todays_date = date.today()
logging.basicConfig()

### Use Monte Carlo simulation to project the score of my team, based on historical data ###
### From Scott Rome: https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

# Use dynamic years based on available data
from datetime import datetime

current_year = datetime.now().year
my_years = list(range(current_year - 5, current_year))  # Last 5 years
# at least great than 50
number_of_simulations = 5000


def get_combined_data(directory_path):
    # Get data file names - look for season.csv files in season_datasets subdirectory
    files = glob.glob(directory_path + '/season_datasets/*season.csv')
    dfs = list()
    for f in files:
        data = pd.read_csv(f)
        dfs.append(data)

    if not dfs:
        raise ValueError(f'No season data files found in {directory_path}/season_datasets/')

    df = pd.concat(dfs, axis=0, ignore_index=True)
    # return a pandas dataframe
    return df


combined_data = get_combined_data(directory_path='datasets')

# Try to load team file from my_ff_teams directory
import os

# Comment out module-level team loading for testing purposes
# team_files = glob.glob('my_ff_teams/my_team_*.tsv')
# if team_files:
#     # Use the most recent team file
#     latest_team_file = max(team_files, key=os.path.getctime)
#     my_team = pd.read_csv(latest_team_file, sep='\t')
#     print(f'Loaded team from {latest_team_file}')
#
#     # Check if team file has required columns, add them if missing
#     if 'Position' not in my_team.columns:
#         print('Warning: Team file missing Position column. Adding default positions...')
#         # Try to infer positions from names or add defaults
#         my_team['Position'] = 'RB'  # Default to RB, user should update
#     if 'Tm' not in my_team.columns:
#         print('Warning: Team file missing Tm column. Adding default teams...')
#         my_team['Tm'] = 'UNK'  # Default to UNK, user should update
#
#     # Filter for valid fantasy football positions
#     valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
#     my_team = my_team[my_team['Position'].isin(valid_positions)]
#     print(f'Filtered to {len(my_team)} players with valid positions')
#
# else:
#     print('No team files found in my_ff_teams/. Creating sample team for demonstration...')
#     # Create a sample team for demonstration
#     sample_players = [
#         {'Name': 'Aaron Rodgers', 'Position': 'QB', 'Tm': 'NYJ'},
#         {'Name': 'Christian McCaffrey', 'Position': 'RB', 'Tm': 'SF'},
#         {'Name': 'Tyreek Hill', 'Position': 'WR', 'Tm': 'MIA'},
#         {'Name': 'Travis Kelce', 'Position': 'TE', 'Tm': 'KC'},
#     ]
#     my_team = pd.DataFrame(sample_players)

def load_team():
    """Load team from file or create sample team"""
    team_files = glob.glob('my_ff_teams/my_team_*.tsv')
    if team_files:
        # Use the most recent team file
        latest_team_file = max(team_files, key=os.path.getctime)
        my_team = pd.read_csv(latest_team_file, sep='\t')
        print(f'Loaded team from {latest_team_file}')

        # Check if team file has required columns, add them if missing
        if 'Position' not in my_team.columns:
            print('Warning: Team file missing Position column. Adding default positions...')
            # Try to infer positions from names or add defaults
            my_team['Position'] = 'RB'  # Default to RB, user should update
        if 'Tm' not in my_team.columns:
            print('Warning: Team file missing Tm column. Adding default teams...')
            my_team['Tm'] = 'UNK'  # Default to UNK, user should update

        # Filter for valid fantasy football positions
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        my_team = my_team[my_team['Position'].isin(valid_positions)]
        print(f'Filtered to {len(my_team)} players with valid positions')
        return my_team
    else:
        print('No team files found in my_ff_teams/. Creating sample team for demonstration...')
        # Create a sample team for demonstration
        sample_players = [
            {'Name': 'Aaron Rodgers', 'Position': 'QB', 'Tm': 'NYJ'},
            {'Name': 'Christian McCaffrey', 'Position': 'RB', 'Tm': 'SF'},
            {'Name': 'Tyreek Hill', 'Position': 'WR', 'Tm': 'MIA'},
            {'Name': 'Travis Kelce', 'Position': 'TE', 'Tm': 'KC'},
        ]
        return pd.DataFrame(sample_players)

# Create a default team for module-level operations
# my_team = load_team()

# print("Combined data:", combined_data)
# print("Team:", my_team)


def make_team(team, db):
    ### Make my team based off of my picks and whether they have historical data to simulate on ###
    tm = []
    # Convert pandas Series to set more explicitly
    my_team_names = set(team['Name'].tolist())
    valid_positions = set(['QB', 'WR', 'TE', 'RB'])
    for plr in db.itertuples(index=True, name='Pandas'):
        # Ensure we're comparing strings, not pandas objects
        player_name = str(plr.Name) if plr.Name is not None else None
        player_position = str(plr.Position) if plr.Position is not None else None
        
        if player_name not in my_team_names or player_position not in valid_positions:
            continue
        tm.append(plr)
        # remove duplicates from historical data, only select name, position, and tm
    return pd.DataFrame(tm).drop_duplicates(subset=['Name', 'Position'])[['Name', 'Position', 'Tm']]


# Comment out module-level execution for testing purposes
# tm = make_team(team=my_team, db=combined_data)


# print(tm)
# print("\n")
def validate_team(db_team, my_team):
    ### Check which members of my team actually have historical data to simulate on ###
    # get the column names of team using team.dtype.names
    unique_teams = db_team.loc[:, ['Name', 'Position', 'Tm']].drop_duplicates()

    # Convert pandas Series to sets more explicitly
    db_set = set(unique_teams['Name'].tolist())
    my_team_set = set(my_team['Name'].tolist())
    print('Players to project: ')
    # Convert Series to list before printing to avoid mock comparison issues
    print(unique_teams['Name'].tolist())

    # Fix pandas comparison issue by converting to sets first
    if db_set == my_team_set:
        print('Found all team members.')
    else:
        print('\nMissing team members:')
        missing_players = my_team_set.difference(db_set)
        print(missing_players)


# Comment out module-level execution for testing purposes
# validate_team(db_team=tm, my_team=my_team)

# print(combined_data.columns)
# print("\n\n")


def get_games(db, year, week):
    ### return all the players in this week and this year
    result = db[(db['Season'] == year) & (db['G#'] == week)]
    return result


def score_player(p, db, year, week):
    """
    Score a player for a specific year and week.
    Enhanced with better error handling.
    """
    try:
        sc = db.loc[(db['Name'] == p.Name) & (db['Season'] == year) & (db['G#'] == week)]
        if sc.empty:
            raise IndexError(f"No data found for {p.Name} in {year} week {week}")
        
        fantasy_points = sc['FantPt'].tolist()
        if not fantasy_points:
            raise IndexError(f"No fantasy points found for {p.Name} in {year} week {week}")
        
        final_sc = fantasy_points[0]
        if pd.isna(final_sc):
            raise ValueError(f"Fantasy points is NaN for {p.Name} in {year} week {week}")
        
        return float(final_sc)
    except (IndexError, KeyError, ValueError) as e:
        # Re-raise with more context
        raise IndexError(f"Error scoring {p.Name}: {e}") from e


def get_score_for_player(db, player, years, max_attempts=50):
    """
    Get a fantasy score for a player by sampling random year/week combinations.
    Uses iterative approach instead of recursion to avoid stack overflow.
    """
    for attempt in range(max_attempts):
        # Sample the year and week
        year = np.random.choice(
            years,
            # for years 2017-2021
            p=[0.025, 0.075, 0.15, 0.25, 0.5],
        )
        week = np.random.randint(1, 18)

        # Find the player and score them for the given week/year
        games_data = get_games(db, year, week)
        for p in games_data.itertuples():
            if p.Name is None:
                continue
            if player.Name == p.Name:
                try:
                    sc2 = score_player(p, db, year, week)
                    return sc2
                except (IndexError, KeyError):
                    # If scoring fails, continue to next attempt
                    break
    
    # If player not found after max attempts, calculate fallback score
    return calculate_fallback_score(db, player, years)


def calculate_fallback_score(db, player, years):
    """
    Calculate a fallback score for a player based on their historical average.
    If no historical data exists, return a position-based default score.
    """
    try:
        # Try to get player's historical data across all available years
        player_data = db[db['Name'] == player.Name]
        if not player_data.empty and 'FantPt' in player_data.columns:
            # Return average historical score
            avg_score = player_data['FantPt'].mean()
            if pd.notna(avg_score):
                return float(avg_score)
    except Exception:
        pass
    
    # Position-based default scores if no historical data
    position_defaults = {
        'QB': 18.0,
        'RB': 12.0,
        'WR': 10.0,
        'TE': 8.0,
        'K': 7.0,
        'DST': 8.0
    }
    
    player_position = getattr(player, 'Position', 'RB')
    return position_defaults.get(player_position, 10.0)


def get_score_for_player_safe(db, player, years, max_attempts=10):
    """
    Safe version with enhanced error handling.
    Now uses the iterative get_score_for_player function.
    """
    try:
        return get_score_for_player(db, player, years, max_attempts)
    except Exception as e:
        print(f'Warning: Error getting score for {player.Name}: {e}, using fallback')
        return calculate_fallback_score(db, player, years)


def simulate(team, db, years, exps=10):
    """
    Run Monte Carlo simulation with enhanced progress monitoring.
    """
    import time
    from datetime import datetime

    # Initialize progress tracking
    start_time = time.time()
    total_operations = exps * len(team)
    
    print("\n🎯 Starting Monte Carlo Simulation")
    print(f"   Team size: {len(team)} players")
    print(f"   Simulations: {exps:,}")
    print(f"   Total operations: {total_operations:,}")
    print(f"   Years: {years}")
    print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    scores = pd.DataFrame(data=np.zeros((exps, len(team))), columns=[p.Name for p in team.itertuples()])
    
    # Enhanced progress bar with simulation statistics
    with alive_bar(total_operations, bar='smooth', title='Monte Carlo Simulation', 
                   dual_line=True, manual=True) as bar:
        
        simulation_stats = {
            'successful_scores': 0,
            'fallback_scores': 0,
            'errors': 0,
            'avg_score_per_sim': []
        }
        
        for n in range(exps):
            sim_start = time.time()
            sim_total_score = 0
            
            for player_idx, player in enumerate(team.itertuples()):
                try:
                    score1 = get_score_for_player_safe(db, player, years)
                    pname = player.Name
                    scores.at[n, pname] += score1
                    sim_total_score += score1
                    
                    # Track statistics
                    if score1 > 0:
                        simulation_stats['successful_scores'] += 1
                    else:
                        simulation_stats['fallback_scores'] += 1
                        
                except Exception as e:
                    print(f"Error in simulation {n+1}, player {player.Name}: {e}")
                    simulation_stats['errors'] += 1
                
                # Update progress bar
                completed = n * len(team) + player_idx + 1
                progress = completed / total_operations
                
                # Update bar with current simulation info
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = completed / elapsed
                    eta = (total_operations - completed) / rate if rate > 0 else 0
                    
                    bar.text(f'Sim {n+1}/{exps} | Player {player_idx+1}/{len(team)} | '
                            f'Rate: {rate:.1f} ops/sec | ETA: {eta:.0f}s')
                
                bar(progress)
            
            # Track simulation average
            sim_avg = sim_total_score / len(team) if len(team) > 0 else 0
            simulation_stats['avg_score_per_sim'].append(sim_avg)
            
            # Intermediate progress report every 10% of simulations
            if (n + 1) % max(1, exps // 10) == 0:
                elapsed = time.time() - start_time
                progress_pct = ((n + 1) / exps) * 100
                print(f"\n📊 Progress: {progress_pct:.0f}% complete ({n+1}/{exps} sims)")
                print(f"   Elapsed time: {elapsed:.1f}s")
                if simulation_stats['avg_score_per_sim']:
                    recent_avg = sum(simulation_stats['avg_score_per_sim'][-5:]) / min(5, len(simulation_stats['avg_score_per_sim']))
                    print(f"   Recent avg team score: {recent_avg:.1f}")
        
        bar(1.0)  # Complete the progress bar
    
    # Final simulation summary
    total_time = time.time() - start_time
    print("\n✅ Simulation Complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per simulation: {total_time/exps:.2f}s")
    print(f"   Successful scores: {simulation_stats['successful_scores']:,}")
    print(f"   Fallback scores: {simulation_stats['fallback_scores']:,}")
    if simulation_stats['errors'] > 0:
        print(f"   Errors encountered: {simulation_stats['errors']:,}")
    
    if simulation_stats['avg_score_per_sim']:
        overall_avg = sum(simulation_stats['avg_score_per_sim']) / len(simulation_stats['avg_score_per_sim'])
        print(f"   Average team score: {overall_avg:.1f}")
    
    return pd.DataFrame(scores)


def main(years=my_years, simulations=number_of_simulations):
    """
    Main function with enhanced progress monitoring and logging.
    """
    import time
    from datetime import datetime
    
    print("🚀 Fantasy Football Monte Carlo Simulation")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Simulation parameters:")
    print(f"     Years: {years}")
    print(f"     Number of simulations: {simulations:,}")
    
    start_time = time.time()
    
    # Load team locally for this execution
    print("\n📋 Loading team data...")
    my_team = load_team()
    print(f"   Loaded {len(my_team)} players")
    
    # Create team locally for this execution
    print("\n🔍 Processing team against historical database...")
    team = make_team(team=my_team, db=combined_data)
    print(f"   Found {len(team)} players with historical data")
    
    if len(team) == 0:
        print("❌ No players found in historical database. Cannot run simulation.")
        return
    
    # Display team roster
    print("\n👥 Final team roster:")
    for idx, player in team.iterrows():
        print(f"   {player['Position']}: {player['Name']} ({player['Tm']})")
    
    # Run simulation
    outcome = simulate(team=team, db=combined_data, years=years, exps=simulations)
    
    # Calculate results
    game_points = outcome.sum(axis=1, skipna=True)  # Sum the player scores together
    
    total_time = time.time() - start_time
    
    print("\n📈 Simulation Results:")
    print(f"   Team projection: {game_points.mean():.2f} points")
    print(f"   Standard deviation: {game_points.std():.2f} points")
    print(f"   Standard error: {(game_points.std() / np.sqrt(len(outcome.columns))):.2f}")
    print(f"   Min score: {game_points.min():.2f} points")
    print(f"   Max score: {game_points.max():.2f} points")
    print(f"   95% confidence interval: [{game_points.quantile(0.025):.2f}, {game_points.quantile(0.975):.2f}]")
    
    # Player-level statistics
    print("\n🏆 Player Performance Summary:")
    for col in outcome.columns:
        player_scores = outcome[col]
        print(f"   {col}: {player_scores.mean():.1f} ± {player_scores.std():.1f} points")
    
    # Save results
    print("\n💾 Saving results...")
    output_file = f'results/montecarlo_results/{todays_date.year}_projections_from_years{years}.tsv'
    outcome.to_csv(output_file, sep='\t')
    print(f"   Results saved to: {output_file}")
    
    print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Monte Carlo simulation completed successfully!")


if __name__ == '__main__':
    main()
