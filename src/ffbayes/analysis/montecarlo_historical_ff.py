#!python
import glob
import logging
import multiprocessing as mp
import os
from datetime import date, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from alive_progress import alive_bar

PROJECT_ROOT = Path.cwd()

# Use dynamic years based on available data

current_year = datetime.now().year

# Configuration with environment variable support for testing
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
USE_MULTIPROCESSING = os.getenv('USE_MULTIPROCESSING', 'true').lower() == 'true'
MAX_CORES = int(os.getenv('MAX_CORES', str(mp.cpu_count())))

if QUICK_TEST:
    print("🚀 QUICK TEST MODE ENABLED")
    my_years = [current_year - 2, current_year - 1]  # Only last 2 years
    number_of_simulations = 200  # Much faster for testing
    print(f"   Using {number_of_simulations} simulations with {len(my_years)} years")
else:
    my_years = list(range(current_year - 5, current_year))  # Last 5 years
    number_of_simulations = 5000

if USE_MULTIPROCESSING:
    print(f"🔥 MULTIPROCESSING ENABLED: Using {MAX_CORES} cores")
else:
    print("🔄 Single-threaded execution")

todays_date = date.today()
logging.basicConfig()

### Use Monte Carlo simulation to project the score of my team, based on historical data ###
### From Scott Rome: https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

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


combined_data = None

# Try to load team file from my_ff_teams directory

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


# Comment out module-level execution for testing
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


# Comment out module-level execution for testing
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


def simulate_batch(team, db, years, batch_size, batch_id=0):
    """
    Run a batch of simulations for multiprocessing.
    Returns a DataFrame with simulation results.
    """
    results = []
    
    for sim in range(batch_size):
        team_score = 0
        player_scores = {}
        
        for player in team.itertuples():
            score = get_score_for_player_safe(db, player, years)
            team_score += score
            player_scores[player.Name] = score
        
        # Add total team score
        player_scores['Total'] = team_score
        results.append(player_scores)
    
    return pd.DataFrame(results)


def simulate_parallel(team, db, years, exps=10):
    """
    Enhanced simulation function with multiprocessing support.
    """
    print("🎯 Starting Parallel Monte Carlo Simulation")
    print(f"   Team size: {len(team)} players")
    print(f"   Simulations: {exps:,}")
    print(f"   Years: {years}")
    print(f"   CPU cores: {MAX_CORES}")
    print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    if USE_MULTIPROCESSING and exps >= 100:  # Only use multiprocessing for larger jobs
        # Split simulations across cores
        batch_size = exps // MAX_CORES
        remainder = exps % MAX_CORES
        
        # Create batches
        batches = [batch_size] * MAX_CORES
        for i in range(remainder):
            batches[i] += 1
        
        print(f"   Batch sizes: {batches}")
        
        # Create partial function with fixed arguments
        simulate_func = partial(simulate_batch, team, db, years)
        
        # Run in parallel
        with mp.Pool(MAX_CORES) as pool:
            batch_args = [(batch_size, i) for i, batch_size in enumerate(batches) if batch_size > 0]
            
            with alive_bar(len(batch_args), title="Parallel Simulation Batches", bar="smooth") as bar:
                results = []
                for batch_size, batch_id in batch_args:
                    result = pool.apply_async(simulate_func, (batch_size, batch_id))
                    results.append(result)
                
                # Collect results
                batch_results = []
                for result in results:
                    batch_df = result.get()
                    batch_results.append(batch_df)
                    bar()
        
        # Combine all batch results
        outcome = pd.concat(batch_results, ignore_index=True)
        
    else:
        # Fall back to single-threaded simulation
        print("   Using single-threaded execution")
        outcome = simulate(team, db, years, exps)
    
    return outcome


def simulate(team, db, years, exps=10):
    """
    Original simulation function (single-threaded).
    """
    print("🎯 Starting Monte Carlo Simulation")
    print(f"   Team size: {len(team)} players")
    print(f"   Simulations: {exps:,}")
    print(f"   Total operations: {exps * len(team):,}")
    print(f"   Years: {years}")
    print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Initialize tracking variables
    total_operations = exps * len(team)
    successful_scores = 0
    fallback_scores = 0
    errors = 0
    simulation_scores = []
    
    results = []
    start_time = datetime.now()
    
    with alive_bar(total_operations, bar='smooth', title='Monte Carlo Simulation',
                   dual_line=True) as bar:
        
        for exp in range(exps):
            team_score = 0
            player_scores = {}
            
            for player in team.itertuples():
                try:
                    score = get_score_for_player_safe(db, player, years)
                    team_score += score
                    player_scores[player.Name] = score
                    
                    # Track success/fallback
                    if score > 0:
                        successful_scores += 1
                    else:
                        fallback_scores += 1
                        
                except Exception as e:
                    logging.warning(f"Error scoring {player.Name}: {e}")
                    player_scores[player.Name] = 0
                    errors += 1
                
                bar()
            
            # Add total team score
            player_scores['Total'] = team_score
            results.append(player_scores)
            simulation_scores.append(team_score)
            
            # Progress reporting every 10%
            if (exp + 1) % max(1, exps // 10) == 0:
                progress_pct = ((exp + 1) / exps) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                recent_avg = np.mean(simulation_scores[-min(100, len(simulation_scores)):])
                
                print(f"\n         📊 Progress: {progress_pct:.0f}% complete ({exp + 1}/{exps} sims)")
                print(f"    Elapsed time: {elapsed:.1f}s")
                print(f"    Recent avg team score: {recent_avg:.1f}")
    
    outcome = pd.DataFrame(results)
    
    # Final statistics
    total_time = (datetime.now() - start_time).total_seconds()
    avg_score = np.mean(simulation_scores) if simulation_scores else 0
    
    print("\n✅ Simulation Complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per simulation: {total_time/exps:.2f}s")
    print(f"   Successful scores: {successful_scores:,}")
    print(f"   Fallback scores: {fallback_scores:,}")
    print(f"   Average team score: {avg_score:.1f}")
    
    return outcome


def main(args=None):
    """
    Main function with standardized interface.
    """
    from ffbayes.utils.script_interface import create_standardized_interface
    
    interface = create_standardized_interface(
        "ffbayes-mc",
        "Fantasy Football Monte Carlo Simulation with standardized interface"
    )
    
    # Parse arguments
    if args is None:
        args = interface.parse_arguments()
    
    # Add model-specific arguments
    parser = interface.setup_argument_parser()
    parser = interface.add_model_arguments(parser)
    parser = interface.add_data_arguments(parser)
    args = parser.parse_args()
    
    # Set up logging
    logger = interface.setup_logging(args)
    
    # Parse years if provided
    years = my_years
    if args.years:
        years = interface.parse_years(args.years)
        logger.info(f"Processing specified years: {years}")
    else:
        logger.info(f"Processing default years: {years}")
    
    # Get simulations count
    simulations = number_of_simulations
    if args.quick_test:
        logger.info("Running in QUICK_TEST mode - using reduced simulations")
        simulations = min(simulations, 1000)  # Limit simulations in quick test mode
    
    # Override with command line arguments if provided
    if args.cores:
        global MAX_CORES
        MAX_CORES = args.cores
        logger.info(f"Using {MAX_CORES} CPU cores")
    
    logger.info(f"Simulation parameters: Years={years}, Simulations={simulations:,}")
    
    # Load team locally for this execution
    logger.info("Loading team data...")
    my_team = interface.handle_errors(load_team)
    logger.info(f"Loaded {len(my_team)} players")
    
    # Create team locally for this execution
    logger.info("Processing team against historical database...")
    # Load combined data here to avoid module-level I/O during import
    global combined_data
    data_dir = interface.get_data_directory(args)
    combined_data = interface.handle_errors(get_combined_data, directory_path=str(data_dir))
    team = interface.handle_errors(make_team, team=my_team, db=combined_data)
    logger.info(f"Found {len(team)} players with historical data")
    
    if len(team) == 0:
        interface.log_error("No players found in historical database. Cannot run simulation.", interface.EXIT_DATA_ERROR)
        return
    
    # Display team roster
    logger.info("Final team roster:")
    for idx, player in team.iterrows():
        logger.info(f"  {player['Position']}: {player['Name']} ({player['Tm']})")
    
    # Run simulation with parallel processing
    start_time = datetime.now()
    outcome = interface.handle_errors(simulate_parallel, team, combined_data, years, simulations)
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Validate Monte Carlo results
    logger.info("Validating Monte Carlo simulation results...")
    validation_results = interface.validate_monte_carlo_model(outcome)
    if not validation_results.get('valid', True):
        logger.warning("Monte Carlo validation issues detected")
    else:
        logger.info("Monte Carlo results validated successfully")
    
    # Generate comprehensive results summary
    logger.info("Simulation Results:")
    
    # Calculate statistics for the Total column (team scores)
    if 'Total' in outcome.columns:
        team_scores = outcome['Total']
        mean_score = team_scores.mean()
        std_score = team_scores.std()
        se_score = std_score / np.sqrt(len(team_scores))
        min_score = team_scores.min()
        max_score = team_scores.max()
        
        # 95% confidence interval
        ci_lower = mean_score - 1.96 * se_score
        ci_upper = mean_score + 1.96 * se_score
        
        logger.info(f"  Team projection: {mean_score:.2f} points")
        logger.info(f"  Standard deviation: {std_score:.2f} points")
        logger.info(f"  Standard error: {se_score:.2f}")
        logger.info(f"  Min score: {min_score:.2f} points")
        logger.info(f"  Max score: {max_score:.2f} points")
        logger.info(f"  95% confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    logger.info("Player Performance Summary:")
    for col in outcome.columns:
        if col != 'Total':  # Skip the total column
            player_scores = outcome[col]
            logger.info(f"  {col}: {player_scores.mean():.1f} ± {player_scores.std():.1f} points")
    
    # Save results
    logger.info("Saving results...")
    output_dir = interface.get_output_directory(args)
    output_file = output_dir / f'{todays_date.year}_projections_from_years{years}.tsv'
    
    outcome.to_csv(output_file, sep='\t')
    logger.info(f"Results saved to: {output_file}")
    
    logger.info(f"Total execution time: {total_time:.1f} seconds")
    interface.log_completion("Monte Carlo simulation completed successfully")
    
    return outcome


if __name__ == '__main__':
    main()
