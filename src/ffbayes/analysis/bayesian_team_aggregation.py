#!/usr/bin/env python3
"""
Bayesian Team Aggregation for Fantasy Football

This script aggregates individual player predictions from both Monte Carlo simulations
and Bayesian models into team-level projections with proper uncertainty quantification.

Key Features:
- Loads individual player projections from Monte Carlo and Bayesian models
- Aggregates individual predictions to team totals using statistical methods
- Propagates uncertainty from individual to team level
- Handles missing players and roster variations gracefully
- Provides team projections with confidence intervals
- Integrates with existing pipeline outputs
"""

import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Configuration
# Production mode by default - test mode must be explicitly enabled
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
DEFAULT_MAX_PLAYERS_PER_TEAM = 20  # Maximum players to consider per team

print(f"🚀 QUICK TEST MODE: {'ENABLED' if QUICK_TEST else 'DISABLED'}")
if QUICK_TEST:
    print("   Using reduced parameters for faster execution")


def load_monte_carlo_results(results_dir: str = None) -> pd.DataFrame:
    """Load Monte Carlo simulation results."""
    if results_dir is None:
        # Always use year-based directory structure for consistency
        current_year = datetime.now().year
        from ffbayes.utils.path_constants import get_monte_carlo_dir
        results_dir = str(get_monte_carlo_dir(current_year))
    """
    Load Monte Carlo simulation results containing individual player projections.
    
    Args:
        results_dir: Directory containing Monte Carlo results
        
    Returns:
        DataFrame with individual player projections and team totals
    """
    print("📊 Loading Monte Carlo simulation results...")
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Monte Carlo results directory not found: {results_dir}")
    
    # Find the most recent Monte Carlo results file
    mc_files = glob.glob(os.path.join(results_dir, '*projections*.tsv'))
    if not mc_files:
        raise FileNotFoundError(f"No Monte Carlo projection files found in {results_dir}")
    
    # CRITICAL FIX: Use modification time instead of creation time
    # Creation time can be misleading when files are moved/copied
    latest_file = max(mc_files, key=os.path.getmtime)
    print(f"   Loading: {os.path.basename(latest_file)}")
    
    # Load the data
    data = pd.read_csv(latest_file, sep='\t')
    print(f"   Loaded {len(data)} simulations with {len(data.columns)-1} players")
    
    return data


def load_bayesian_individual_predictions(results_dir: str = None) -> Dict:
    """Load Bayesian individual player predictions with uncertainty estimates."""
    if results_dir is None:
        from ffbayes.utils.path_constants import get_hybrid_mc_dir
        current_year = datetime.now().year
        results_dir = str(get_hybrid_mc_dir(current_year))
    """
    Load Bayesian individual player predictions with uncertainty estimates.
    
    Args:
        results_dir: Directory containing Bayesian model results
        
    Returns:
        Dictionary containing player predictions and uncertainty
    """
    print("🧠 Loading Bayesian individual predictions...")
    
    if not os.path.exists(results_dir):
        print("   ⚠️  Bayesian results directory not found, skipping Bayesian predictions")
        return {}
    
    # Look for the most recent results file
    results_files = glob.glob(os.path.join(results_dir, 'hybrid_model_results.json'))
    if not results_files:
        print("   ⚠️  No Bayesian results files found, skipping Bayesian predictions")
        return {}
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"   Loading: {os.path.basename(latest_file)}")
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        # Extract relevant information - handle both old and new Hybrid MC format
        if 'player_predictions' in results:
            # Old format (unified_model_results.json)
            player_predictions = results.get('player_predictions', {})
        else:
            # New Hybrid MC format - player data is at top level
            player_predictions = {}
            for player_name, player_data in results.items():
                if isinstance(player_data, dict) and 'monte_carlo' in player_data:
                    # Extract Monte Carlo predictions
                    mc_data = player_data['monte_carlo']
                    player_predictions[player_name] = {
                        'mean': mc_data.get('mean', 0),
                        'std': mc_data.get('std', 0),
                        'position': player_data.get('position', 'UNK'),
                        'team': player_data.get('team', 'UNK')
                    }
        
        bayesian_data = {
            'mae_bayesian': results.get('mae_bayesian', None),
            'mae_baseline': results.get('mae_baseline', None),
            'team_names': results.get('team_names', []),
            'test_data_shape': results.get('test_data_shape', []),
            'predictions_shape': results.get('predictions_shape', []),
            'timestamp': results.get('timestamp', ''),
            # CRITICAL: Include player predictions for team aggregation
            'player_predictions': player_predictions
        }
        
        # Count available players instead of teams
        num_players = len(bayesian_data['player_predictions'])
        print(f"   Loaded Bayesian results with {num_players} player predictions")
        return bayesian_data
        
    except Exception as e:
        print(f"   ⚠️  Error loading Bayesian results: {e}")
        return {}


def get_player_positions(available_players: List[str], team_file: str = None) -> Dict[str, str]:
    """
    Get position information for players from the team file or combined dataset.
    
    Args:
        available_players: List of player names
        team_file: Optional path to team file for position lookup
        
    Returns:
        Dictionary mapping player names to positions
    """
    print("📊 Loading player position information...")
    
    # First try to get positions from the team file (more reliable)
    if team_file and os.path.exists(team_file):
        try:
            team_df = pd.read_csv(team_file, sep='\t')
            if 'Name' in team_df.columns and 'Position' in team_df.columns:
                print(f"   Loading positions from team file: {os.path.basename(team_file)}")
                player_positions = {}
                for player in available_players:
                    player_data = team_df[team_df['Name'] == player]
                    if not player_data.empty:
                        player_positions[player] = player_data.iloc[0]['Position']
                    else:
                        player_positions[player] = 'UNK'
                
                found_positions = len([p for p in player_positions.values() if p != 'UNK'])
                print(f"   Found positions for {found_positions}/{len(available_players)} players from team file")
                if found_positions == len(available_players):
                    return player_positions
                else:
                    print("   ⚠️  Some positions missing from team file, falling back to dataset")
        except Exception as e:
            print(f"   ⚠️  Error reading team file: {e}")
    
    # Fallback to combined dataset files using path constants
    from ffbayes.utils.path_constants import COMBINED_DATASETS_DIR
    dataset_files = glob.glob(str(COMBINED_DATASETS_DIR / '*_modern.csv'))
    if not dataset_files:
        print("   ⚠️  No combined dataset files found")
        return {player: 'UNK' for player in available_players}
    
    latest_file = max(dataset_files, key=os.path.getmtime)
    print(f"   Loading positions from: {os.path.basename(latest_file)}")
    
    try:
        # Load the dataset and get unique player-position mappings
        df = pd.read_csv(latest_file)
        if 'Name' not in df.columns or 'Position' not in df.columns:
            print("   ⚠️  Dataset missing Name or Position columns")
            return {player: 'UNK' for player in available_players}
        
        # Get the most recent position for each player
        player_positions = {}
        for player in available_players:
            player_data = df[df['Name'] == player]
            if not player_data.empty:
                # Get the most recent position (highest season)
                latest_record = player_data.loc[player_data['Season'].idxmax()]
                player_positions[player] = latest_record['Position']
            else:
                player_positions[player] = 'UNK'
        
        print(f"   Found positions for {len([p for p in player_positions.values() if p != 'UNK'])}/{len(available_players)} players")
        return player_positions
        
    except Exception as e:
        print(f"   ⚠️  Error loading position data: {e}")
        return {player: 'UNK' for player in available_players}


def aggregate_individual_to_team_projections(
    mc_data: pd.DataFrame,
    team_roster: List[str],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    team_file: str = None
) -> Dict:
    """
    Aggregate individual player projections to team-level projections.
    
    Args:
        mc_data: Monte Carlo simulation results DataFrame
        team_roster: List of player names in the team
        confidence_level: Confidence level for uncertainty intervals
        
    Returns:
        Dictionary containing team projections and uncertainty
    """
    print(f"🔗 Aggregating {len(team_roster)} players to team projections...")
    
    # CRITICAL VALIDATION: Check if Monte Carlo results have proper variance
    if 'Total' in mc_data.columns:
        total_scores = mc_data['Total'].values
        total_std = np.std(total_scores)
        total_mean = np.mean(total_scores)
        
        if total_std <= 0.001:
            print("   🚨 CRITICAL ERROR: Monte Carlo simulation produced identical results!")
            print(f"   All {len(total_scores)} simulations returned: {total_mean:.1f}")
            print(f"   Standard deviation: {total_std:.6f}")
            print("   This indicates the Monte Carlo simulation is broken")
            print("   Expected: Multiple different simulation results with variance")
            print("   Actual: Deterministic results with zero uncertainty")
            raise ValueError("Monte Carlo simulation failed - all results identical")
        else:
            print("   ✅ Monte Carlo validation passed")
            print(f"   Simulations: {len(total_scores)}")
            print(f"   Team score range: {total_scores.min():.1f} - {total_scores.max():.1f}")
            print(f"   Mean: {total_mean:.1f}, Std: {total_std:.2f}")
    
    # Filter data to only include players in the roster
    available_players = [p for p in team_roster if p in mc_data.columns]
    missing_players = [p for p in team_roster if p not in mc_data.columns]
    
    if missing_players:
        print(f"   ⚠️  Missing data for {len(missing_players)} players: {missing_players}")
    
    if not available_players:
        raise ValueError("No player data available for team aggregation")
    
    print(f"   Using {len(available_players)} players with available data")
    
    # Calculate team totals for each simulation
    team_scores = []
    for _, row in mc_data.iterrows():
        team_score = sum(row[player] for player in available_players)
        team_scores.append(team_score)
    
    team_scores = np.array(team_scores)
    
    # Calculate team-level statistics
    team_mean = np.mean(team_scores)
    team_std = np.std(team_scores)
    team_median = np.median(team_scores)
    
    # Calculate confidence intervals
    confidence_interval = stats.t.interval(
        confidence_level, 
        len(team_scores) - 1, 
        loc=team_mean, 
        scale=team_std / np.sqrt(len(team_scores))
    )
    
    # Calculate percentiles
    percentiles = np.percentile(team_scores, [5, 25, 50, 75, 95])
    
    # Get player position information
    player_positions = get_player_positions(available_players, team_file)
    
    # Individual player contributions
    player_contributions = {}
    for player in available_players:
        player_scores = mc_data[player].values
        player_contributions[player] = {
            'mean': np.mean(player_scores),
            'std': np.std(player_scores),
            'contribution_pct': np.mean(player_scores) / team_mean * 100,
            'position': player_positions.get(player, 'UNK')
        }
    
    team_projection = {
        'team_score_mean': team_mean,
        'team_score_std': team_std,
        'team_score_median': team_median,
        'confidence_interval': confidence_interval,
        'percentiles': {
            'p5': percentiles[0],
            'p25': percentiles[1],
            'p50': percentiles[2],
            'p75': percentiles[3],
            'p95': percentiles[4]
        },
        'player_contributions': player_contributions,
        'available_players': available_players,
        'missing_players': missing_players,
        'simulation_count': len(team_scores),
        'confidence_level': confidence_level
    }
    
    print(f"   Team projection: {team_mean:.1f} ± {team_std:.1f} points")
    print(f"   Confidence interval: [{confidence_interval[0]:.1f}, {confidence_interval[1]:.1f}]")
    
    return team_projection


def propagate_uncertainty_individual_to_team(
    individual_predictions: List[Dict],
    correlation_factor: float = 0.3
) -> Dict:
    """
    Propagate uncertainty from individual player predictions to team level.
    
    Args:
        individual_predictions: List of player prediction dictionaries
        correlation_factor: Assumed correlation between player performances
        
    Returns:
        Dictionary with team-level uncertainty quantification
    """
    print("📈 Propagating uncertainty from individual to team level...")
    
    if not individual_predictions:
        raise ValueError("No individual predictions provided")
    
    # Extract individual means and uncertainties
    means = [pred['predicted_score'] for pred in individual_predictions]
    uncertainties = [pred['uncertainty'] for pred in individual_predictions]
    
    # Calculate team total mean
    team_mean = sum(means)
    
    # Calculate team total uncertainty with correlation
    # Formula: sqrt(sum(variances) + 2 * correlation * sum(covariances))
    variances = [u**2 for u in uncertainties]
    
    # Calculate covariance terms (simplified assumption)
    covariance_sum = 0
    for i in range(len(uncertainties)):
        for j in range(i+1, len(uncertainties)):
            covariance_sum += uncertainties[i] * uncertainties[j]
    
    team_variance = sum(variances) + 2 * correlation_factor * covariance_sum
    team_uncertainty = np.sqrt(team_variance)
    
    # Calculate confidence intervals
    confidence_interval = stats.norm.interval(
        DEFAULT_CONFIDENCE_LEVEL, 
        loc=team_mean, 
        scale=team_uncertainty
    )
    
    uncertainty_propagation = {
        'team_mean': team_mean,
        'team_uncertainty': team_uncertainty,
        'individual_means': means,
        'individual_uncertainties': uncertainties,
        'correlation_factor': correlation_factor,
        'confidence_interval': confidence_interval,
        'uncertainty_ratio': team_uncertainty / team_mean if team_mean > 0 else 0
    }
    
    print(f"   Team uncertainty: {team_uncertainty:.2f} points")
    print(f"   Uncertainty ratio: {uncertainty_propagation['uncertainty_ratio']:.2%}")
    
    return uncertainty_propagation


def handle_missing_players_and_roster_variations(
    base_roster: List[str],
    available_players: List[str],
    substitution_rules: Optional[Dict] = None
) -> Dict:
    """
    Handle missing players and roster variations gracefully.
    
    Args:
        base_roster: Original team roster
        available_players: Players with available data
        substitution_rules: Rules for player substitutions
        
    Returns:
        Dictionary with roster adjustments and recommendations
    """
    print("🔄 Handling roster variations and missing players...")
    
    missing_players = [p for p in base_roster if p not in available_players]
    available_count = len(available_players)
    missing_count = len(missing_players)
    
    roster_handling = {
        'original_roster_size': len(base_roster),
        'available_players_count': available_count,
        'missing_players_count': missing_count,
        'coverage_percentage': available_count / len(base_roster) * 100 if base_roster else 0,
        'missing_players': missing_players,
        'available_players': available_players,
        'recommendations': []
    }
    
    # Generate recommendations based on coverage
    if roster_handling['coverage_percentage'] < 50:
        roster_handling['recommendations'].append(
            "Low roster coverage - consider adding more players or using historical averages"
        )
    elif roster_handling['coverage_percentage'] < 75:
        roster_handling['recommendations'].append(
            "Moderate roster coverage - projections may have increased uncertainty"
        )
    else:
        roster_handling['recommendations'].append(
            "Good roster coverage - projections should be reliable"
        )
    
    if missing_players:
        roster_handling['recommendations'].append(
            f"Missing data for {missing_count} players - using available data only"
        )
    
    print(f"   Roster coverage: {roster_handling['coverage_percentage']:.1f}%")
    print(f"   Missing players: {missing_count}")
    
    return roster_handling


def create_bayesian_team_projection(
    bayesian_data: Dict,
    team_roster: List[str],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
) -> Optional[Dict]:
    """
    Create team-level projections from Bayesian individual player predictions.
    
    Args:
        bayesian_data: Bayesian model results containing player predictions
        team_roster: List of player names in the team
        confidence_level: Confidence level for uncertainty intervals
        
    Returns:
        Dictionary containing Bayesian team projections and uncertainty, or None if no data
    """
    print("🧠 Creating Bayesian team projections from individual predictions...")
    
    if not bayesian_data or 'player_predictions' not in bayesian_data:
        print("   ⚠️  No Bayesian player predictions available")
        return None
    
    player_predictions = bayesian_data['player_predictions']
    
    # Filter to only include players in the roster
    available_players = [p for p in team_roster if p in player_predictions]
    missing_players = [p for p in team_roster if p not in player_predictions]
    
    if missing_players:
        print(f"   ⚠️  Missing Bayesian predictions for {len(missing_players)} players: {missing_players}")
    
    if not available_players:
        print("   ❌ No Bayesian predictions available for team roster")
        return None
    
    print(f"   Using {len(available_players)} players with Bayesian predictions")
    
    # Extract individual means and uncertainties
    player_means = []
    player_uncertainties = []
    player_positions = {}
    
    for player in available_players:
        pred = player_predictions[player]
        player_means.append(pred['mean'])
        player_uncertainties.append(pred['std'])
        player_positions[player] = pred.get('position', 'UNK')
    
    # Calculate team total mean
    team_mean = sum(player_means)
    
    # Calculate team total uncertainty with correlation assumption
    # Use the same correlation factor as in the existing function
    correlation_factor = 0.3
    variances = [u**2 for u in player_uncertainties]
    
    # Calculate covariance terms (simplified assumption)
    covariance_sum = 0
    for i in range(len(player_uncertainties)):
        for j in range(i+1, len(player_uncertainties)):
            covariance_sum += player_uncertainties[i] * player_uncertainties[j]
    
    team_variance = sum(variances) + 2 * correlation_factor * covariance_sum
    team_uncertainty = np.sqrt(team_variance)
    
    # Calculate confidence intervals using normal distribution assumption
    confidence_interval = stats.norm.interval(
        confidence_level, 
        loc=team_mean, 
        scale=team_uncertainty
    )
    
    # Calculate percentiles using normal distribution
    percentiles = {
        'p5': team_mean - 1.645 * team_uncertainty,
        'p25': team_mean - 0.674 * team_uncertainty,
        'p50': team_mean,
        'p75': team_mean + 0.674 * team_uncertainty,
        'p95': team_mean + 1.645 * team_uncertainty
    }
    
    # Individual player contributions
    player_contributions = {}
    for i, player in enumerate(available_players):
        player_contributions[player] = {
            'mean': player_means[i],
            'std': player_uncertainties[i],
            'contribution_pct': player_means[i] / team_mean * 100,
            'position': player_positions[player]
        }
    
    # Calculate coverage percentage
    coverage_percentage = len(available_players) / len(team_roster) * 100
    
    bayesian_team_projection = {
        'team_score_mean': team_mean,
        'team_score_std': team_uncertainty,
        'team_score_median': team_mean,  # Normal distribution assumption
        'confidence_interval': confidence_interval,
        'percentiles': percentiles,
        'player_contributions': player_contributions,
        'available_players': available_players,
        'missing_players': missing_players,
        'prediction_method': 'bayesian_hierarchical',
        'confidence_level': confidence_level,
        'correlation_factor': correlation_factor,
        'coverage_percentage': coverage_percentage,
        'note': f'Partial team projection based on {len(available_players)}/{len(team_roster)} players'
    }
    
    print(f"   Bayesian team projection: {team_mean:.1f} ± {team_uncertainty:.1f} points")
    print(f"   Confidence interval: [{confidence_interval[0]:.1f}, {confidence_interval[1]:.1f}]")
    print(f"   Coverage: {coverage_percentage:.1f}% of team roster")
    
    return bayesian_team_projection


def integrate_with_existing_outputs(
    monte_carlo_projection: Dict,
    bayesian_data: Optional[Dict] = None
) -> Dict:
    """
    Integrate team aggregation results with existing model outputs.
    
    Args:
        monte_carlo_projection: Team projection from Monte Carlo aggregation
        bayesian_data: Optional Bayesian model data for comparison
        
    Returns:
        Integrated results with comparisons and insights
    """
    print("🔗 Integrating with existing model outputs...")
    
    integrated_results = {
        'monte_carlo_projection': monte_carlo_projection,
        'bayesian_data': bayesian_data,
        'integration_timestamp': datetime.now().isoformat(),
        'comparisons': {},
        'insights': []
    }
    
    # Create Bayesian team projections if individual predictions are available
    bayesian_team_projection = None
    if bayesian_data and 'player_predictions' in bayesian_data:
        # Get team roster from Monte Carlo projection
        team_roster = monte_carlo_projection.get('available_players', [])
        if team_roster:
            bayesian_team_projection = create_bayesian_team_projection(
                bayesian_data, team_roster
            )
            if bayesian_team_projection:
                integrated_results['bayesian_team_projection'] = bayesian_team_projection
                print("   ✅ Created Bayesian team projections for comparison")
    
    # Add comparisons if both projections are available
    if bayesian_team_projection:
        mc_mean = monte_carlo_projection.get('team_score_mean', 0)
        mc_std = monte_carlo_projection.get('team_score_std', 0)
        bayes_mean = bayesian_team_projection.get('team_score_mean', 0)
        bayes_std = bayesian_team_projection.get('team_score_std', 0)
        bayes_coverage = bayesian_team_projection.get('coverage_percentage', 0)
        
        if mc_mean > 0 and bayes_mean > 0:
            # Mean difference analysis
            mean_diff = abs(mc_mean - bayes_mean)
            mean_diff_pct = (mean_diff / mc_mean) * 100
            
            integrated_results['comparisons']['mean_difference'] = mean_diff
            integrated_results['comparisons']['mean_difference_pct'] = mean_diff_pct
            
            # Uncertainty comparison
            if bayes_std > 0:
                uncertainty_ratio = mc_std / bayes_std
                integrated_results['comparisons']['uncertainty_ratio'] = uncertainty_ratio
                
                if uncertainty_ratio < 0.8:
                    integrated_results['insights'].append(
                        "Monte Carlo shows lower uncertainty than Bayesian model"
                    )
                elif uncertainty_ratio > 1.2:
                    integrated_results['insights'].append(
                        "Monte Carlo shows higher uncertainty than Bayesian model"
                    )
                else:
                    integrated_results['insights'].append(
                        "Similar uncertainty estimates between models"
                    )
            
            # Mean agreement analysis
            if mean_diff_pct < 5.0:
                integrated_results['insights'].append(
                    "Excellent agreement between Monte Carlo and Bayesian models"
                )
            elif mean_diff_pct < 10.0:
                integrated_results['insights'].append(
                    "Good agreement between Monte Carlo and Bayesian models"
                )
            elif mean_diff_pct < 20.0:
                integrated_results['insights'].append(
                    "Moderate agreement between Monte Carlo and Bayesian models"
                )
            else:
                integrated_results['insights'].append(
                    "Significant disagreement between Monte Carlo and Bayesian models"
                )
            
            # Add coverage note for partial projections
            if bayes_coverage < 100:
                integrated_results['insights'].append(
                    f"⚠️  Bayesian projection based on {bayes_coverage:.1f}% of team roster"
                )
                integrated_results['insights'].append(
                    "💡 Consider using players with available Bayesian predictions for more accurate comparisons"
                )
        else:
            integrated_results['insights'].append(
                "⚠️  Unable to compare projections due to invalid data"
            )
    else:
        # No Bayesian team projections available - provide insights about model capabilities
        integrated_results['insights'].append(
            "📊 Monte Carlo team projections available for current test team"
        )
        integrated_results['insights'].append(
            "🧠 Bayesian model trained on 2023 data - different player set than test team"
        )
        integrated_results['insights'].append(
            "💡 For full comparison, use players available in both datasets"
        )
        
        # Add Monte Carlo insights
        mc_mean = monte_carlo_projection.get('team_score_mean', 0)
        mc_std = monte_carlo_projection.get('team_score_std', 0)
        if mc_mean > 0:
            integrated_results['insights'].append(
                f"🎯 Test team projection: {mc_mean:.1f} ± {mc_std:.1f} points"
            )
            uncertainty_pct = (mc_std / mc_mean * 100) if mc_mean > 0 else 0
            integrated_results['insights'].append(
                f"📈 Uncertainty: {uncertainty_pct:.1f}% of mean score"
            )
    
    # Add MAE comparison if available (legacy support)
    if bayesian_data and bayesian_data.get('mae_bayesian') is not None:
        mc_mae = monte_carlo_projection.get('team_score_std', 0)
        bayesian_mae = bayesian_data.get('mae_bayesian', 0)
        
        if bayesian_mae > 0:
            mae_ratio = mc_mae / bayesian_mae
            integrated_results['comparisons']['mae_ratio'] = mae_ratio
            
            if mae_ratio < 1:
                integrated_results['insights'].append(
                    "Monte Carlo projections show lower uncertainty than Bayesian model (MAE-based)"
                )
            else:
                integrated_results['insights'].append(
                    "Monte Carlo projections show higher uncertainty than Bayesian model (MAE-based)"
                )
    
    # Add general insights
    team_score = monte_carlo_projection.get('team_score_mean', 0)
    if team_score > 120:
        integrated_results['insights'].append("High-scoring team projection - strong fantasy potential")
    elif team_score < 80:
        integrated_results['insights'].append("Low-scoring team projection - consider roster adjustments")
    
    uncertainty_ratio = monte_carlo_projection.get('team_score_std', 0) / team_score if team_score > 0 else 0
    if uncertainty_ratio > 0.2:
        integrated_results['insights'].append("High uncertainty - projections may be less reliable")
    elif uncertainty_ratio < 0.1:
        integrated_results['insights'].append("Low uncertainty - projections should be reliable")
    
    print(f"   Generated {len(integrated_results['insights'])} insights")
    
    return integrated_results


def save_team_aggregation_results(
    results: Dict,
    output_dir: str = None
) -> str:
    """Save team aggregation results to JSON file."""
    if output_dir is None:
        from ffbayes.utils.path_constants import get_team_aggregation_dir
        output_dir = str(get_team_aggregation_dir(datetime.now().year))
    """
    Save team aggregation results to files.
    
    Args:
        results: Team aggregation results dictionary
        output_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    print("💾 Saving team aggregation results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename for consistency with post-draft visualizations
    results_file = os.path.join(output_dir, 'team_analysis_results.json')
    
    # Save results as JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Results saved to: {results_file}")
    
    return results_file


# REMOVED: generate_team_aggregation_visualizations function
# This was creating useless duplicate charts that are much better handled by
# the comprehensive post-draft visualization script


def main():
    """Main function to run Bayesian team aggregation."""
    print("=" * 70)
    print("Bayesian Team Aggregation for Fantasy Football")
    print("Aggregating individual player predictions to team-level projections")
    print("=" * 70)
    
    try:
        # Load Monte Carlo results
        mc_data = load_monte_carlo_results()
        
        # Load Bayesian results (optional)
        bayesian_data = load_bayesian_individual_predictions()
        
        # CRITICAL: Load team roster - configurable path, no hardcoding
        # Priority: 1. Command line argument, 2. Environment variable, 3. Default based on QUICK_TEST
        import argparse

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Bayesian Team Aggregation')
        parser.add_argument('--team-file', type=str, help='Path to team roster file')
        args, _ = parser.parse_known_args()
        
        # Determine team file path - use test file for testing, production file for real drafts
        if args.team_file:
            team_file = args.team_file
            print(f"   Using team file from command line: {team_file}")
        elif os.getenv('TEAM_FILE'):
            team_file = os.getenv('TEAM_FILE')
            print(f"   Using team file from environment: {team_file}")
        elif QUICK_TEST:
            # Testing mode - use dedicated test team file with same 16 players every time
            from ffbayes.utils.path_constants import get_default_team_file
            team_file = str(get_default_team_file())
            print(f"   QUICK_TEST mode - using standard test team: {team_file}")
        else:
            # Production mode - look for user's actual draft picks
            current_year = datetime.now().year
            from ffbayes.utils.path_constants import get_teams_dir
            team_file = str(get_teams_dir() / f'drafted_team_{current_year}.tsv')
            print(f"   Production mode - looking for user draft picks: {team_file}")
        
        if not os.path.exists(team_file):
            raise FileNotFoundError(
                f"Team roster file not found: {team_file}. "
                "Production model requires real team data. "
                "No fallbacks or hardcoded rosters allowed."
            )
        
        team_df = pd.read_csv(team_file, sep='\t')
        # Normalize to unified column names
        if 'Name' not in team_df.columns and 'PLAYER' in team_df.columns:
            team_df = team_df.rename(columns={'PLAYER': 'Name'})
        if 'Position' not in team_df.columns and 'POS' in team_df.columns:
            team_df = team_df.rename(columns={'POS': 'Position'})
        if 'Team' not in team_df.columns:
            print("   🔎 Inferring Team column from unified dataset...")
            try:
                from ffbayes.data_pipeline.unified_data_loader import \
                    load_unified_dataset
                unified_df = load_unified_dataset('datasets')
                latest_team = (unified_df.sort_values(['Name', 'Season'])
                                        .groupby('Name')
                                        .tail(1)[['Name', 'Tm']]
                                        .rename(columns={'Tm': 'Team'}))
                team_df = team_df.merge(latest_team, on='Name', how='left')
            except Exception as e:
                print(f"   ⚠️  Could not infer Team from unified dataset: {e}")
                team_df['Team'] = None
        
        if 'Name' not in team_df.columns:
            raise ValueError(
                f"Team roster file missing 'Name' column: {team_file}. "
                "Production model requires properly formatted team data."
            )
        
        sample_roster = team_df['Name'].tolist()
        if not sample_roster:
            raise ValueError(
                f"Team roster file is empty: {team_file}. "
                "Production model requires non-empty team data."
            )
        
        print(f"   Loaded team roster from: {team_file}")
        print(f"   Team size: {len(sample_roster)} players")
        
        # Aggregate individual projections to team level
        team_projection = aggregate_individual_to_team_projections(mc_data, sample_roster, team_file=team_file)
        
        # Handle roster variations
        roster_handling = handle_missing_players_and_roster_variations(
            sample_roster, 
            team_projection['available_players']
        )
        
        # Integrate with existing outputs
        integrated_results = integrate_with_existing_outputs(team_projection, bayesian_data)

        # Enrich with team_strength and player_projections for downstream visuals
        try:
            # Per-player mean/std from MC simulations
            player_cols = [c for c in mc_data.columns if c != 'Total']
            per_player_mean = mc_data[player_cols].mean(axis=0)
            per_player_std = mc_data[player_cols].std(axis=0)
            # Positions
            player_positions = get_player_positions(player_cols, team_file=team_file)
            # Team strength: average projected points by position
            strength = {}
            for pos in set(player_positions.values()):
                if pos == 'UNK':
                    continue
                names_in_pos = [p for p, ppos in player_positions.items() if ppos == pos and p in per_player_mean.index]
                if names_in_pos:
                    strength[pos] = float(per_player_mean[names_in_pos].mean())
            integrated_results['team_strength'] = strength
            # Player projections dict
            player_proj = {}
            for p in player_cols:
                if p in per_player_mean.index:
                    player_proj[p] = {
                        'mean': float(per_player_mean[p]),
                        'std': float(per_player_std[p]),
                        'position': player_positions.get(p, 'UNK')
                    }
            integrated_results['player_projections'] = player_proj
        except Exception as e:
            print(f"   ⚠️  Could not compute team_strength/player_projections: {e}")
        
        # Save results
        results_file = save_team_aggregation_results(integrated_results)
        

        
        # Note: Visualizations are now handled by the comprehensive post-draft visualization script
        # to avoid redundancy and provide better, more actionable charts
        plot_files = []
        
        # Print summary
        print("\n" + "=" * 70)
        print("Team Aggregation Summary:")
        print(f"- Team Projection: {team_projection['team_score_mean']:.1f} ± {team_projection['team_score_std']:.1f} points")
        print(f"- Roster Coverage: {roster_handling['coverage_percentage']:.1f}%")
        print(f"- Confidence Level: {team_projection['confidence_level']:.0%}")
        print(f"- Results saved to: {results_file}")
        print(f"- Visualizations: {len(plot_files)} plots generated")
        print("=" * 70)
        
        return integrated_results
        
    except Exception as e:
        print(f"❌ Error in team aggregation: {e}")
        return None


if __name__ == '__main__':
    main()
