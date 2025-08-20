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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
DEFAULT_MAX_PLAYERS_PER_TEAM = 20  # Maximum players to consider per team

print(f"🚀 QUICK TEST MODE: {'ENABLED' if QUICK_TEST else 'DISABLED'}")
if QUICK_TEST:
    print("   Using reduced parameters for faster execution")


def load_monte_carlo_results(results_dir: str = 'results/montecarlo_results') -> pd.DataFrame:
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
    
    latest_file = max(mc_files, key=os.path.getctime)
    print(f"   Loading: {os.path.basename(latest_file)}")
    
    # Load the data
    data = pd.read_csv(latest_file, sep='\t')
    print(f"   Loaded {len(data)} simulations with {len(data.columns)-1} players")
    
    return data


def load_bayesian_individual_predictions(results_dir: str = 'results/bayesian-hierarchical-results') -> Dict:
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
    results_files = glob.glob(os.path.join(results_dir, 'modern_model_results.json'))
    if not results_files:
        print("   ⚠️  No Bayesian results files found, skipping Bayesian predictions")
        return {}
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"   Loading: {os.path.basename(latest_file)}")
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        # Extract relevant information
        bayesian_data = {
            'mae_bayesian': results.get('mae_bayesian', None),
            'mae_baseline': results.get('mae_baseline', None),
            'team_names': results.get('team_names', []),
            'test_data_shape': results.get('test_data_shape', []),
            'predictions_shape': results.get('predictions_shape', []),
            'timestamp': results.get('timestamp', '')
        }
        
        print(f"   Loaded Bayesian results with {len(bayesian_data['team_names'])} teams")
        return bayesian_data
        
    except Exception as e:
        print(f"   ⚠️  Error loading Bayesian results: {e}")
        return {}


def aggregate_individual_to_team_projections(
    mc_data: pd.DataFrame,
    team_roster: List[str],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
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
    
    # Individual player contributions
    player_contributions = {}
    for player in available_players:
        player_scores = mc_data[player].values
        player_contributions[player] = {
            'mean': np.mean(player_scores),
            'std': np.std(player_scores),
            'contribution_pct': np.mean(player_scores) / team_mean * 100
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
    
    # Add comparisons if Bayesian data is available
    if bayesian_data and bayesian_data.get('mae_bayesian') is not None:
        mc_mae = monte_carlo_projection.get('team_score_std', 0)
        bayesian_mae = bayesian_data.get('mae_bayesian', 0)
        
        if bayesian_mae > 0:
            mae_ratio = mc_mae / bayesian_mae
            integrated_results['comparisons']['mae_ratio'] = mae_ratio
            
            if mae_ratio < 1:
                integrated_results['insights'].append(
                    "Monte Carlo projections show lower uncertainty than Bayesian model"
                )
            else:
                integrated_results['insights'].append(
                    "Monte Carlo projections show higher uncertainty than Bayesian model"
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
    output_dir: str = 'results/team_aggregation'
) -> str:
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
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'team_aggregation_results_{timestamp}.json')
    
    # Save results as JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Results saved to: {results_file}")
    
    return results_file


def generate_team_aggregation_visualizations(
    results: Dict,
    output_dir: str = 'plots'
) -> List[str]:
    """
    Generate visualizations for team aggregation results.
    
    Args:
        results: Team aggregation results dictionary
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    print("📊 Generating team aggregation visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Plot 1: Team Score Distribution
        if 'monte_carlo_projection' in results:
            mc_proj = results['monte_carlo_projection']
            
            plt.figure(figsize=(12, 8))
            
            # Create a simulated distribution based on mean and std
            if mc_proj.get('team_score_mean') and mc_proj.get('team_score_std'):
                mean = mc_proj['team_score_mean']
                std = mc_proj['team_score_std']
                x = np.linspace(mean - 3*std, mean + 3*std, 100)
                y = stats.norm.pdf(x, mean, std)
                
                plt.plot(x, y, 'b-', linewidth=2, label='Team Score Distribution')
                plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.1f}')
                plt.axvline(mean + std, color='g', linestyle=':', label=f'+1σ: {mean + std:.1f}')
                plt.axvline(mean - std, color='g', linestyle=':', label=f'-1σ: {mean - std:.1f}')
                
                plt.xlabel('Team Score (Fantasy Points)')
                plt.ylabel('Probability Density')
                plt.title('Team Score Distribution with Uncertainty')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_file = os.path.join(output_dir, f'team_score_distribution_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_file)
        
        # Note: Player contribution breakdown is now handled by the comprehensive
        # visualization script (team_score_breakdown chart) to avoid redundancy
        
        print(f"   Generated {len(plot_files)} visualization plots")
        
    except Exception as e:
        print(f"   ⚠️  Error generating visualizations: {e}")
    
    return plot_files


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
        
        # Define a sample team roster (in practice, this would come from user input)
        sample_roster = [
            'Mark Ingram', 'Marvin Jones', 'Rex Burkhead', 'Tyreek Hill',
            'Gerald Everett', 'Joe Mixon', 'Dalton Schultz', 'Marquise Brown',
            'Josh Jacobs', 'Chase Claypool', 'Jalen Hurts', 'Nico Collins',
            'Jalen Tolbert', 'Isaiah Spiller'
        ]
        
        print(f"\n📋 Sample Team Roster: {len(sample_roster)} players")
        
        # Aggregate individual projections to team level
        team_projection = aggregate_individual_to_team_projections(mc_data, sample_roster)
        
        # Handle roster variations
        roster_handling = handle_missing_players_and_roster_variations(
            sample_roster, 
            team_projection['available_players']
        )
        
        # Integrate with existing outputs
        integrated_results = integrate_with_existing_outputs(team_projection, bayesian_data)
        
        # Save results
        results_file = save_team_aggregation_results(integrated_results)
        
        # Generate visualizations
        plot_files = generate_team_aggregation_visualizations(integrated_results)
        
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
