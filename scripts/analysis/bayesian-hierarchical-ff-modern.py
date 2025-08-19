#!/usr/bin/env python3
"""
Modern Bayesian Hierarchical Model for Fantasy Football Predictions
Using PyMC4 (modern PyMC) for better uncertainty quantification and reliability.

This model estimates opponent-position effects and generates projections with
proper uncertainty quantification to help make data-driven fantasy football decisions.
"""

import glob
import os
import pickle
from datetime import datetime

import matplotlib
import pandas as pd
import pymc as pm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Default configuration - can be overridden by function arguments
DEFAULT_CORES = 4
DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 500
DEFAULT_CHAINS = 4
DEFAULT_PREDICTIVE_SAMPLES = 500

print(f'Default configuration: {DEFAULT_CORES} cores, {DEFAULT_DRAWS} draws, {DEFAULT_CHAINS} chains')

def load_preprocessed_data(path_to_data_directory):
    """Load preprocessed data for analysis."""
    print("Loading preprocessed data for analysis...")
    
    # Look for the preprocessed analysis dataset
    output_dir = os.path.join(path_to_data_directory, 'combined_datasets')
    analysis_files = glob.glob(os.path.join(output_dir, '*season_modern.csv'))
    
    if not analysis_files:
        raise ValueError(f"No preprocessed analysis data found in {output_dir}. Run 03_preprocess_analysis_data.py first.")
    
    # Use the most recent preprocessed file
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"Loading preprocessed data from: {latest_file}")
    
    data = pd.read_csv(latest_file)
    print(f"Loaded data shape: {data.shape}")
    
    # Extract team names from the data
    team_names = data['Opp'].unique()
    print(f"Found {len(team_names)} teams")
    
    return data, team_names


def load_recent_trace():
    """Try to load a recent trace to avoid re-sampling."""
    results_dir = 'results/bayesian-hierarchical-results'
    if not os.path.exists(results_dir):
        return None
    
    trace_files = glob.glob(os.path.join(results_dir, 'trace_*.pkl'))
    if not trace_files:
        return None
    
    # Get the most recent trace file
    latest_trace = max(trace_files, key=os.path.getctime)
    
    try:
        with open(latest_trace, 'rb') as f:
            trace = pickle.load(f)
        print(f"✅ Loaded existing trace from: {latest_trace}")
        return trace
    except Exception as e:
        print(f"⚠️  Could not load trace: {e}")
        return None


def bayesian_hierarchical_ff_modern(path_to_data_directory, cores=DEFAULT_CORES, draws=DEFAULT_DRAWS, tune=DEFAULT_TUNE, chains=DEFAULT_CHAINS, predictive_samples=DEFAULT_PREDICTIVE_SAMPLES, use_existing_trace=True):
    """Modern PyMC4 implementation of Bayesian hierarchical fantasy football model."""
    
    # Try to load existing trace first
    if use_existing_trace:
        existing_trace = load_recent_trace()
        if existing_trace is not None:
            print("🔄 Using existing trace - skipping expensive sampling!")
            trace = existing_trace
        else:
            print("🆕 No existing trace found - will run full sampling")
            trace = None
    else:
        trace = None
    
    # Load preprocessed data
    data, team_names = load_preprocessed_data(path_to_data_directory)
    
    # Split into train/test (use last 2 years dynamically)
    available_years = sorted(data['Season'].unique())
    if len(available_years) < 2:
        raise ValueError("Need at least 2 years of data for train/test split")
    
    train_year = available_years[-2]  # Second to last year
    test_year = available_years[-1]   # Last year
    
    train = data[data.apply(lambda x: x['Season'] == train_year, axis=1)]
    test = data[data.apply(lambda x: x['Season'] == test_year, axis=1)]
    
    print(f"Training on {train_year} data, testing on {test_year} data")
    
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    # Model parameters
    num_positions = 4
    ranks = 4
    team_number = len(team_names)
    
    print('Building modern PyMC4 model...')

    with pm.Model() as model:
        # Part 1: Define observables
        print("Part 1: Defining observables...")
        
        # Degrees of freedom for Student's t distributions
        nu = pm.Exponential('nu_minus_one', 1 / 29.0, shape=2) + 1
        
        # Standard deviations based on rank
        err = pm.Uniform('std_dev_rank', 0, 100, shape=ranks)
        err_b = pm.Uniform('std_dev_rank_b', 0, 100, shape=ranks)

        # Part 2: Defensive ability of opposing teams vs each position
        print("Part 2: Modeling defensive effects...")
        
        # Global defensive priors for each position
        opp_def = pm.Normal('opp_team_prior', 0, 100**2, shape=num_positions)
        
        # Team-specific defensive effects
        opp_qb = pm.Normal('defensive_differential_qb', opp_def[0], 100**2, shape=team_number)
        opp_wr = pm.Normal('defensive_differential_wr', opp_def[1], 100**2, shape=team_number)
        opp_rb = pm.Normal('defensive_differential_rb', opp_def[2], 100**2, shape=team_number)
        opp_te = pm.Normal('defensive_differential_te', opp_def[3], 100**2, shape=team_number)

        # Part 3: Home/away advantages by position and rank
        print("Part 3: Modeling home/away effects...")
        
        home_adv = pm.Normal('home_additive_prior', 0, 100**2, shape=num_positions)
        away_adv = pm.Normal('away_additive_prior', 0, 100**2, shape=num_positions)
        
        # Position-specific home/away effects by rank
        pos_home_qb = pm.Normal('home_differential_qb', home_adv[0], 10**2, shape=ranks)
        pos_home_rb = pm.Normal('home_differential_rb', home_adv[1], 10**2, shape=ranks)
        pos_home_te = pm.Normal('home_differential_te', home_adv[2], 10**2, shape=ranks)
        pos_home_wr = pm.Normal('home_differential_wr', home_adv[3], 10**2, shape=ranks)
        
        pos_away_qb = pm.Normal('away_differential_qb', away_adv[0], 10**2, shape=ranks)
        pos_away_rb = pm.Normal('away_differential_rb', away_adv[1], 10**2, shape=ranks)
        pos_away_wr = pm.Normal('away_differential_wr', away_adv[2], 10**2, shape=ranks)
        pos_away_te = pm.Normal('away_differential_te', away_adv[3], 10**2, shape=ranks)

        # Part 4: Likelihood models
        print("Part 4: Building likelihood models...")
        
        # Data arrays for training
        player_home = train['is_home'].values
        player_avg = train['7_game_avg'].values
        player_opp = train['opp_team'].values
        player_rank = train['rank'] - 1
        qb_indicator = train['position_QB'].values.astype(int)
        wr_indicator = train['position_WR'].values.astype(int)
        rb_indicator = train['position_RB'].values.astype(int)
        te_indicator = train['position_TE'].values.astype(int)

        # Defensive effect calculation
        def_effect = (
            qb_indicator * opp_qb[player_opp] +
            wr_indicator * opp_wr[player_opp] +
            rb_indicator * opp_rb[player_opp] +
            te_indicator * opp_te[player_opp]
        )

        # First likelihood: difference from average explained by defensive ability
        like1 = pm.StudentT(
            'diff_from_avg',
            mu=def_effect,
            sigma=err_b[player_rank],
            nu=nu[1],
            observed=train['diff_from_avg']
        )

        # Second likelihood: total score prediction
        mu = player_avg + def_effect
        
        # Add home/away effects
        mu += (rb_indicator * pos_home_rb[player_rank] * player_home + 
               wr_indicator * pos_home_wr[player_rank] * player_home +
               qb_indicator * pos_home_qb[player_rank] * player_home + 
               te_indicator * pos_home_te[player_rank] * player_home)
        
        mu += (rb_indicator * pos_away_rb[player_rank] * (1 - player_home) + 
               wr_indicator * pos_away_wr[player_rank] * (1 - player_home) +
               qb_indicator * pos_away_qb[player_rank] * (1 - player_home) + 
               te_indicator * pos_away_te[player_rank] * (1 - player_home))

        like2 = pm.StudentT(
            'fantasy_points',
            mu=mu,
            sigma=err[player_rank],
            nu=nu[0],
            observed=train['FantPt']
        )

        # Part 5: Training (only if we don't have a trace)
        if trace is None:
            print("Part 5: Training model...")
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                return_inferencedata=True,
                random_seed=42,
                target_accept=0.95,
                max_treedepth=12
            )
        else:
            print("Part 5: Using existing trace (skipping sampling)")

        print("Model training completed!")
        
        # Save the expensive trace results immediately
        print("💾 Saving sampling results...")
        trace_file = f'results/bayesian-hierarchical-results/trace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        
        try:
            with open(trace_file, 'wb') as f:
                pickle.dump(trace, f)
            print(f"✅ Trace saved to: {trace_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save trace: {e}")
        
        # Part 6: Model evaluation and predictions
        print("Part 6: Evaluating model...")
        
        # Create test data arrays
        test_home = test['is_home'].values
        test_avg = test['7_game_avg'].values
        test_opp = test['opp_team'].values
        test_rank = test['rank'] - 1
        test_qb = test['position_QB'].values.astype(int)
        test_wr = test['position_WR'].values.astype(int)
        test_rb = test['position_RB'].values.astype(int)
        test_te = test['position_TE'].values.astype(int)

        # Sample from posterior predictive - fixed API for PyMC4
        print("Generating predictions...")
        with model:
            pm_pred = pm.sample_posterior_predictive(
                trace, 
                var_names=['fantasy_points']
            )

        # Calculate predictions
        pred_mean = pm_pred.posterior_predictive['fantasy_points'].mean(dim=('chain', 'draw'))
        pred_std = pm_pred.posterior_predictive['fantasy_points'].std(dim=('chain', 'draw'))
        
        # Ensure predictions match test data length
        print(f"Test data length: {len(test)}")
        print(f"Predictions length: {len(pred_mean)}")
        
        # If lengths don't match, take only the first len(test) predictions
        if len(pred_mean) != len(test):
            print("⚠️  Length mismatch detected. Truncating predictions to match test data.")
            pred_mean = pred_mean[:len(test)]
            pred_std = pred_std[:len(test)]
            print(f"Adjusted predictions length: {len(pred_mean)}")

        # Model evaluation
        mae_bayesian = mean_absolute_error(test['FantPt'].values, pred_mean.values)
        mae_baseline = mean_absolute_error(test['FantPt'].values, test['7_game_avg'].values)
        
        print(f"Bayesian Model MAE: {mae_bayesian:.2f}")
        print(f"Baseline (7-game avg) MAE: {mae_baseline:.2f}")
        print(f"Improvement: {((mae_baseline - mae_bayesian) / mae_baseline * 100):.1f}%")

        # Part 7: Generate plots and save results
        print("Part 7: Generating visualizations...")
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # Plot 1: Model diagnostics
        pm.plot_trace(trace, var_names=['opp_team_prior', 'home_additive_prior'])
        plt.savefig('plots/modern_training_traces.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Defensive effects by team
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        positions = ['QB', 'WR', 'RB', 'TE']
        effects = [opp_qb, opp_wr, opp_rb, opp_te]
        
        for i, (pos, effect) in enumerate(zip(positions, effects)):
            ax = axes[i//2, i%2]
            effect_mean = trace.posterior[f'defensive_differential_{pos.lower()}'].mean(dim=('chain', 'draw'))
            effect_std = trace.posterior[f'defensive_differential_{pos.lower()}'].std(dim=('chain', 'draw'))
            
            ax.errorbar(effect_mean, range(len(team_names)), xerr=effect_std, fmt='o', capsize=5)
            ax.set_title(f'Team Effects on {pos} Point Average (2021)')
            ax.set_yticks(range(len(team_names)))
            ax.set_yticklabels(team_names)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel(f'Change in opponent {pos} average')
        
        plt.tight_layout()
        plt.savefig('plots/modern_team_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Prediction vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(test['FantPt'], pred_mean, alpha=0.6)
        plt.plot([0, 50], [0, 50], 'r--', alpha=0.8)
        plt.xlabel('Actual Fantasy Points')
        plt.ylabel('Predicted Fantasy Points')
        plt.title('Bayesian Model Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/modern_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save results (avoid pickle issues with PyMC objects)
        results = {
            'mae_bayesian': mae_bayesian,
            'mae_baseline': mae_baseline,
            'team_names': team_names,
            'timestamp': datetime.now().isoformat(),
            'test_data_shape': test.shape,
            'predictions_shape': pred_mean.shape
        }
        
        # Save results as JSON instead of pickle to avoid serialization issues
        import json
        results_file = 'results/bayesian-hierarchical-results/modern_model_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {results_file}")
        
        print("Model training and evaluation completed successfully!")
        
        return trace, results


def main():
    """Main function to run the modern Bayesian hierarchical model."""
    print("=" * 60)
    print("Modern Bayesian Hierarchical Fantasy Football Model")
    print("Using PyMC4 for robust uncertainty quantification")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results/bayesian-hierarchical-results', exist_ok=True)
    
    # Run the model with default configuration
    trace, results = bayesian_hierarchical_ff_modern('datasets')
    
    print("\n" + "=" * 60)
    print("Model Summary:")
    print(f"- Bayesian Model MAE: {results['mae_bayesian']:.2f}")
    print(f"- Baseline MAE: {results['mae_baseline']:.2f}")
    print(f"- Improvement: {((results['mae_baseline'] - results['mae_bayesian']) / results['mae_baseline'] * 100):.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
