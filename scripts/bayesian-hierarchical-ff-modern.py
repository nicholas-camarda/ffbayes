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
import numpy as np
import pandas as pd
import pymc as pm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Configuration
CORES = 7  # multiprocessing.cpu_count() - 1
print(f'Using {CORES} cores')

def create_dataset(path_to_data_directory):
    """Create and preprocess the fantasy football dataset."""
    print("Loading and preprocessing data...")
    
    # Read in the datasets and combine
    all_files = glob.glob(os.path.join(path_to_data_directory, '*.csv'))
    data_temp = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # Sort properly
    data = data_temp.sort_values(
        by=['Season', 'Name', 'G#'], ascending=[True, True, True]
    )

    # One-hot-encode the positions
    data['pos_id'] = data['Position']
    data['position'] = data['Position']
    data = pd.get_dummies(data, columns=['position'])

    # Identify teams with integer encoding
    ids = np.array([k for k in data['Opp'].unique()])
    team_names = ids.copy()
    data['opp_team'] = data['Opp'].apply(lambda x: np.where(x == ids)[0][0])
    data['team'] = data['Tm'].apply(lambda x: np.where(x == ids)[0][0])

    # Create home/away indicator - Away column contains team names, so if Away == Tm, it's away
    data['is_home'] = (data['Away'] != data['Tm']).astype(int)

    # Position encoding
    pos_ids = np.array([k for k in data['pos_id'].unique()])
    pos_ids_nonan = pos_ids[np.where(pos_ids != 'nan')]
    onehot_pos_ids = list(map(int, data['pos_id'].isin(pos_ids_nonan)))
    data['pos_id'] = onehot_pos_ids

    # Calculate seven game rolling average
    num_day_roll_avg = 7
    data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].transform(
        lambda x: x.rolling(num_day_roll_avg, min_periods=num_day_roll_avg).mean()
    )

    # Rank based on the 7-game average
    ranks = data.groupby(['Name', 'Season'])['7_game_avg'].rank(
        pct=False, method='average'
    )
    quartile_ranks = pd.qcut(ranks, 4, labels=False, duplicates='drop')
    data['rank'] = quartile_ranks.tolist()

    data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

    # Remove all NA and convert rank to integer
    data = data.dropna(axis=0)
    data = data.astype({'rank': int})

    # Save processed data
    data.to_csv('combined_datasets/2017-2021season_modern.csv')
    print(f"Processed data shape: {data.shape}")
    
    return data, team_names


def bayesian_hierarchical_ff_modern(path_to_data_directory, cores=CORES):
    """Modern PyMC4 implementation of Bayesian hierarchical fantasy football model."""
    
    # Load and preprocess data
    data, team_names = create_dataset(path_to_data_directory)
    
    # Split into train/test (2020 for training, 2021 for testing)
    train = data[data.apply(lambda x: x['Season'] == 2020, axis=1)]
    test = data[data.apply(lambda x: x['Season'] == 2021, axis=1)]
    
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

        # Part 5: Training
        print("Part 5: Training model...")
        trace = pm.sample(
            draws=1000,  # Reduced from 2000
            tune=500,    # Reduced from 1000
            cores=cores,
            return_inferencedata=True,
            random_seed=42,
            target_accept=0.95,  # Increased target acceptance
            max_treedepth=12     # Increased max tree depth
        )

        print("Model training completed!")
        
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

        # Save results
        results = {
            'model': model,
            'trace': trace,
            'predictions': pm_pred,
            'test_data': test,
            'mae_bayesian': mae_bayesian,
            'mae_baseline': mae_baseline,
            'team_names': team_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('results/bayesian-hierarchical-results/modern_model_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("Results saved to results/bayesian-hierarchical-results/modern_model_results.pkl")
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
    
    # Run the model
    trace, results = bayesian_hierarchical_ff_modern('datasets', cores=CORES)
    
    print("\n" + "=" * 60)
    print("Model Summary:")
    print(f"- Bayesian Model MAE: {results['mae_bayesian']:.2f}")
    print(f"- Baseline MAE: {results['mae_baseline']:.2f}")
    print(f"- Improvement: {((results['mae_baseline'] - results['mae_bayesian']) / results['mae_baseline'] * 100):.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
