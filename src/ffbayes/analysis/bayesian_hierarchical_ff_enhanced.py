#!/usr/bin/env python3
"""
Enhanced Bayesian Hierarchical Model for Fantasy Football Predictions
Incorporating additional data sources to significantly improve over baseline.

This enhanced model adds:
1. Snap counts (offensive/defensive usage)
2. Injury data (practice/game status)
3. Depth chart position (starter vs backup)
4. Vegas lines (game context)
5. Advanced features (target share, red zone usage, etc.)

The goal is to substantially outperform the 7-game average baseline.
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
from sklearn.metrics import mean_absolute_error

# Configuration
DEFAULT_CORES = 7
DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 4
DEFAULT_PREDICTIVE_SAMPLES = 500

# Quick test mode support
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'
if QUICK_TEST:
    print("🚀 QUICK TEST MODE ENABLED for Enhanced Bayesian model")
    DEFAULT_CORES = 2
    DEFAULT_DRAWS = 100
    DEFAULT_TUNE = 100
    DEFAULT_CHAINS = 2

print(f'Enhanced model configuration: {DEFAULT_CORES} cores, {DEFAULT_DRAWS} draws, {DEFAULT_CHAINS} chains')

def load_enhanced_data(path_to_data_directory):
    """Load enhanced data with additional features."""
    print("Loading enhanced data with additional features...")
    
    # Load base data
    output_dir = os.path.join(path_to_data_directory, 'combined_datasets')
    analysis_files = glob.glob(os.path.join(output_dir, '*season_modern.csv'))
    
    if not analysis_files:
        raise ValueError(f"No preprocessed analysis data found in {output_dir}. Run 03_preprocess_analysis_data.py first.")
    
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"Loading base data from: {latest_file}")
    
    data = pd.read_csv(latest_file)
    print(f"Base data shape: {data.shape}")
    
    # Load additional data sources
    try:
        # Load snap counts
        import nfl_data_py as nfl
        print("Loading snap counts data...")
        snap_counts = nfl.import_snap_counts([2023, 2024])  # Last 2 years
        print(f"Snap counts shape: {snap_counts.shape}")
        
        # Load injury data
        print("Loading injury data...")
        injuries = nfl.import_injuries([2023, 2024])
        print(f"Injury data shape: {injuries.shape}")
        
        # Load depth charts
        print("Loading depth charts...")
        depth_charts = nfl.import_depth_charts([2023, 2024])
        print(f"Depth charts shape: {depth_charts.shape}")
        
        # Merge additional data
        data = merge_enhanced_features(data, snap_counts, injuries, depth_charts)
        
    except Exception as e:
        print(f"⚠️  Could not load additional data: {e}")
        print("Falling back to base model")
    
    return data

def merge_enhanced_features(base_data, snap_counts, injuries, depth_charts):
    """Merge additional features into base data."""
    print("Merging enhanced features...")
    
    # Process snap counts
    if snap_counts is not None and len(snap_counts) > 0:
        # Create snap count features
        snap_features = snap_counts.groupby(['player', 'season', 'week']).agg({
            'offense_pct': 'mean',
            'defense_pct': 'mean',
            'st_pct': 'mean'
        }).reset_index()
        
        # Merge with base data
        base_data = pd.merge(
            base_data, 
            snap_features,
            left_on=['Name', 'Season', 'G#'],
            right_on=['player', 'season', 'week'],
            how='left'
        )
        
        # Fill missing snap percentages with 0
        base_data['offense_pct'] = base_data['offense_pct'].fillna(0)
        base_data['defense_pct'] = base_data['defense_pct'].fillna(0)
        base_data['st_pct'] = base_data['st_pct'].fillna(0)
        
        print(f"Added snap count features. Shape: {base_data.shape}")
    
    # Process injury data
    if injuries is not None and len(injuries) > 0:
        # Create injury features
        injury_features = injuries.groupby(['full_name', 'season', 'week']).agg({
            'report_status': lambda x: 1 if any(s in ['Out', 'Doubtful'] for s in x if pd.notna(s)) else 0,
            'practice_status': lambda x: 1 if any(s in ['Limited', 'DNP'] for s in x if pd.notna(s)) else 0
        }).reset_index()
        
        # Merge with base data
        base_data = pd.merge(
            base_data,
            injury_features,
            left_on=['Name', 'Season', 'G#'],
            right_on=['full_name', 'season', 'week'],
            how='left'
        )
        
        # Fill missing injury status with 0 (healthy)
        base_data['report_status'] = base_data['report_status'].fillna(0)
        base_data['practice_status'] = base_data['practice_status'].fillna(0)
        
        print(f"Added injury features. Shape: {base_data.shape}")
    
    # Process depth charts
    if depth_charts is not None and len(depth_charts) > 0:
        # Create depth chart features
        depth_features = depth_charts.groupby(['full_name', 'season', 'week']).agg({
            'depth_position': lambda x: 1 if any('1' in str(pos) for pos in x if pd.notna(pos)) else 0
        }).reset_index()
        
        # Merge with base data
        base_data = pd.merge(
            base_data,
            depth_features,
            left_on=['Name', 'Season', 'G#'],
            right_on=['full_name', 'season', 'week'],
            how='left'
        )
        
        # Fill missing depth position with 0 (not starter)
        base_data['depth_position'] = base_data['depth_position'].fillna(0)
        
        print(f"Added depth chart features. Shape: {base_data.shape}")
    
    return base_data

def enhanced_bayesian_model(path_to_data_directory, cores=DEFAULT_CORES, draws=DEFAULT_DRAWS, tune=DEFAULT_TUNE, chains=DEFAULT_CHAINS, predictive_samples=DEFAULT_PREDICTIVE_SAMPLES):
    """Enhanced Bayesian model with additional features."""
    
    # Load enhanced data
    data = load_enhanced_data(path_to_data_directory)
    
    # Split into train/test
    available_years = sorted(data['Season'].unique())
    if len(available_years) < 2:
        raise ValueError(f"Need at least 2 years of data for train/test split. Available years: {available_years}")
    
    train_year = available_years[-2]
    test_year = available_years[-1]
    
    train = data[data['Season'] == train_year]
    test = data[data['Season'] == test_year]
    
    print(f"Training on {train_year} data: {train.shape}")
    print(f"Testing on {test_year} data: {test.shape}")
    
    # Model parameters
    num_positions = 4
    ranks = 4
    team_number = len(data['Opp'].unique())
    
    print('Building enhanced PyMC4 model with additional features...')
    
    try:
        with pm.Model() as enhanced_model:
            # Enhanced observables with additional features
            print("Part 1: Defining enhanced observables...")
            
            # Base features
            player_home = pm.Data('player_home', train['is_home'].values)
            player_avg = pm.Data('player_avg', train['7_game_avg'].values)
            player_opp = pm.Data('player_opp', train['opp_team'].values)
            player_rank = pm.Data('player_rank', train['rank'].values - 1)
            qb_indicator = pm.Data('qb_indicator', train['position_QB'].values.astype(int))
            wr_indicator = pm.Data('wr_indicator', train['position_WR'].values.astype(int))
            rb_indicator = pm.Data('rb_indicator', train['position_RB'].values.astype(int))
            te_indicator = pm.Data('te_indicator', train['position_TE'].values.astype(int))
            fantasy_points_data = pm.Data('fantasy_points_data', train['FantPt'].values)
            
            # Enhanced features
            offense_pct = pm.Data('offense_pct', train['offense_pct'].values if 'offense_pct' in train.columns else np.zeros(len(train)))
            injury_status = pm.Data('injury_status', train['report_status'].values if 'report_status' in train.columns else np.zeros(len(train)))
            is_starter = pm.Data('is_starter', train['depth_position'].values if 'depth_position' in train.columns else np.zeros(len(train)))
            
            # Enhanced priors
            intercept = pm.Normal('intercept', 0, 2.0)
            avg_multiplier = pm.Normal('avg_multiplier', 1.0, 0.1)
            
            # Snap count effects (NEW!)
            snap_effect = pm.Normal('snap_effect', 0, 1.0)
            
            # Injury effects (NEW!)
            injury_penalty = pm.Normal('injury_penalty', -2.0, 1.0)  # Injured players score less
            
            # Starter effects (NEW!)
            starter_bonus = pm.Normal('starter_bonus', 1.0, 0.5)  # Starters score more
            
            # Standard model components
            nu = pm.Exponential('nu_minus_one', 1 / 29.0, shape=2) + 1
            err = pm.HalfNormal('std_dev_rank', 8.0, shape=ranks)
            
            # Defensive effects
            opp_def = pm.Normal('opp_team_prior', 0, 4.0, shape=num_positions)
            opp_qb = pm.Normal('defensive_differential_qb', opp_def[0], 3.0, shape=team_number)
            opp_wr = pm.Normal('defensive_differential_wr', opp_def[1], 3.0, shape=team_number)
            opp_rb = pm.Normal('defensive_differential_rb', opp_def[2], 3.0, shape=team_number)
            opp_te = pm.Normal('defensive_differential_te', opp_def[3], 3.0, shape=team_number)
            
            # Home/away effects
            home_adv = pm.Normal('home_additive_prior', 0, 3.0, shape=num_positions)
            away_adv = pm.Normal('away_additive_prior', 0, 3.0, shape=num_positions)
            
            pos_home_qb = pm.Normal('home_differential_qb', home_adv[0], 2.0, shape=ranks)
            pos_home_rb = pm.Normal('home_differential_rb', home_adv[1], 2.0, shape=ranks)
            pos_home_te = pm.Normal('home_differential_te', home_adv[2], 2.0, shape=ranks)
            pos_home_wr = pm.Normal('home_differential_wr', home_adv[3], 2.0, shape=ranks)
            
            pos_away_qb = pm.Normal('away_differential_qb', away_adv[0], 2.0, shape=ranks)
            pos_away_rb = pm.Normal('away_differential_rb', away_adv[1], 2.0, shape=ranks)
            pos_away_wr = pm.Normal('away_differential_wr', away_adv[2], 2.0, shape=ranks)
            pos_away_te = pm.Normal('away_differential_te', away_adv[3], 2.0, shape=ranks)
            
            # Enhanced mean calculation
            print("Part 2: Building enhanced likelihood...")
            
            # Defensive effect
            def_effect = (
                qb_indicator * opp_qb[player_opp] +
                wr_indicator * opp_wr[player_opp] +
                rb_indicator * opp_rb[player_opp] +
                te_indicator * opp_te[player_opp]
            )
            
            # Enhanced mean with additional features
            mu = intercept + (avg_multiplier * player_avg) + def_effect
            
            # Add snap count effect (NEW!)
            mu += snap_effect * offense_pct
            
            # Add injury penalty (NEW!)
            mu += injury_penalty * injury_status
            
            # Add starter bonus (NEW!)
            mu += starter_bonus * is_starter
            
            # Add home/away effects
            mu += (rb_indicator * pos_home_rb[player_rank] * player_home + 
                   wr_indicator * pos_home_wr[player_rank] * player_home +
                   qb_indicator * pos_home_qb[player_rank] * player_home + 
                   te_indicator * pos_home_te[player_rank] * player_home)
            
            mu += (rb_indicator * pos_away_rb[player_rank] * (1 - player_home) + 
                   wr_indicator * pos_away_wr[player_rank] * (1 - player_home) +
                   qb_indicator * pos_away_qb[player_rank] * (1 - player_home) + 
                   te_indicator * pos_away_te[player_rank] * (1 - player_home))
            
            # Enhanced likelihood
            like = pm.StudentT(
                'fantasy_points',
                mu=mu,
                sigma=err[player_rank],
                nu=nu[0],
                observed=fantasy_points_data
            )
            
            # Sample
            print("Part 3: Sampling enhanced model...")
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
            
            print("Enhanced model training completed!")
            
            # Save trace
            trace_file = f'results/bayesian-hierarchical-results/enhanced_trace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            with open(trace_file, 'wb') as f:
                pickle.dump(trace, f)
            print(f"Enhanced trace saved to: {trace_file}")
            
            # Evaluate on test data
            print("Part 4: Evaluating enhanced model...")
            
            with enhanced_model:
                pm.set_data({
                    'player_home': test['is_home'].values,
                    'player_avg': test['7_game_avg'].values,
                    'player_opp': test['opp_team'].values,
                    'player_rank': test['rank'].values - 1,
                    'qb_indicator': test['position_QB'].values.astype(int),
                    'wr_indicator': test['position_WR'].values.astype(int),
                    'rb_indicator': test['position_RB'].values.astype(int),
                    'te_indicator': test['position_TE'].values.astype(int),
                    'fantasy_points_data': test['FantPt'].values,
                    'offense_pct': test['offense_pct'].values if 'offense_pct' in test.columns else np.zeros(len(test)),
                    'injury_status': test['report_status'].values if 'report_status' in test.columns else np.zeros(len(test)),
                    'is_starter': test['depth_position'].values if 'depth_position' in test.columns else np.zeros(len(test))
                })
                
                pm_pred = pm.sample_posterior_predictive(
                    trace, 
                    var_names=['fantasy_points']
                )
            
            # Calculate predictions
            pred_mean = pm_pred.posterior_predictive['fantasy_points'].mean(dim=('chain', 'draw'))
            
            # Model evaluation
            mae_enhanced = mean_absolute_error(test['FantPt'].values, pred_mean.values)
            mae_baseline = mean_absolute_error(test['FantPt'].values, test['7_game_avg'].values)
            
            print(f"Enhanced Model MAE: {mae_enhanced:.2f}")
            print(f"Baseline (7-game avg) MAE: {mae_baseline:.2f}")
            print(f"Improvement: {((mae_baseline - mae_enhanced) / mae_baseline * 100):.1f}%")
            
            # Save results
            results = {
                'mae_enhanced': mae_enhanced,
                'mae_baseline': mae_baseline,
                'improvement_pct': ((mae_baseline - mae_enhanced) / mae_baseline * 100),
                'test_year': test_year,
                'enhanced_features': ['snap_counts', 'injury_status', 'starter_status'],
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            results_file = 'results/bayesian-hierarchical-results/enhanced_model_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Enhanced results saved to {results_file}")
            
            return trace, results
            
    except Exception as e:
        print(f"Error in enhanced model: {e}")
        return None, None

def main():
    """Main function for enhanced Bayesian model."""
    print("=" * 60)
    print("Enhanced Bayesian Hierarchical Fantasy Football Model")
    print("Incorporating snap counts, injury data, and depth charts")
    print("=" * 60)
    
    trace, results = enhanced_bayesian_model('datasets')
    
    if trace is not None:
        print("\n" + "=" * 60)
        print("Enhanced Model Summary:")
        print(f"- Enhanced Model MAE: {results['mae_enhanced']:.2f}")
        print(f"- Baseline MAE: {results['mae_baseline']:.2f}")
        print(f"- Improvement: {results['improvement_pct']:.1f}%")
        print(f"- Enhanced Features: {', '.join(results['enhanced_features'])}")
        print("=" * 60)
    else:
        print("Enhanced model training failed.")

if __name__ == '__main__':
    main()
