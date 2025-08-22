#!/usr/bin/env python3
"""
Unified Bayesian Hierarchical Model for Fantasy Football Predictions
Incorporating the most promising additional data sources to improve over baseline.

This unified model focuses on:
1. Base features (7-game average, opponent effects, home/away)
2. Snap counts (playing time - HIGH IMPACT)
3. Injury data (player availability - MEDIUM IMPACT)

The goal is to achieve 5-15% MAE improvement over the 7-game average baseline.
"""

import glob
import os
import pickle
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_error

matplotlib.use('Agg')

# Configuration
DEFAULT_CORES = 7
DEFAULT_DRAWS = 1000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 4
DEFAULT_PREDICTIVE_SAMPLES = 500

# Production mode by default - NO MORE HARDCODED TEST MODE
# Test mode must be explicitly enabled with QUICK_TEST=true
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'

if QUICK_TEST:
    print("⚠️  QUICK TEST MODE EXPLICITLY ENABLED for Unified Bayesian model")
    print("⚠️  WARNING: This will use reduced parameters and unreliable metrics")
    DEFAULT_CORES = 1
    DEFAULT_DRAWS = 20
    DEFAULT_TUNE = 20
    DEFAULT_CHAINS = 1
else:
    print("🚀 PRODUCTION MODE: Full sampling parameters enabled")
    print("✅ This will provide reliable, production-quality results")

print(f'Model configuration: {DEFAULT_CORES} cores, {DEFAULT_DRAWS} draws, {DEFAULT_CHAINS} chains')

def load_weather_data(seasons, games_data=None):
    """Load weather data for NFL games."""
    print("Loading weather data...")
    
    # TODO: Implement real weather data integration with OpenWeatherMap API
    # For now, return None to avoid using synthetic data
    print("⚠️  Weather data integration not yet implemented with real API")
    print("⚠️  Skipping weather features to avoid synthetic data in model training")
    return None

def load_vegas_odds_data(seasons, games_data=None):
    """Load Vegas odds data for NFL games."""
    print("Loading Vegas odds data...")
    
    # TODO: Implement real Vegas odds data integration
    # Potential sources: ESPN API, SportsData.io, Odds API, or historical betting line databases
    # For now, return None to avoid using synthetic data
    print("⚠️  Vegas odds data integration not yet implemented with real API")
    print("⚠️  Skipping Vegas odds features to avoid synthetic data in model training")
    return None

def load_advanced_stats(seasons):
    """Load advanced NFL statistics."""
    print("Loading advanced stats...")
    
    try:
        import nfl_data_py as nfl

        # Convert seasons to list if it's not already
        if not isinstance(seasons, (list, range)):
            seasons = list(seasons)
        
        # Ensure seasons is a proper list for nfl_data_py
        seasons_list = list(seasons) if hasattr(seasons, '__iter__') else [seasons]
        print(f"Loading data for seasons: {seasons_list}")
        
        # Load snap counts
        snap_counts = nfl.import_snap_counts(seasons_list)
        print(f"Snap counts shape: {snap_counts.shape}")
        
        # Load injury data
        injuries = nfl.import_injuries(seasons_list)
        print(f"Injury data shape: {injuries.shape}")
        
        # TODO: Load additional advanced stats from ESPN API or similar
        # Target share, red zone usage, air yards, etc.
        # For now, return only available real data
        advanced_stats = None  # Placeholder for future ESPN API integration
        
        return snap_counts, injuries, advanced_stats
        
    except Exception as e:
        print(f"⚠️  Could not load advanced stats: {e}")
        print(f"Seasons type: {type(seasons)}, value: {seasons}")
        return None, None, None

def calculate_opponent_defense_ratings(base_data):
    """Calculate sophisticated opponent defense ratings with deduplication to avoid row multiplication."""
    print("Calculating opponent defense ratings with deduplication...")
    
    # Calculate defense ratings at team-season level to avoid row multiplication
    defense_ratings = base_data.groupby(['Opp', 'Season']).agg({
        'FantPt': ['mean', 'std'],
        '7_game_avg': 'mean'
    }).reset_index()
    
    # Flatten column names
    defense_ratings.columns = ['Opp', 'Season', 'defense_fantasy_mean', 'defense_fantasy_std', 'defense_avg_opponent']
    
    # Calculate position-specific defense ratings
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = base_data[base_data['Position'] == position]
        if len(pos_data) > 0:
            pos_defense = pos_data.groupby(['Opp', 'Season']).agg({
                'FantPt': ['mean', 'std']
            }).reset_index()
            pos_defense.columns = ['Opp', 'Season', f'defense_{position.lower()}_mean', f'defense_{position.lower()}_std']
            
            # Merge with main defense ratings
            defense_ratings = pd.merge(defense_ratings, pos_defense, on=['Opp', 'Season'], how='left')
        else:
            # Add empty columns if no data for position
            defense_ratings[f'defense_{position.lower()}_mean'] = np.nan
            defense_ratings[f'defense_{position.lower()}_std'] = np.nan
    
    # Fill missing values with overall means
    for col in defense_ratings.columns:
        if col.startswith('defense_') and col.endswith('_mean'):
            defense_ratings[col] = defense_ratings[col].fillna(defense_ratings['defense_fantasy_mean'])
        elif col.startswith('defense_') and col.endswith('_std'):
            defense_ratings[col] = defense_ratings[col].fillna(defense_ratings['defense_fantasy_std'])
    
    # Calculate defense efficiency (lower is better)
    defense_ratings['defense_efficiency'] = defense_ratings['defense_fantasy_mean'] / defense_ratings['defense_avg_opponent']
    
    # Normalize defense ratings
    for col in defense_ratings.columns:
        if col.startswith('defense_') and (col.endswith('_mean') or col.endswith('_std')):
            mean_val = defense_ratings[col].mean()
            std_val = defense_ratings[col].std()
            if std_val > 0:
                defense_ratings[col] = (defense_ratings[col] - mean_val) / std_val
    
    print(f"Defense ratings shape (deduplicated): {defense_ratings.shape}")
    return defense_ratings

def add_temporal_features(base_data):
    """Add temporal features without row multiplication."""
    print("Adding temporal features...")
    
    # Week-level effects (early, mid, late season)
    base_data['week_early'] = (base_data['G#'] <= 4).astype(int)
    base_data['week_mid'] = ((base_data['G#'] > 4) & (base_data['G#'] <= 12)).astype(int)
    base_data['week_late'] = (base_data['G#'] > 12).astype(int)
    
    # Season progress (normalized week number)
    base_data['season_progress'] = (base_data['G#'] - 1) / 17.0  # Normalize to 0-1
    
    # Player experience (rookie vs veteran)
    # Assume players with more than 2 seasons are veterans
    base_data['is_rookie'] = (base_data['Season'] - base_data['Season'].min() <= 1).astype(int)
    base_data['is_veteran'] = (base_data['Season'] - base_data['Season'].min() > 2).astype(int)
    
    print(f"Added temporal features. Shape: {base_data.shape}")
    return base_data

def merge_unified_features(base_data):
    """Merge additional features into base data with deduplication to avoid row multiplication."""
    print("Merging unified features with deduplication...")
    
    # Load additional data sources
    seasons = base_data['Season'].unique()
    snap_counts, injuries, advanced_stats = load_advanced_stats(seasons)
    # weather_data = load_weather_data(seasons, base_data)  # Disabled until real API integration
    # vegas_odds_data = load_vegas_odds_data(seasons, base_data)  # Disabled until real API integration
    
    # Calculate opponent defense ratings (deduplicated at team-season level)
    defense_ratings = calculate_opponent_defense_ratings(base_data)
    
    # Add temporal features (no row multiplication)
    base_data = add_temporal_features(base_data)
    
    # Merge snap counts
    if snap_counts is not None and len(snap_counts) > 0:
        # Filter for meaningful snap counts (at least 10% offense snaps)
        meaningful_snaps = snap_counts[snap_counts['offense_pct'] >= 0.1].copy()
        print(f"Meaningful snap counts (>=10% offense): {len(meaningful_snaps)} out of {len(snap_counts)}")
        
        snap_features = meaningful_snaps.groupby(['player', 'season', 'week']).agg({
            'offense_pct': 'mean',
            'defense_pct': 'mean',
            'st_pct': 'mean'
        }).reset_index()
        
        base_data = pd.merge(
            base_data,
            snap_features,
            left_on=['Name', 'Season', 'G#'],
            right_on=['player', 'season', 'week'],
            how='left'
        )
        
        # Fill missing values with 0 (no meaningful snap data)
        base_data['offense_pct'] = base_data['offense_pct'].fillna(0)
        base_data['defense_pct'] = base_data['defense_pct'].fillna(0)
        base_data['st_pct'] = base_data['st_pct'].fillna(0)
        
        # Calculate meaningful snap coverage
        meaningful_coverage = (base_data['offense_pct'] > 0).mean() * 100
        print(f"Snap count coverage: {meaningful_coverage:.1f}% of players have meaningful snap data")
        print(f"Added snap count features. Shape: {base_data.shape}")
    
    # Merge injury data
    if injuries is not None and len(injuries) > 0:
        # Filter for meaningful injury data (not None)
        meaningful_injuries = injuries[injuries['report_status'].notna()].copy()
        print(f"Meaningful injury data (not None): {len(meaningful_injuries)} out of {len(injuries)}")
        
        # Create a function to normalize names for matching
        def normalize_name(name):
            """Normalize name for matching by removing punctuation and converting to lowercase."""
            if pd.isna(name):
                return ""
            return str(name).lower().replace('.', '').replace("'", '').strip()
        
        # Normalize names in both datasets
        meaningful_injuries['normalized_name'] = meaningful_injuries['full_name'].apply(normalize_name)
        base_data['normalized_name'] = base_data['Name'].apply(normalize_name)
        
        injury_features = meaningful_injuries.groupby(['normalized_name', 'season', 'week']).agg({
            'report_status': lambda x: 1 if any(s in ['Out', 'Doubtful'] for s in x if pd.notna(s)) else 0,
            'practice_status': lambda x: 1 if any(s in ['Limited', 'DNP'] for s in x if pd.notna(s)) else 0
        }).reset_index()
        
        # Debug merge
        print(f"Injury features shape: {injury_features.shape}")
        print("Sample injury features:")
        print(injury_features.head())
        
        base_data = pd.merge(
            base_data,
            injury_features,
            left_on=['normalized_name', 'Season', 'G#'],
            right_on=['normalized_name', 'season', 'week'],
            how='left'
        )
        
        # Fill missing values with 0 (no injury data)
        base_data['report_status'] = base_data['report_status'].fillna(0)
        base_data['practice_status'] = base_data['practice_status'].fillna(0)
        
        # Remove temporary column
        base_data = base_data.drop('normalized_name', axis=1)
        
        # Calculate injury coverage
        injury_coverage = (base_data['report_status'] > 0).mean() * 100
        print(f"Injury data coverage: {injury_coverage:.1f}% of players have injury data")
        print(f"Added injury features. Shape: {base_data.shape}")
        
        # Debug: Check if any injury data was actually merged
        if 'report_status' in base_data.columns:
            non_zero_injuries = (base_data['report_status'] > 0).sum()
            print(f"Players with injury data: {non_zero_injuries} out of {len(base_data)}")
            
            # If injury coverage is very low, disable injury features
            if non_zero_injuries < 10:  # Less than 10 players with injury data
                print("⚠️  Injury data coverage too low, disabling injury features")
                base_data['report_status'] = 0
                base_data['practice_status'] = 0
        else:
            print("Warning: report_status column not found after merge")
            base_data['report_status'] = 0
            base_data['practice_status'] = 0
    
    # Weather data integration disabled until real API implementation
    # TODO: Implement OpenWeatherMap API integration for real historical weather data
    
    # Merge defense ratings (deduplicated at team-season level to avoid row multiplication)
    base_data = pd.merge(
        base_data,
        defense_ratings,
        left_on=['Opp', 'Season'],
        right_on=['Opp', 'Season'],
        how='left'
    )
    
    # Fill missing defense ratings with 0 (neutral effect)
    defense_cols = [col for col in base_data.columns if col.startswith('defense_')]
    for col in defense_cols:
        base_data[col] = base_data[col].fillna(0)
    
    print(f"Added defense ratings. Shape: {base_data.shape}")
    print(f"Unified data shape: {base_data.shape}")
    return base_data

def unified_bayesian_model(path_to_data_directory, cores=DEFAULT_CORES, draws=DEFAULT_DRAWS, tune=DEFAULT_TUNE, chains=DEFAULT_CHAINS, predictive_samples=DEFAULT_PREDICTIVE_SAMPLES):
    """Unified Bayesian model with focused feature set."""
    
    # Load and merge unified data
    print("Loading unified data with focused features...")
    
    output_dir = os.path.join(path_to_data_directory, 'combined_datasets')
    analysis_files = glob.glob(os.path.join(output_dir, '*season_modern.csv'))
    
    if not analysis_files:
        raise ValueError(f"No preprocessed analysis data found in {output_dir}. Run 03_preprocess_analysis_data.py first.")
    
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"Loading base data from: {latest_file}")
    
    data = pd.read_csv(latest_file)
    print(f"Base data shape: {data.shape}")

    # Downsample aggressively in QUICK_TEST to ensure speed
    if QUICK_TEST:
        positions = ['QB', 'RB', 'WR', 'TE']
        years = sorted(data['Season'].unique())
        if len(years) >= 2:
            years = years[-2:]
        frames = []
        for yr in years:
            year_df = data[data['Season'] == yr]
            for p in positions:
                frames.append(year_df[year_df['Position'] == p].head(100))
        if frames:
            data = pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Downsampled for QUICK_TEST: {data.shape}")
    
    # Merge unified features
    data = merge_unified_features(data)

    # Build stable opponent team index mapping (0..k-1) across entire dataset
    opp_idx_codes, opp_uniques = pd.factorize(data['Opp'])
    data['opp_idx'] = opp_idx_codes.astype(int)
    n_teams = int(data['opp_idx'].max()) + 1
    
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
    
    print('Building unified PyMC model with focused features...')
    
    # Check if injury data is available
    has_injury_data = 'report_status' in data.columns and data['report_status'].sum() > 0
    print(f"Injury data available: {has_injury_data}")
    
    # Weather data integration disabled until real API implementation
    has_weather_data = False
    print(f"Weather data available: {has_weather_data} (disabled until real API integration)")
    
    try:
        with pm.Model() as unified_model:
            # Base observables
            print("Part 1: Defining unified observables...")
            
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
            
            # Defense features (deduplicated)
            defense_rating_qb = pm.Data('defense_rating_qb', train['defense_qb_mean'].values if 'defense_qb_mean' in train.columns else np.zeros(len(train)))
            defense_rating_rb = pm.Data('defense_rating_rb', train['defense_rb_mean'].values if 'defense_rb_mean' in train.columns else np.zeros(len(train)))
            defense_rating_wr = pm.Data('defense_rating_wr', train['defense_wr_mean'].values if 'defense_wr_mean' in train.columns else np.zeros(len(train)))
            defense_rating_te = pm.Data('defense_rating_te', train['defense_te_mean'].values if 'defense_te_mean' in train.columns else np.zeros(len(train)))
            
            # Temporal features
            week_early = pm.Data('week_early', train['week_early'].values if 'week_early' in train.columns else np.zeros(len(train)))
            week_mid = pm.Data('week_mid', train['week_mid'].values if 'week_mid' in train.columns else np.zeros(len(train)))
            week_late = pm.Data('week_late', train['week_late'].values if 'week_late' in train.columns else np.zeros(len(train)))
            season_progress = pm.Data('season_progress', train['season_progress'].values if 'season_progress' in train.columns else np.zeros(len(train)))
            is_rookie = pm.Data('is_rookie', train['is_rookie'].values if 'is_rookie' in train.columns else np.zeros(len(train)))
            is_veteran = pm.Data('is_veteran', train['is_veteran'].values if 'is_veteran' in train.columns else np.zeros(len(train)))
            
            # Weather features - disabled until real API integration
            # if has_weather_data:
            #     temperature = pm.Data('temperature', train['temperature'].values)
            #     wind_speed = pm.Data('wind_speed', train['wind_speed'].values)
            #     precipitation = pm.Data('precipitation', train['precipitation'].values)
            #     is_outdoor = pm.Data('is_outdoor', train['is_outdoor'].values)
            
            # Unified priors
            intercept = pm.Normal('intercept', 0, 2.0)
            avg_multiplier = pm.Normal('avg_multiplier', 1.0, 0.1)
            
            # Usage effects
            snap_effect = pm.Normal('snap_effect', 0, 1.0)
            
            # Injury penalty - only include if injury data is available
            if has_injury_data:
                injury_penalty = pm.Normal('injury_penalty', -2.0, 1.0)
            else:
                injury_penalty = 0  # No injury penalty if no injury data
            
            # Defense effect priors (deduplicated)
            defense_effect_strength = pm.Normal('defense_effect_strength', 0, 1.0)
            defense_sensitivity_qb = pm.Normal('defense_sensitivity_qb', 0, 0.5)
            defense_sensitivity_rb = pm.Normal('defense_sensitivity_rb', 0, 0.5)
            defense_sensitivity_wr = pm.Normal('defense_sensitivity_wr', 0, 0.5)
            defense_sensitivity_te = pm.Normal('defense_sensitivity_te', 0, 0.5)
            
            # Temporal effect priors
            week_early_effect = pm.Normal('week_early_effect', 0, 1.0)
            week_mid_effect = pm.Normal('week_mid_effect', 0, 1.0)
            week_late_effect = pm.Normal('week_late_effect', 0, 1.0)
            season_trend_effect = pm.Normal('season_trend_effect', 0, 1.0)
            experience_rookie_effect = pm.Normal('experience_rookie_effect', -1.0, 1.0)
            experience_veteran_effect = pm.Normal('experience_veteran_effect', 1.0, 1.0)
            
            # Weather effects - disabled until real API integration
            # if has_weather_data:
            #     temp_effect = pm.Normal('temp_effect', 0, 0.02)  # Small effect per degree
            #     wind_effect = pm.Normal('wind_effect', -0.1, 0.05)  # Negative effect of wind
            #     precip_penalty = pm.Normal('precip_penalty', -1.0, 0.5)  # Rain penalty
            # else:
            temp_effect = 0
            wind_effect = 0
            precip_penalty = 0
            
            # Standard model components
            nu = pm.Exponential('nu_minus_one', 1 / 29.0, shape=2) + 1
            err = pm.HalfNormal('std_dev_rank', 8.0, shape=ranks)
            
            # Defensive effects
            opp_def = pm.Normal('opp_team_prior', 0, 4.0, shape=num_positions)
            opp_qb = pm.Normal('defensive_differential_qb', opp_def[0], 3.0, shape=n_teams)
            opp_wr = pm.Normal('defensive_differential_wr', opp_def[1], 3.0, shape=n_teams)
            opp_rb = pm.Normal('defensive_differential_rb', opp_def[2], 3.0, shape=n_teams)
            opp_te = pm.Normal('defensive_differential_te', opp_def[3], 3.0, shape=n_teams)
            
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
            
            # Unified mean calculation
            print("Part 2: Building unified likelihood...")
            
            # Base effects
            def_effect = (
                qb_indicator * opp_qb[train['opp_idx']] +
                wr_indicator * opp_wr[train['opp_idx']] +
                rb_indicator * opp_rb[train['opp_idx']] +
                te_indicator * opp_te[train['opp_idx']]
            )
            
            # Start with base model
            mu = intercept + (avg_multiplier * player_avg) + def_effect
            
            # Add usage effects
            mu += snap_effect * offense_pct
            
            # Add injury effects only if injury data is available
            if has_injury_data:
                mu += injury_penalty * injury_status
            
            # Add enhanced defense effects (deduplicated)
            enhanced_defense_effect = (
                defense_effect_strength * (
                    qb_indicator * defense_sensitivity_qb * defense_rating_qb +
                    rb_indicator * defense_sensitivity_rb * defense_rating_rb +
                    wr_indicator * defense_sensitivity_wr * defense_rating_wr +
                    te_indicator * defense_sensitivity_te * defense_rating_te
                )
            )
            mu += enhanced_defense_effect
            
            # Add temporal effects
            temporal_effects = (
                week_early_effect * week_early +
                week_mid_effect * week_mid +
                week_late_effect * week_late +
                season_trend_effect * season_progress +
                experience_rookie_effect * is_rookie +
                experience_veteran_effect * is_veteran
            )
            mu += temporal_effects
            
            # Weather effects - disabled until real API integration
            # if has_weather_data:
            #     # Temperature effect (optimal around 70°F)
            #     temp_deviation = pm.math.abs(temperature - 70)
            #     mu += temp_effect * temp_deviation * is_outdoor
            #     
            #     # Wind effect (negative impact)
            #     mu += wind_effect * wind_speed * is_outdoor
            #     
            #     # Precipitation effect (negative impact)
            #     mu += precip_penalty * precipitation * is_outdoor
            
            # Add home/away effects
            mu += (rb_indicator * pos_home_rb[player_rank] * player_home + 
                   wr_indicator * pos_home_wr[player_rank] * player_home +
                   qb_indicator * pos_home_qb[player_rank] * player_home + 
                   te_indicator * pos_home_te[player_rank] * player_home)
            
            mu += (rb_indicator * pos_away_rb[player_rank] * (1 - player_home) + 
                   wr_indicator * pos_away_wr[player_rank] * (1 - player_home) +
                   qb_indicator * pos_away_qb[player_rank] * (1 - player_home) + 
                   te_indicator * pos_away_te[player_rank] * (1 - player_home))
            
            # Unified likelihood
            like = pm.StudentT(
                'fantasy_points',
                mu=mu,
                sigma=err[player_rank],
                nu=nu[0],
                observed=fantasy_points_data
            )
            
            # Optimized model architecture with enhanced sampling (Task 16.5)
            print("Part 3: Sampling unified model with optimized architecture...")
            
            # Enhanced sampling parameters for better convergence
            # Adjust initialization strategy based on number of chains
            if chains == 1:
                init_strategy = None  # Use default initialization for single chain
            else:
                init_strategy = 'jitter+adapt_diag'  # Multi-chain initialization
            
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                return_inferencedata=True,
                random_seed=42,
                target_accept=0.95,  # Optimal acceptance rate for hierarchical models
                max_treedepth=12,    # Sufficient depth for complex hierarchical structure
                initvals=init_strategy,  # Adaptive initialization strategy
                compute_convergence_checks=True  # Enable convergence diagnostics
            )
            
            print("Unified model training completed!")
            
            # Enhanced convergence diagnostics
            print("Checking model convergence...")
            try:
                import arviz as az

                # Calculate R-hat and effective sample size
                summary = az.summary(trace, round_to=4)
                print("Model convergence summary:")
                print(f"R-hat range: {summary['r_hat'].min():.3f} - {summary['r_hat'].max():.3f}")
                print(f"Effective sample size range: {summary['ess_bulk'].min():.0f} - {summary['ess_bulk'].max():.0f}")
                
                # Check for convergence issues
                max_rhat = summary['r_hat'].max()
                min_ess = summary['ess_bulk'].min()
                
                if max_rhat > 1.1:
                    print("⚠️  WARNING: Some parameters may not have converged (R-hat > 1.1)")
                else:
                    print("✅ Model convergence looks good (R-hat ≤ 1.1)")
                    
                if min_ess < 100:
                    print("⚠️  WARNING: Some parameters have low effective sample size (< 100)")
                else:
                    print("✅ Effective sample sizes are adequate (≥ 100)")
                    
            except ImportError:
                print("ArviZ not available for convergence diagnostics")
                print("Install with: pip install arviz")
            
            # Save trace with draft year instead of timestamp
            current_year = datetime.now().year
            trace_file = f'results/bayesian-hierarchical-results/unified_trace_{current_year}.pkl'
            with open(trace_file, 'wb') as f:
                pickle.dump(trace, f)
            print(f"Unified trace saved to: {trace_file}")
            
            # Evaluate on test data
            print("Part 4: Evaluating unified model...")
            
            with unified_model:
                # Enhanced test data preparation with all hierarchical features
                test_data_dict = {
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
                    'injury_status': test['report_status'].values if 'report_status' in test.columns else np.zeros(len(test))
                }
                
                # Add opponent defense ratings (Task 16.1)
                if 'defense_qb_mean' in test.columns:
                    test_data_dict['defense_rating_qb'] = test['defense_qb_mean'].values
                else:
                    test_data_dict['defense_rating_qb'] = np.zeros(len(test))
                    
                if 'defense_rb_mean' in test.columns:
                    test_data_dict['defense_rating_rb'] = test['defense_rb_mean'].values
                else:
                    test_data_dict['defense_rating_rb'] = np.zeros(len(test))
                    
                if 'defense_wr_mean' in test.columns:
                    test_data_dict['defense_rating_wr'] = test['defense_wr_mean'].values
                else:
                    test_data_dict['defense_rating_wr'] = np.zeros(len(test))
                    
                if 'defense_te_mean' in test.columns:
                    test_data_dict['defense_rating_te'] = test['defense_te_mean'].values
                else:
                    test_data_dict['defense_rating_te'] = np.zeros(len(test))
                
                # Add temporal features (Task 16.2)
                if 'week_early' in test.columns:
                    test_data_dict['week_early'] = test['week_early'].values
                else:
                    test_data_dict['week_early'] = np.zeros(len(test))
                    
                if 'week_mid' in test.columns:
                    test_data_dict['week_mid'] = test['week_mid'].values
                else:
                    test_data_dict['week_mid'] = np.zeros(len(test))
                    
                if 'week_late' in test.columns:
                    test_data_dict['week_late'] = test['week_late'].values
                else:
                    test_data_dict['week_late'] = np.zeros(len(test))
                    
                if 'season_progress' in test.columns:
                    test_data_dict['season_progress'] = test['season_progress'].values
                else:
                    test_data_dict['season_progress'] = np.zeros(len(test))
                    
                if 'is_rookie' in test.columns:
                    test_data_dict['is_rookie'] = test['is_rookie'].values
                else:
                    test_data_dict['is_rookie'] = np.zeros(len(test))
                    
                if 'is_veteran' in test.columns:
                    test_data_dict['is_veteran'] = test['is_veteran'].values
                else:
                    test_data_dict['is_veteran'] = np.zeros(len(test))
                
                # Weather data - disabled until real API integration
                # if has_weather_data:
                #     test_data_dict.update({
                #         'temperature': test['temperature'].values,
                #         'wind_speed': test['wind_speed'].values,
                #         'precipitation': test['precipitation'].values,
                #         'is_outdoor': test['is_outdoor'].values
                #     })
                
                pm.set_data(test_data_dict)
                
                pm_pred = pm.sample_posterior_predictive(
                    trace, 
                    var_names=['fantasy_points']
                )
            
            # Enhanced uncertainty quantification (Task 16.4)
            pred_mean = pm_pred.posterior_predictive['fantasy_points'].mean(dim=('chain', 'draw'))
            pred_std = pm_pred.posterior_predictive['fantasy_points'].std(dim=('chain', 'draw'))
            
            # Calculate confidence intervals (5th, 25th, 75th, 95th percentiles)
            pred_percentiles = pm_pred.posterior_predictive['fantasy_points'].quantile([0.05, 0.25, 0.75, 0.95], dim=('chain', 'draw'))
            pred_5th = pred_percentiles.sel(quantile=0.05)
            pred_25th = pred_percentiles.sel(quantile=0.25)
            pred_75th = pred_percentiles.sel(quantile=0.75)
            pred_95th = pred_percentiles.sel(quantile=0.95)
            
            # Enhanced model evaluation with uncertainty metrics
            mae_unified = mean_absolute_error(test['FantPt'].values, pred_mean.values)
            mae_baseline = mean_absolute_error(test['FantPt'].values, test['7_game_avg'].values)
            
            # Uncertainty-aware metrics
            mean_uncertainty = pred_std.mean().values
            uncertainty_ratio = mean_uncertainty / pred_mean.mean().values if pred_mean.mean().values > 0 else 0
            
            # Coverage analysis (how often actual values fall within prediction intervals)
            within_50_ci = ((test['FantPt'].values >= pred_25th.values) & 
                           (test['FantPt'].values <= pred_75th.values)).mean()
            within_90_ci = ((test['FantPt'].values >= pred_5th.values) & 
                           (test['FantPt'].values <= pred_95th.values)).mean()
            
            print(f"Unified Model MAE: {mae_unified:.2f}")
            print(f"Baseline (7-game avg) MAE: {mae_baseline:.2f}")
            print(f"Improvement: {((mae_baseline - mae_unified) / mae_baseline * 100):.1f}%")
            print(f"Mean Uncertainty: {mean_uncertainty:.2f} points")
            print(f"Uncertainty Ratio: {uncertainty_ratio:.3f}")
            print(f"50% CI Coverage: {within_50_ci:.1%}")
            print(f"90% CI Coverage: {within_90_ci:.1%}")
            
            if QUICK_TEST:
                print("WARNING: QUICK_TEST mode detected — MAE and improvement metrics are not reliable and should not be trusted for evaluation.")
            
            # Save results
            unified_features = ['snap_counts']
            if has_injury_data:
                unified_features.append('injury_status')
            # Weather features disabled until real API integration
            # if has_weather_data:
            #     unified_features.extend(['temperature', 'wind_speed', 'precipitation'])
            
            # Enhanced results with uncertainty quantification (Task 16.4)
            results = {
                'mae_unified': mae_unified,
                'mae_baseline': mae_baseline,
                'improvement_pct': ((mae_baseline - mae_unified) / mae_baseline * 100),
                'test_year': test_year,
                'unified_features': unified_features,
                'has_weather_data': has_weather_data,
                'has_injury_data': has_injury_data,
                # Enhanced uncertainty metrics
                'mean_uncertainty': float(mean_uncertainty),
                'uncertainty_ratio': float(uncertainty_ratio),
                'coverage_50_ci': float(within_50_ci),
                'coverage_90_ci': float(within_90_ci),
                'hierarchical_features': {
                    'opponent_defense': 'defense_rating_qb' in test_data_dict,
                    'temporal_hierarchies': 'week_early' in test_data_dict,
                    'enhanced_uncertainty': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            results_file = 'results/bayesian-hierarchical-results/unified_model_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Unified results saved to {results_file}")
            
            return trace, results
            
    except Exception as e:
        print(f"Error in unified model: {e}")
        return None, None

def main():
    """Main function for unified Bayesian model."""
    print("=" * 60)
    print("Unified Bayesian Hierarchical Fantasy Football Model")
    print("Incorporating snap counts and injury data")
    print("=" * 60)
    
    trace, results = unified_bayesian_model('datasets')
    
    if trace is not None:
        print("\n" + "=" * 60)
        print("Enhanced Hierarchical Bayesian Model Summary:")
        print(f"- Unified Model MAE: {results['mae_unified']:.2f}")
        print(f"- Baseline MAE: {results['mae_baseline']:.2f}")
        print(f"- Improvement: {results['improvement_pct']:.1f}%")
        print(f"- Unified Features: {', '.join(results['unified_features'])}")
        
        # Enhanced uncertainty metrics
        if 'mean_uncertainty' in results:
            print(f"- Mean Uncertainty: {results['mean_uncertainty']:.2f} points")
            print(f"- Uncertainty Ratio: {results['uncertainty_ratio']:.3f}")
            print(f"- 50% CI Coverage: {results['coverage_50_ci']:.1%}")
            print(f"- 90% CI Coverage: {results['coverage_90_ci']:.1%}")
        
        # Hierarchical features status
        if 'hierarchical_features' in results:
            print("\nHierarchical Features Implemented:")
            for feature, status in results['hierarchical_features'].items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {feature.replace('_', ' ').title()}")
        
        print("=" * 60)
    else:
        print("Enhanced hierarchical model training failed.")

if __name__ == '__main__':
    main()
