#!/usr/bin/env python3
"""
Quick Bayesian Model Test
Test the Bayesian model with recent 2023-2024 data in under 5 minutes.
"""

import time

import numpy as np
import pandas as pd
import pymc as pm
from alive_progress import alive_bar


def main():
    print("=" * 60)
    print("QUICK BAYESIAN MODEL TEST (2023-2024)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load recent data
    print("📊 Loading recent data...")
    try:
        df = pd.read_csv("combined_datasets/recent_2023_2025.csv")
        print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    # Quick data preprocessing
    print("\n🔧 Quick data preprocessing...")
    
    # Filter for relevant columns and positions
    relevant_cols = ['player_display_name', 'position', 'recent_team', 'season', 'week', 
                    'fantasy_points', 'fantasy_points_ppr', 'opponent_team']
    
    df_clean = df[relevant_cols].copy()
    df_clean = df_clean.dropna()
    
    # Filter for main positions
    positions = ['QB', 'RB', 'WR', 'TE']
    df_clean = df_clean[df_clean['position'].isin(positions)]
    
    print(f"   ✅ Cleaned data: {len(df_clean):,} rows")
    print("   📊 Position breakdown:")
    for pos in positions:
        count = len(df_clean[df_clean['position'] == pos])
        print(f"      {pos}: {count:,} rows")
    
    # Create simple features
    print("\n🎯 Creating simple features...")
    
    # Calculate rolling averages (3-game) with progress bar
    print("   🔄 Calculating rolling averages...")
    with alive_bar(len(df_clean), title="Creating Features", bar="smooth") as bar:
        for player in df_clean['player_display_name'].unique():
            player_mask = df_clean['player_display_name'] == player
            player_data = df_clean[player_mask].copy()
            
            # Calculate rolling average for this player
            df_clean.loc[player_mask, 'fantasy_points_3game_avg'] = (
                player_data['fantasy_points'].rolling(3, min_periods=1).mean()
            )
            
            bar.text(f"Processing {player}")
            bar()
    
    # Home/Away indicator (simplified for now)
    df_clean['is_home'] = 0  # We'll need to add this from schedule data later
    
    # Position encoding
    position_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df_clean['position_id'] = df_clean['position'].map(position_map)
    
    print("   ✅ Features created")
    
    # Split data for quick testing
    print("\n✂️  Splitting data for testing...")
    
    # Use 2023 for training, 2024 for testing
    train = df_clean[df_clean['season'] == 2023].copy()
    test = df_clean[df_clean['season'] == 2024].copy()
    
    print(f"   📈 Training: {len(train):,} rows (2023)")
    print(f"   🧪 Testing: {len(test):,} rows (2024)")
    
    if len(train) < 100 or len(test) < 50:
        print("   ⚠️  Not enough data for meaningful testing")
        return
    
    # Simple Bayesian model (quick version)
    print("\n🧠 Building simple Bayesian model...")
    
    with pm.Model() as simple_model:
        # Priors
        intercept = pm.Normal('intercept', mu=10, sigma=5)
        position_effect = pm.Normal('position_effect', mu=0, sigma=2, shape=4)
        home_effect = pm.Normal('home_effect', mu=0, sigma=1)
        avg_effect = pm.Normal('avg_effect', mu=0.5, sigma=0.2)
        
        # Standard deviation
        sigma = pm.HalfNormal('sigma', sigma=5)
        
        # Expected value
        mu = (intercept + 
              position_effect[train['position_id']] + 
              home_effect * train['is_home'] + 
              avg_effect * train['fantasy_points_3game_avg'])
        
        # Likelihood
        fantasy_points = pm.Normal('fantasy_points', mu=mu, sigma=sigma, observed=train['fantasy_points'])
        
        # Quick sampling (small number of draws)
        print("   🔄 Sampling (quick version)...")
        trace = pm.sample(
            draws=500,  # Reduced for speed
            tune=200,   # Reduced for speed
            cores=1,    # Single core for stability
            return_inferencedata=True,
            random_seed=42
        )
    
    print(f"   ✅ Model training completed in {time.time() - start_time:.1f} seconds")
    
    # Quick predictions
    print("\n🔮 Making quick predictions...")
    
    with simple_model:
        # Get posterior means
        intercept_val = float(trace.posterior['intercept'].mean().values)
        home_effect_val = float(trace.posterior['home_effect'].mean().values)
        avg_effect_val = float(trace.posterior['avg_effect'].mean().values)
        position_effects = trace.posterior['position_effect'].mean(dim=('chain', 'draw')).values
        
        print("   📊 Model parameters:")
        print(f"      Intercept: {intercept_val:.2f}")
        print(f"      Home effect: {home_effect_val:.2f}")
        print(f"      Average effect: {avg_effect_val:.2f}")
        print(f"      Position effects: {position_effects}")
        
        # Calculate predictions safely
        predictions = []
        for idx, row in test.iterrows():
            pos_id = int(row['position_id'])
            if pos_id < len(position_effects):
                pred = float(intercept_val + 
                       position_effects[pos_id] + 
                       home_effect_val * row['is_home'] + 
                       avg_effect_val * row['fantasy_points_3game_avg'])
                predictions.append(pred)
            else:
                # Fallback for unknown positions
                pred = float(intercept_val + 
                       home_effect_val * row['is_home'] + 
                       avg_effect_val * row['fantasy_points_3game_avg'])
                predictions.append(pred)
        
        predictions = np.array(predictions)
    
    # Quick evaluation
    print("\n📊 Quick evaluation...")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(test['fantasy_points'], predictions)
    rmse = np.sqrt(mean_squared_error(test['fantasy_points'], predictions))
    
    # Baseline (just use 3-game average)
    baseline_mae = mean_absolute_error(test['fantasy_points'], test['fantasy_points_3game_avg'])
    baseline_rmse = np.sqrt(mean_squared_error(test['fantasy_points'], test['fantasy_points_3game_avg']))
    
    print("   📈 Model Performance:")
    print(f"      MAE: {mae:.2f} (vs baseline: {baseline_mae:.2f})")
    print(f"      RMSE: {rmse:.2f} (vs baseline: {baseline_rmse:.2f})")
    
    improvement_mae = ((baseline_mae - mae) / baseline_mae) * 100
    print(f"      Improvement: {improvement_mae:.1f}%")
    
    # Model summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    
    total_time = time.time() - start_time
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📊 Data used: {len(train):,} training, {len(test):,} test")
    print(f"🎯 Model performance: {improvement_mae:.1f}% improvement over baseline")
    
    if improvement_mae > 0:
        print("✅ Model shows improvement - ready for full implementation!")
    else:
        print("⚠️  Model needs tuning - consider feature engineering")
    
    print("\n🎯 Next steps:")
    print("   1. Add more features (weather, injuries, etc.)")
    print("   2. Test with larger dataset")
    print("   3. Implement weekly predictions")

if __name__ == "__main__":
    main()
