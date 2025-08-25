#!/usr/bin/env python3
"""
baseline_naive_model.py - Baseline (Naive) Fantasy Football Model
Simple baseline model using 7-game averages for comparison.

This model provides a baseline against which to compare:
- Monte Carlo model
- Bayesian model
- Any other advanced models

Uses the unified dataset for consistency.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_unified_dataset(data_directory='datasets'):
    """Load the unified dataset."""
    from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset
    return load_unified_dataset(data_directory)

def create_baseline_predictions(data, target_players=None):
    """Create baseline predictions using 7-game averages."""
    print("📊 Creating baseline predictions...")
    
    if target_players is None:
        # Use players from recent seasons for consistency with Hybrid MC
        recent_seasons = [data['Season'].max() - 1, data['Season'].max()]
        target_players = data[data['Season'].isin(recent_seasons)]['Name'].unique()
        print(f"   Using players from seasons {recent_seasons}: {len(target_players)} players")
        print("   Note: Baseline still uses single-season 7-game averages for simplicity")
    else:
        print(f"   Using specified target players: {len(target_players)} players")
    
    predictions = {}
    
    for player_name in target_players:
        # Get player's data from the most recent season (baseline approach)
        # But first check if player exists in recent seasons
        recent_seasons = [data['Season'].max() - 1, data['Season'].max()]
        player_exists = data[(data['Name'] == player_name) & (data['Season'].isin(recent_seasons))]
        
        if len(player_exists) == 0:
            print(f"   ⚠️  Player {player_name} not found in recent seasons {recent_seasons}")
            continue
            
        # Get data from most recent season for baseline prediction
        player_data = data[(data['Name'] == player_name) & (data['Season'] == data['Season'].max())]
        
        if len(player_data) == 0:
            # Fallback to any recent season data
            player_data = player_exists
        
        # Use 7-game average as baseline prediction
        if '7_game_avg' in player_data.columns:
            baseline_prediction = player_data['7_game_avg'].iloc[-1]  # Most recent game
        else:
            # Fallback to simple average if 7-game average not available
            baseline_prediction = player_data['FantPt'].mean()
        
        predictions[player_name] = {
            'mean': float(baseline_prediction),
            'std': float(player_data['FantPt'].std()),
            'confidence_interval': [
                float(baseline_prediction - player_data['FantPt'].std()),
                float(baseline_prediction + player_data['FantPt'].std())
            ],
            'position': str(player_data['Position'].iloc[0]),
            'team': str(player_data['Tm'].iloc[0]) if 'Tm' in player_data.columns else None,
            'method': '7_game_average_baseline',
            'data_points': len(player_data)
        }
    
    print(f"✅ Created baseline predictions for {len(predictions)} players")
    return predictions

def evaluate_baseline_model(data, test_season=None):
    """Evaluate baseline model performance on historical data."""
    print("📈 Evaluating baseline model...")
    
    if test_season is None:
        # Use second-to-last season for testing
        available_seasons = sorted(data['Season'].unique())
        if len(available_seasons) < 2:
            raise ValueError("Need at least 2 seasons for train/test evaluation")
        
        train_season = available_seasons[-2]
        test_season = available_seasons[-1]
    else:
        train_season = test_season - 1
    
    print(f"   Training on {train_season}, testing on {test_season}")
    
    # Get test data
    test_data = data[data['Season'] == test_season]
    
    # Create baseline predictions for test data
    baseline_preds = []
    actual_values = []
    
    for _, row in test_data.iterrows():
        player_name = row['Name']
        
        # Get player's training data (previous season)
        train_data = data[(data['Name'] == player_name) & (data['Season'] == train_season)]
        
        if len(train_data) > 0:
            # Use 7-game average from training season
            if '7_game_avg' in train_data.columns:
                pred = train_data['7_game_avg'].iloc[-1]
            else:
                pred = train_data['FantPt'].mean()
            
            baseline_preds.append(pred)
            actual_values.append(row['FantPt'])
    
    if len(baseline_preds) == 0:
        raise ValueError("No predictions generated for evaluation")
    
    # Calculate metrics
    mae = mean_absolute_error(actual_values, baseline_preds)
    rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(baseline_preds)) ** 2))
    
    print(f"   Baseline MAE: {mae:.2f}")
    print(f"   Baseline RMSE: {rmse:.2f}")
    print(f"   Evaluated on {len(baseline_preds)} predictions")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'num_predictions': len(baseline_preds),
        'train_season': train_season,
        'test_season': test_season
    }

def run_baseline_model(data_directory='datasets', target_players=None):
    """Run the complete baseline model pipeline."""
    print("=" * 60)
    print("Baseline (Naive) Fantasy Football Model")
    print("=" * 60)
    
    try:
        # Load unified dataset
        data = load_unified_dataset(data_directory)
        print(f"✅ Loaded unified dataset: {data.shape}")
        
        # Evaluate baseline model
        evaluation_results = evaluate_baseline_model(data)
        
        # Create predictions for target players
        predictions = create_baseline_predictions(data, target_players)
        
        # Save results
        results = {
            'model_type': 'baseline_naive',
            'evaluation': evaluation_results,
            'predictions': predictions,
            'num_players': len(predictions),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to results directory using path constants
        from ffbayes.utils.path_constants import get_results_dir
        current_year = datetime.now().year
        results_dir = get_results_dir(current_year) / 'baseline_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        from ffbayes.utils.path_constants import get_results_dir
        results_file = get_results_dir(datetime.now().year) / 'baseline_model_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Baseline model failed: {e}")
        raise

def main():
    """Main function for baseline model."""
    try:
        results = run_baseline_model()
        print("\n🎉 Baseline model completed successfully!")
        print(f"MAE: {results['evaluation']['mae']:.2f}")
        print(f"Predictions: {results['num_players']} players")
        
    except Exception as e:
        print(f"\n💥 Baseline model failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
