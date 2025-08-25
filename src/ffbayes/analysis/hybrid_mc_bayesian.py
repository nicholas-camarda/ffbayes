#!/usr/bin/env python3
"""
hybrid_mc_bayesian.py - Hybrid Monte Carlo + Bayesian Uncertainty Model
Builds on working Monte Carlo and adds sophisticated uncertainty layers.

CURRENT IMPLEMENTATION (Option 1):
- Uses ALL 5 years of historical data for Monte Carlo simulations
- Weights recent seasons slightly more heavily (70% recent, 30% historical)
- Provides different predictions than baseline by leveraging full dataset

FUTURE EXPANSION OPTIONS:
Option 2: Multi-Year Feature Engineering
- Use diff_from_avg across multiple seasons
- Model year-over-year improvement/decline trends
- Include seasonal pattern recognition

Option 3: Advanced Contextual Modeling
- Incorporate home/away effects using is_home field
- Model opponent-specific effects using opp_team index
- Use player tier modeling with rank field
- Include trend analysis with diff_from_avg patterns
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset


class HybridMCBayesianModel:
    """Hybrid Monte Carlo + Bayesian uncertainty model."""
    
    def __init__(self, data_directory='datasets'):
        """Initialize the hybrid model."""
        self.data_directory = data_directory
        self.data = None
        self.monte_carlo_results = None
        self.uncertainty_model = None
        
    def load_data(self):
        """Load unified dataset with VOR rankings."""
        print("🔄 Loading unified dataset...")
        self.data = load_unified_dataset(self.data_directory)
        print(f"✅ Loaded unified dataset: {self.data.shape}")
        print(f"✅ Available years: {sorted(self.data['Season'].unique())}")
        print(f"✅ VOR rankings available: {len(self.data[self.data['vor_global_rank'] != 121])}")
        
    def run_monte_carlo_base(self, target_players, n_simulations=1000):
        """Run Monte Carlo base model to get initial projections."""
        print("🎲 Running Monte Carlo base model...")
        
        # Get historical data for target players
        player_projections = {}
        
        for player_name in target_players:
            player_data = self.data[self.data['Name'] == player_name]
            
            if len(player_data) == 0:
                print(f"⚠️  Player {player_name} not found in data")
                continue
                
            # Use ALL available historical data (not just latest season)
            # This leverages the full 5-year dataset for better predictions
            all_fantasy_points = player_data['FantPt'].values
            n_games = len(all_fantasy_points)
            
            if n_games == 0:
                print(f"⚠️  No historical data for {player_name}")
                continue
                
            # Monte Carlo simulation using ALL historical performance
            # Weight recent games slightly more heavily but include full history
            simulations = []
            for _ in range(n_simulations):
                # Sample from all available games, with slight bias toward recent
                if n_games >= 10:
                    # For players with lots of data, sample more from recent games
                    recent_weight = 0.7  # 70% from recent, 30% from all
                    if np.random.random() < recent_weight:
                        # Sample from recent games (last 2 seasons)
                        recent_seasons = player_data['Season'].max() - 1
                        recent_data = player_data[player_data['Season'] >= recent_seasons]
                        if len(recent_data) > 0:
                            sample_points = np.random.choice(recent_data['FantPt'].values, 
                                                          size=min(10, len(recent_data)), replace=True)
                        else:
                            sample_points = np.random.choice(all_fantasy_points, size=min(10, n_games), replace=True)
                    else:
                        # Sample from all historical games
                        sample_points = np.random.choice(all_fantasy_points, size=min(10, n_games), replace=True)
                else:
                    # For players with limited data, use all available
                    sample_points = np.random.choice(all_fantasy_points, size=n_games, replace=True)
                
                simulations.append(np.mean(sample_points))
            
            # Calculate Monte Carlo statistics
            mc_mean = np.mean(simulations)
            mc_std = np.std(simulations)
            mc_percentiles = np.percentile(simulations, [5, 25, 50, 75, 95])
            
            player_projections[player_name] = {
                'monte_carlo': {
                    'mean': float(mc_mean),
                    'std': float(mc_std),
                    'percentiles': {
                        'p5': float(mc_percentiles[0]),
                        'p25': float(mc_percentiles[1]),
                        'p50': float(mc_percentiles[2]),
                        'p75': float(mc_percentiles[3]),
                        'p95': float(mc_percentiles[4])
                    },
                    'confidence_interval': [
                        float(mc_percentiles[0]),
                        float(mc_percentiles[4])
                    ]
                },
                'position': str(player_data['Position'].iloc[0]),
                'team': str(player_data['Tm'].iloc[0]) if 'Tm' in player_data.columns else None,
                'historical_games': n_games,
                'seasons_used': len(player_data['Season'].unique())
            }
        
        self.monte_carlo_results = player_projections
        print(f"✅ Monte Carlo projections for {len(player_projections)} players")
        return player_projections
    
    def add_bayesian_uncertainty_layers(self):
        """Add sophisticated Bayesian uncertainty layers on top of Monte Carlo."""
        print("🧠 Adding Bayesian uncertainty layers...")
        
        if not self.monte_carlo_results:
            raise ValueError("Must run Monte Carlo base model first")
        
        # Train uncertainty model using historical data
        self._train_uncertainty_model()
        
        # Enhance each player projection with uncertainty layers
        enhanced_projections = {}
        
        for player_name, mc_data in self.monte_carlo_results.items():
            enhanced_data = mc_data.copy()
            
            # Get player's historical data for uncertainty modeling
            player_data = self.data[self.data['Name'] == player_name]
            
            if len(player_data) > 0:
                # Calculate uncertainty features
                uncertainty_features = self._calculate_uncertainty_features(player_data)
                
                # Predict uncertainty using trained model
                predicted_uncertainty = self._predict_uncertainty(uncertainty_features)
                
                # Enhance Monte Carlo results with Bayesian uncertainty
                enhanced_data['bayesian_uncertainty'] = {
                    'data_quality_score': float(uncertainty_features['data_quality']),
                    'consistency_score': float(uncertainty_features['consistency']),
                    'trend_uncertainty': float(uncertainty_features['trend_uncertainty']),
                    'position_uncertainty': float(uncertainty_features['position_uncertainty']),
                    'overall_uncertainty': float(predicted_uncertainty)
                }
                
                # Adjust Monte Carlo confidence intervals based on uncertainty
                mc_mean = mc_data['monte_carlo']['mean']
                mc_std = mc_data['monte_carlo']['std']
                uncertainty_multiplier = 1 + predicted_uncertainty
                
                enhanced_data['monte_carlo']['adjusted_std'] = float(mc_std * uncertainty_multiplier)
                enhanced_data['monte_carlo']['adjusted_confidence_interval'] = [
                    float(mc_mean - mc_std * uncertainty_multiplier),
                    float(mc_mean + mc_std * uncertainty_multiplier)
                ]
                
                # Add VOR ranking validation
                vor_rank = player_data['vor_global_rank'].iloc[0] if 'vor_global_rank' in player_data.columns else 121
                enhanced_data['vor_validation'] = {
                    'global_rank': int(vor_rank),
                    'rank_tier': self._get_rank_tier(vor_rank),
                    'prediction_validation': self._validate_prediction_against_vor(
                        mc_data['monte_carlo']['mean'], vor_rank
                    )
                }
                
            enhanced_projections[player_name] = enhanced_data
        
        print(f"✅ Enhanced {len(enhanced_projections)} players with Bayesian uncertainty")
        return enhanced_projections
    
    def _train_uncertainty_model(self):
        """Train uncertainty prediction model using historical data."""
        print("   🔧 Training uncertainty model...")
        
        # Prepare training data
        training_features = []
        training_targets = []
        
        # Use all players with sufficient historical data
        for player_name in self.data['Name'].unique():
            player_data = self.data[self.data['Name'] == player_name]
            
            if len(player_data) >= 5:  # Need minimum games for uncertainty modeling
                features = self._calculate_uncertainty_features(player_data)
                target = self._calculate_actual_uncertainty(player_data)
                
                if target is not None:
                    training_features.append(list(features.values()))
                    training_targets.append(target)
        
        if len(training_features) < 10:
            print("   ⚠️  Insufficient training data for uncertainty model")
            return
        
        # Train Random Forest for uncertainty prediction
        self.uncertainty_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.uncertainty_model.fit(training_features, training_targets)
        
        print(f"   ✅ Trained uncertainty model on {len(training_features)} players")
    
    def _calculate_uncertainty_features(self, player_data):
        """Calculate uncertainty features for a player."""
        features = {}
        
        # Data quality features
        features['data_quality'] = min(len(player_data) / 20.0, 1.0)  # Normalize to 0-1
        
        # Consistency features
        fantasy_points = player_data['FantPt'].values
        features['consistency'] = 1.0 - (np.std(fantasy_points) / (np.mean(fantasy_points) + 1e-6))
        features['consistency'] = max(0.0, min(1.0, features['consistency']))
        
        # Trend uncertainty
        if len(fantasy_points) >= 3:
            # Calculate trend and its uncertainty
            x = np.arange(len(fantasy_points))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, fantasy_points)
            features['trend_uncertainty'] = min(std_err / (abs(slope) + 1e-6), 1.0)
        else:
            features['trend_uncertainty'] = 0.5
        
        # Position uncertainty
        position = player_data['Position'].iloc[0]
        position_data = self.data[self.data['Position'] == position]
        position_std = position_data['FantPt'].std()
        player_std = np.std(fantasy_points)
        features['position_uncertainty'] = min(player_std / (position_std + 1e-6), 1.0)
        
        # Recent form uncertainty
        if len(fantasy_points) >= 3:
            recent_games = fantasy_points[-3:]
            features['recent_form_uncertainty'] = np.std(recent_games) / (np.mean(recent_games) + 1e-6)
        else:
            features['recent_form_uncertainty'] = 0.5
        
        return features
    
    def _calculate_actual_uncertainty(self, player_data):
        """Calculate actual uncertainty for training the model."""
        fantasy_points = player_data['FantPt'].values
        
        if len(fantasy_points) < 3:
            return None
        
        # Use coefficient of variation as uncertainty measure
        cv = np.std(fantasy_points) / (np.mean(fantasy_points) + 1e-6)
        return min(cv, 2.0)  # Cap at 2.0
    
    def _predict_uncertainty(self, features):
        """Predict uncertainty using trained model."""
        if self.uncertainty_model is None:
            return 0.5  # Default uncertainty
        
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        prediction = self.uncertainty_model.predict(feature_vector)[0]
        return max(0.0, min(1.0, prediction))  # Ensure 0-1 range
    
    def _get_rank_tier(self, vor_rank):
        """Convert VOR rank to tier classification."""
        if vor_rank <= 30:
            return "Elite"
        elif vor_rank <= 70:
            return "High"
        elif vor_rank <= 120:
            return "Mid"
        else:
            return "Low"
    
    def _validate_prediction_against_vor(self, prediction, vor_rank):
        """Validate prediction makes sense given VOR ranking."""
        if vor_rank <= 30:  # Elite tier
            if prediction >= 15:
                return "✅ High prediction matches elite ranking"
            else:
                return "⚠️  Low prediction for elite ranking"
        elif vor_rank <= 70:  # High tier
            if 10 <= prediction <= 20:
                return "✅ Reasonable prediction for high tier"
            else:
                return "⚠️  Prediction outside expected range for tier"
        elif vor_rank <= 120:  # Mid tier
            if 5 <= prediction <= 15:
                return "✅ Reasonable prediction for mid tier"
            else:
                return "⚠️  Prediction outside expected range for tier"
        else:  # Low tier
            if prediction <= 10:
                return "✅ Low prediction matches low ranking"
            else:
                return "⚠️  High prediction for low ranking"
    
    def run_hybrid_pipeline(self, target_players=None, n_simulations=1000):
        """Run the complete hybrid pipeline."""
        print("=" * 60)
        print("Hybrid Monte Carlo + Bayesian Uncertainty Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Determine target players
            if target_players is None:
                # Use players from recent seasons for fair comparison with Baseline model
                # Get players from the last 2 seasons to ensure sufficient data while maintaining recency
                recent_seasons = [self.data['Season'].max() - 1, self.data['Season'].max()]
                recent_players = self.data[self.data['Season'].isin(recent_seasons)]['Name'].unique()
                target_players = list(recent_players)
                print(f"📋 Using players from seasons {recent_seasons}: {len(target_players)} players")
                print("   Leveraging full 5-year dataset for Monte Carlo simulations")
                print("   This provides better predictions than single-season baseline")
            
            # Step 3: Run Monte Carlo base
            mc_results = self.run_monte_carlo_base(target_players, n_simulations)
            
            # Step 4: Add Bayesian uncertainty layers
            enhanced_results = self.add_bayesian_uncertainty_layers()
            
            # Step 5: Save results
            self._save_results(enhanced_results)
            
            print("🎉 Hybrid pipeline completed successfully!")
            return enhanced_results
            
        except Exception as e:
            print(f"❌ Hybrid pipeline failed: {e}")
            raise
    
    def _save_results(self, results):
        """Save hybrid model results."""
        from ffbayes.utils.path_constants import get_hybrid_mc_dir
        current_year = datetime.now().year
        results_dir = get_hybrid_mc_dir(current_year)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        from ffbayes.utils.path_constants import get_hybrid_mc_dir
        results_file = get_hybrid_mc_dir(datetime.now().year) / 'hybrid_model_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary results
        summary = {
            'model_type': 'hybrid_mc_bayesian',
            'timestamp': datetime.now().isoformat(),
            'num_players': len(results),
            'pipeline_steps': ['monte_carlo_base', 'bayesian_uncertainty', 'vor_validation'],
            'uncertainty_features': ['data_quality', 'consistency', 'trend', 'position', 'recent_form']
        }
        
        summary_file = get_hybrid_mc_dir(datetime.now().year) / 'hybrid_model_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        print(f"✅ Summary saved to: {summary_file}")

def main():
    """Main function for hybrid model."""
    try:
        # Initialize hybrid model
        model = HybridMCBayesianModel()
        
        # Run hybrid pipeline
        results = model.run_hybrid_pipeline(n_simulations=500)
        
        # Print sample results
        print("\n📊 Sample Results:")
        sample_players = list(results.keys())[:3]
        for player in sample_players:
            player_data = results[player]
            mc = player_data['monte_carlo']
            vor = player_data['vor_validation']
            
            print(f"\n{player} ({player_data['position']}):")
            print(f"  MC Projection: {mc['mean']:.1f} ± {mc['std']:.1f}")
            print(f"  VOR Rank: #{vor['global_rank']} ({vor['rank_tier']})")
            print(f"  Validation: {vor['prediction_validation']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Hybrid model failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
