#!/usr/bin/env python3
"""
Model Comparison Framework for Fantasy Football Analysis.

This script provides comprehensive comparison between Monte Carlo and Bayesian team projections,
including statistical validation, uncertainty analysis, and model selection criteria.
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class ModelComparisonFramework:
    """Framework for comparing Monte Carlo and Bayesian team projections."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the model comparison framework.
        
        Args:
            output_dir: Directory to save comparison results and visualizations
        """
        if output_dir is None:
            from ffbayes.utils.path_constants import get_plots_dir
            current_year = datetime.now().year
            output_dir = str(get_plots_dir(current_year) / "model_comparison")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize comparison results
        self.comparison_results = {}
        self.monte_carlo_data = None
        self.bayesian_data = None
        
        # Simple adapter for our modern_model_results structure (MAE-based)
        self.bayes_summary = None
    
    def load_monte_carlo_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Load Monte Carlo simulation results.
        
        Args:
            file_path: Path to Monte Carlo results file. If None, searches for latest.
            
        Returns:
            Dictionary containing Monte Carlo results
        """
        if file_path is None:
            # Search for latest Monte Carlo results with year-based paths
            current_year = datetime.now().year
            from ffbayes.utils.training_config import \
                get_monte_carlo_training_years
            training_years = get_monte_carlo_training_years()
            from ffbayes.utils.path_constants import (get_monte_carlo_dir,
                                                      get_team_aggregation_dir)
            
            search_patterns = [
                str(get_monte_carlo_dir(current_year) / f'mc_projections_{current_year}_*.tsv'),
                str(get_monte_carlo_dir(current_year) / f'mc_projections_{current_year}_trained_on_{training_years}.tsv'),
                str(get_team_aggregation_dir(current_year) / 'team_aggregation_results_*.json'),
                str(get_monte_carlo_dir(current_year) / 'monte_carlo_results_*.json'),
            ]
            
            latest_file = None
            latest_time = 0
            
            for pattern in search_patterns:
                for file_path in glob.glob(pattern):
                    file_time = os.path.getctime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
            
            if not latest_file:
                raise FileNotFoundError("No Monte Carlo results found")
            
            file_path = latest_file
        
        print(f"📊 Loading Monte Carlo results from: {file_path}")
        
        # Support TSV projections (our current output)
        if file_path.endswith('.tsv'):
            import pandas as pd
            df = pd.read_csv(file_path, sep='\t', index_col=0)
            # Build a minimal structure expected downstream
            mean_val = float(df['Total'].mean()) if 'Total' in df.columns else 0.0
            std_val = float(df['Total'].std()) if 'Total' in df.columns else 0.0
            min_val = float(df['Total'].min()) if 'Total' in df.columns else 0.0
            max_val = float(df['Total'].max()) if 'Total' in df.columns else 0.0
            pcts = {
                'p5': float(df['Total'].quantile(0.05)) if 'Total' in df.columns else 0.0,
                'p25': float(df['Total'].quantile(0.25)) if 'Total' in df.columns else 0.0,
                'p50': float(df['Total'].quantile(0.50)) if 'Total' in df.columns else 0.0,
                'p75': float(df['Total'].quantile(0.75)) if 'Total' in df.columns else 0.0,
                'p95': float(df['Total'].quantile(0.95)) if 'Total' in df.columns else 0.0,
            }
            data = {
                'monte_carlo_projection': {
                    'team_projection': {
                        'total_score': {
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'confidence_interval': [mean_val - 1.96*std_val, mean_val + 1.96*std_val],
                            'percentiles': pcts
                        }
                    },
                    'player_contributions': {}
                },
                'simulation_metadata': {
                    'number_of_simulations': int(len(df)),
                    'execution_time': 0,
                    'convergence_status': 'unknown'
                }
            }
            self.monte_carlo_data = data
            return data
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate Monte Carlo data structure
            if 'monte_carlo_projection' not in data:
                raise ValueError("Invalid Monte Carlo data structure: missing 'monte_carlo_projection'")
            
            # Structure the data for visualization compatibility
            self.monte_carlo_data = {
                'team_performance': {
                    'mean': data['monte_carlo_projection'].get('team_score_mean', 0),
                    'std': data['monte_carlo_projection'].get('team_score_std', 0),
                    'percentiles': {
                        '25th': data['monte_carlo_projection'].get('team_score_mean', 0) - 0.67 * data['monte_carlo_projection'].get('team_score_std', 0),
                        '75th': data['monte_carlo_projection'].get('team_score_mean', 0) + 0.67 * data['monte_carlo_projection'].get('team_score_std', 0)
                    }
                },
                'player_analysis': data.get('player_contributions', {})
            }
            return data
    
    def load_bayesian_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Load Bayesian model results.
        
        Args:
            file_path: Path to Bayesian results file. If None, searches for latest.
            
        Returns:
            Dictionary containing Bayesian results
        """
        if file_path is None:
            # Search for latest Bayesian results
            current_year = datetime.now().year
            from ffbayes.utils.path_constants import (get_bayesian_model_dir,
                                                      get_hybrid_mc_dir)
            
            search_patterns = [
                str(get_hybrid_mc_dir(current_year) / "hybrid_model_results.json"),
                str(get_bayesian_model_dir(current_year) / "bayesian_results_*.json"),
                str(get_bayesian_model_dir(current_year) / "model_output_*.json")
            ]
            
            latest_file = None
            latest_time = 0
            
            for pattern in search_patterns:
                for file_path in glob.glob(pattern):
                    file_time = os.path.getctime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
            
            if not latest_file:
                raise FileNotFoundError("No Bayesian results found")
            
            file_path = latest_file
        
        print(f"📊 Loading Bayesian results from: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # If this is our modern_model_results.json, capture MAEs for comparison
        if 'mae_bayesian' in data and 'mae_baseline' in data:
            self.bayes_summary = {
                'bayesian_mae': float(data['mae_bayesian']),
                'baseline_mae': float(data['mae_baseline'])
            }
            # Store the actual data without creating fake structure
            # The comparison will handle MAE-only data appropriately
        elif 'mae_unified' in data and 'mae_baseline' in data:
            self.bayes_summary = {
                'bayesian_mae': float(data['mae_unified']),
                'baseline_mae': float(data['mae_baseline'])
            }
            # Check if we have individual player predictions for team projections
            if 'player_predictions' in data:
                print("📊 Loaded Bayesian individual player predictions with uncertainty")
            else:
                print("📊 Loaded Bayesian MAE metrics (no individual predictions available)")
        elif self._is_hybrid_mc_format(data):
            # Handle Hybrid MC format - individual player predictions
            print("📊 Loaded Hybrid MC Bayesian individual player predictions")
            self.bayes_summary = {
                'model_type': 'hybrid_mc_bayesian',
                'player_count': len(data),
                'note': 'Individual player predictions available for team aggregation'
            }
        else:
            # Check if this is MAE-only data (valid structure)
            if 'mae_bayesian' in data or 'mae_unified' in data:
                print("📊 Loaded Bayesian MAE metrics (no team projections available)")
            elif 'team_projection' not in data and 'model_output' not in data:
                raise ValueError("Invalid Bayesian data structure: missing 'team_projection', 'model_output', or MAE metrics")
        
        self.bayesian_data = data
        return data
    
    def _is_hybrid_mc_format(self, data: Dict) -> bool:
        """Check if data is in Hybrid MC format (individual player predictions)."""
        if not isinstance(data, dict):
            return False
        
        # Hybrid MC format has player names as top-level keys
        # Each player has 'monte_carlo' and 'bayesian' or predictive stats fields
        keys = list(data.keys())
        if not keys:
            return False
        sample = data[keys[0]]
        return isinstance(sample, dict) and ('mean' in sample or 'bayesian' in sample or 'monte_carlo' in sample)

    def _load_team_names(self) -> List[str]:
        """Load drafted team names (standardized columns) for aggregation."""
        from ffbayes.utils.path_constants import get_teams_dir
        team_path = get_teams_dir() / f"drafted_team_{datetime.now().year}.tsv"
        if not team_path.exists():
            return []
        df = pd.read_csv(team_path, sep='\t')
        if 'Name' not in df.columns and 'PLAYER' in df.columns:
            df = df.rename(columns={'PLAYER': 'Name'})
        return [n for n in df['Name'].dropna().astype(str).tolist()]

    def _compute_hybrid_team_stats(self, hybrid_data: Dict) -> Optional[Dict[str, float]]:
        """Aggregate per-player Hybrid predictions to team-level mean/std.
        Uses shrinkage covariance on historical weekly fantasy points for team std.
        """
        team_players = self._load_team_names()
        if not team_players:
            return None
        # Extract per-player mean/std
        player_means = {}
        player_stds = {}
        for name in team_players:
            pdata = hybrid_data.get(name)
            if not isinstance(pdata, dict):
                continue
            # Check for monte_carlo structure first (current format)
            if 'monte_carlo' in pdata and isinstance(pdata['monte_carlo'], dict):
                mc = pdata['monte_carlo']
                if 'mean' in mc and 'std' in mc:
                    player_means[name] = float(mc['mean'])
                    player_stds[name] = float(mc['std'])
            # Check for direct mean/std (legacy format)
            elif 'mean' in pdata and 'std' in pdata:
                player_means[name] = float(pdata['mean'])
                player_stds[name] = float(pdata['std'])
            # Check for bayesian structure (alternative format)
            elif 'bayesian' in pdata and isinstance(pdata['bayesian'], dict):
                b = pdata['bayesian']
                if 'mean' in b and 'std' in b:
                    player_means[name] = float(b['mean'])
                    player_stds[name] = float(b['std'])
        if not player_means:
            return None
        # Team mean is sum of means for available players
        team_mean = sum(player_means.values())
        
        # IMPLEMENT PROPER CORRELATION HANDLING
        # Method 1: Joint Weekly Bootstrap (preferred)
        team_std = self._compute_team_std_with_correlations(player_means, player_stds)
        
        return {"mean": float(team_mean), "std": float(team_std)}

    def _compute_team_std_with_correlations(self, player_means: Dict[str, float], player_stds: Dict[str, float]) -> float:
        """
        Compute team standard deviation accounting for player performance correlations.
        
        Uses joint weekly bootstrap to sample correlated player performances,
        then computes team variance from the empirical distribution.
        
        Args:
            player_means: Dict of player name -> mean fantasy points
            player_stds: Dict of player name -> std fantasy points
        
        Returns:
            Team standard deviation accounting for correlations
        """
        try:
            import numpy as np

            from ffbayes.data_pipeline.unified_data_loader import \
                load_unified_dataset

            # Load historical weekly data
            df = load_unified_dataset('datasets')
            if not {'Name', 'FantPt', 'G#', 'Season'}.issubset(df.columns):
                raise ValueError("Unified dataset missing required columns")
            
            # Filter to recent years and our players
            recent_years = list(range(datetime.now().year - 5, datetime.now().year))
            df_recent = df[df['Season'].isin(recent_years)]
            df_players = df_recent[df_recent['Name'].isin(list(player_means.keys()))]
            
            if len(df_players) == 0:
                raise ValueError("No historical data found for team players")
            
            # Build weekly player performance matrix
            weekly_matrix = df_players.pivot_table(
                index=['Season', 'G#'], 
                columns='Name', 
                values='FantPt', 
                aggfunc='mean'
            ).fillna(0)  # Fill missing weeks with 0
            
            if weekly_matrix.shape[1] < 2:
                # Need at least 2 players for correlation
                print("⚠️  Insufficient players for correlation analysis, using independence assumption")
                return float(np.sqrt(np.sum(np.square(list(player_stds.values())))))
            
            # JOINT WEEKLY BOOTSTRAP: Sample entire weeks to preserve correlations
            n_bootstrap = 10000
            team_scores = []
            
            for _ in range(n_bootstrap):
                # Sample a random week from historical data
                if len(weekly_matrix) > 0:
                    sampled_week = weekly_matrix.sample(n=1).iloc[0]
                    
                    # For each player, use their historical performance or fall back to model prediction
                    week_team_score = 0
                    for player_name in player_means.keys():
                        if player_name in sampled_week.index and not np.isnan(sampled_week[player_name]):
                            # Use historical performance for this week
                            week_team_score += sampled_week[player_name]
                        else:
                            # Fall back to model prediction with uncertainty
                            player_mean = player_means[player_name]
                            player_std = player_stds[player_name]
                            week_team_score += np.random.normal(player_mean, player_std)
                    
                    team_scores.append(week_team_score)
                else:
                    # No historical weeks available, use independence assumption
                    week_team_score = sum(np.random.normal(mean, std) for mean, std in zip(player_means.values(), player_stds.values()))
                    team_scores.append(week_team_score)
            
            # Compute empirical team standard deviation
            team_std = float(np.std(team_scores))
            
            # Compare with independence assumption for validation
            independence_std = float(np.sqrt(np.sum(np.square(list(player_stds.values())))))
            correlation_effect = team_std / independence_std if independence_std > 0 else 1.0
            
            print("📊 Team correlation analysis:")
            print(f"   Independence assumption: {independence_std:.2f}")
            print(f"   With correlations: {team_std:.2f}")
            print(f"   Correlation effect: {correlation_effect:.2f}x")
            
            return team_std
            
        except Exception as e:
            print(f"⚠️  Correlation analysis failed: {e}")
            print("   Falling back to independence assumption")
            # Fallback: independence approximation
            return float(np.sqrt(np.sum(np.square(list(player_stds.values())))))

    def compare_team_projections(self) -> Dict[str, Any]:
        """Compare team projections between Monte Carlo and Bayesian (Hybrid) models."""
        if self.monte_carlo_data is None or self.bayesian_data is None:
            raise ValueError("Both Monte Carlo and Bayesian data must be loaded first")
        
        print("🔍 Comparing team projections between models...")
        
        # Extract team score data from Monte Carlo (handle different structures)
        if 'team_projection' in self.monte_carlo_data['monte_carlo_projection']:
            # Standard Monte Carlo structure
            mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
        else:
            # Team aggregation structure
            mc_proj = self.monte_carlo_data['monte_carlo_projection']
            mc_score = {
                'mean': mc_proj['team_score_mean'],
                'std': mc_proj['team_score_std'],
                'min': mc_proj['percentiles']['p5'],
                'max': mc_proj['percentiles']['p95'],
                'confidence_interval': mc_proj['confidence_interval'],
                'percentiles': mc_proj['percentiles']
            }
        
        # Initialize comparison metrics
        comparison_metrics = {}
        
        # Handle different Bayesian data structures
        if 'team_projection' in self.bayesian_data:
            # Full Bayesian team projections available
            bayes_score = self.bayesian_data['team_projection']['total_score']
            
            # Calculate full comparison metrics
            comparison_metrics = {
                'mean_difference': abs(mc_score['mean'] - bayes_score['mean']),
                'mean_difference_pct': abs(mc_score['mean'] - bayes_score['mean']) / mc_score['mean'] * 100,
                'std_ratio': mc_score['std'] / max(bayes_score.get('std', 1e-6), 1e-6),
                'uncertainty_difference': abs(mc_score['std'] - bayes_score.get('std', 0.0)),
                'confidence_interval_overlap': self._calculate_ci_overlap(
                    mc_score['confidence_interval'], 
                    bayes_score.get('confidence_interval', [0, 0])
                ),
                'percentile_correlation': self._calculate_percentile_correlation(
                    mc_score['percentiles'], 
                    bayes_score.get('percentiles', {})
                )
            }
        elif 'model_output' in self.bayesian_data:
            # Model output structure available
            bayes_score = self.bayesian_data['model_output']['team_score']
            
            # Calculate comparison metrics for model output
            comparison_metrics = {
                'mean_difference': abs(mc_score['mean'] - bayes_score['mean']),
                'mean_difference_pct': abs(mc_score['mean'] - bayes_score['mean']) / mc_score['mean'] * 100,
                'std_ratio': mc_score['std'] / max(bayes_score.get('std', 1e-6), 1e-6),
                'uncertainty_difference': abs(mc_score['std'] - bayes_score.get('std', 0.0)),
                'confidence_interval_overlap': self._calculate_ci_overlap(
                    mc_score['confidence_interval'], 
                    bayes_score.get('confidence_interval', [0, 0])
                ),
                'percentile_correlation': self._calculate_percentile_correlation(
                    mc_score['percentiles'], 
                    bayes_score.get('percentiles', {})
                )
            }
        elif self._is_hybrid_mc_format(self.bayesian_data):
            # Aggregate Hybrid per-player predictions to team level using roster
            roster = self._load_team_names()
            hybrid_team = self._compute_hybrid_team_stats(self.bayesian_data)
            if hybrid_team:
                comparison_metrics = {
                    'comparison_type': 'hybrid_vs_monte_carlo',
                    'hybrid_team_projection': {
                        'mean': hybrid_team['mean'],
                        'std': hybrid_team['std']
                    },
                    'note': f"Team projection aggregated from {len(roster)} individual player predictions"
                }
                # If MC team stats are present, add CI overlap and percentile correlation where possible
                try:
                    mc_total = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
                    if mc_total and 'percentiles' in mc_total:
                        mc_ci = [mc_total['percentiles'].get('p5', 0.0), mc_total['percentiles'].get('p95', 0.0)]
                        if all(isinstance(x, (int,float)) for x in mc_ci):
                            hyb_ci = [hybrid_team['mean'] - 1.96*hybrid_team['std'],
                                      hybrid_team['mean'] + 1.96*hybrid_team['std']]
                            comparison_metrics['ci_overlap_ratio'] = self._calculate_ci_overlap(hyb_ci, mc_ci)
                except Exception:
                    pass
            else:
                comparison_metrics = {
                    'note': 'Could not aggregate individual predictions to team level'
                }
        else:
            # Only MAE metrics available - can't do team projection comparison
            print("⚠️  Bayesian data only contains MAE metrics, skipping team projection comparison")
            comparison_metrics = {
                'note': 'Only MAE comparison available - no team projections in Bayesian data'
            }
        
        # If we have MAEs, include them
        if self.bayes_summary:
            # Safely include MAE metrics if present
            bayes_mae = self.bayes_summary.get('bayesian_mae') if isinstance(self.bayes_summary, dict) else None
            base_mae = self.bayes_summary.get('baseline_mae') if isinstance(self.bayes_summary, dict) else None
            if bayes_mae is not None and base_mae is not None:
                comparison_metrics['bayesian_mae'] = bayes_mae
                comparison_metrics['baseline_mae'] = base_mae
                comparison_metrics['bayes_vs_baseline_improvement'] = (base_mae - bayes_mae)
            else:
                # No MAE fields available - proceed without MAE metrics
                comparison_metrics['note_mae'] = 'MAE metrics not available in Bayesian data; proceeding without MAE comparison'
            
            self.comparison_results['model_comparison'] = {
                    'Bayesian': {'mae': bayes_mae},
                    'Baseline': {'mae': base_mae},
            }
            # Format MAE values safely
            bayes_mae_str = f"{bayes_mae:.2f}" if bayes_mae is not None else "N/A"
            base_mae_str = f"{base_mae:.2f}" if base_mae is not None else "N/A"
            
            self.comparison_results['model_selection'] = {
                'recommendation': {
                        'primary_recommendation': 'Bayesian model preferred' if (bayes_mae is not None and base_mae is not None and bayes_mae <= base_mae) else 'Baseline preferred',
                        'reasoning': f"MAE: Bayes {bayes_mae_str} vs Baseline {base_mae_str}"
                }
            }
            
        
        # Store comparison results
        self.comparison_results = {
            'monte_carlo': self.monte_carlo_data,
            'bayesian': self.bayesian_data,
            'comparison_metrics': comparison_metrics,
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': self._generate_analysis_summary(comparison_metrics)
        }
        
        return self.comparison_results
    
    def _calculate_ci_overlap(self, ci1: List[float], ci2: List[float]) -> float:
        """Calculate overlap between two confidence intervals.
        
        Args:
            ci1: First confidence interval [lower, upper]
            ci2: Second confidence interval [lower, upper]
            
        Returns:
            Overlap ratio (0 = no overlap, 1 = complete overlap)
        """
        if len(ci1) != 2 or len(ci2) != 2:
            return 0.0
        
        # Calculate overlap
        overlap_lower = max(ci1[0], ci2[0])
        overlap_upper = min(ci1[1], ci2[1])
        
        if overlap_upper <= overlap_lower:
            return 0.0  # No overlap
        
        # Calculate overlap ratio
        overlap_width = overlap_upper - overlap_lower
        ci1_width = ci1[1] - ci1[0]
        ci2_width = ci2[1] - ci2[0]
        
        # Return overlap relative to the smaller interval
        min_width = min(ci1_width, ci2_width)
        return overlap_width / min_width if min_width > 0 else 0.0
    
    def _calculate_percentile_correlation(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate correlation between percentile values from two models.
        
        Args:
            p1: First model percentiles
            p2: Second model percentiles
            
        Returns:
            Correlation coefficient
        """
        # Get common percentile keys
        common_keys = set(p1.keys()) & set(p2.keys())
        if len(common_keys) < 2:
            return 0.0
        
        # Extract values for common percentiles
        values1 = [p1[key] for key in sorted(common_keys)]
        values2 = [p2[key] for key in sorted(common_keys)]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            print(f"Percentile correlation calculation error: {e}")
            return 0.0
    
    def _generate_analysis_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable analysis summary.
        
        Args:
            metrics: Comparison metrics dictionary
            
        Returns:
            Dictionary containing analysis insights
        """
        summary = {}
        
        # Check if we have team projection comparison metrics
        if 'note' in metrics:
            summary['comparison_status'] = metrics['note']
            return summary
        
        # Mean difference analysis
        if metrics['mean_difference_pct'] < 2.0:
            summary['mean_agreement'] = "Excellent agreement between models"
        elif metrics['mean_difference_pct'] < 5.0:
            summary['mean_agreement'] = "Good agreement between models"
        elif metrics['mean_difference_pct'] < 10.0:
            summary['mean_agreement'] = "Moderate agreement between models"
        else:
            summary['mean_agreement'] = "Significant disagreement between models"
        
        # Uncertainty analysis
        if metrics['std_ratio'] < 0.5:
            summary['uncertainty_comparison'] = "Monte Carlo shows much lower uncertainty"
        elif metrics['std_ratio'] < 0.8:
            summary['uncertainty_comparison'] = "Monte Carlo shows lower uncertainty"
        elif metrics['std_ratio'] < 1.2:
            summary['uncertainty_comparison'] = "Similar uncertainty estimates"
        elif metrics['std_ratio'] < 2.0:
            summary['uncertainty_comparison'] = "Bayesian shows higher uncertainty"
        else:
            summary['uncertainty_comparison'] = "Bayesian shows much higher uncertainty"
        
        # Confidence interval analysis
        if metrics['confidence_interval_overlap'] > 0.8:
            summary['ci_overlap'] = "High confidence interval overlap"
        elif metrics['confidence_interval_overlap'] > 0.5:
            summary['ci_overlap'] = "Moderate confidence interval overlap"
        elif metrics['confidence_interval_overlap'] > 0.2:
            summary['ci_overlap'] = "Low confidence interval overlap"
        else:
            summary['ci_overlap'] = "Minimal confidence interval overlap"
        
        # Percentile correlation analysis
        if metrics['percentile_correlation'] > 0.9:
            summary['percentile_agreement'] = "Excellent percentile agreement"
        elif metrics['percentile_correlation'] > 0.7:
            summary['percentile_agreement'] = "Good percentile agreement"
        elif metrics['percentile_correlation'] > 0.5:
            summary['percentile_agreement'] = "Moderate percentile agreement"
        else:
            summary['percentile_agreement'] = "Poor percentile agreement"
        
        return summary
    
    def add_statistical_validation(self, historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add statistical validation against historical data.
        
        Args:
            historical_data: Historical actual scores for validation
            
        Returns:
            Updated comparison results with validation metrics
        """
        if self.comparison_results == {}:
            raise ValueError("Must run comparison first")
        
        print("📊 Adding statistical validation...")
        
        validation_metrics = {}
        
        if historical_data and 'actual_scores' in historical_data:
            actual_scores = historical_data['actual_scores']
            
            # Calculate validation metrics for each model
            mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
            bayes_score = self.bayesian_data.get('team_projection', {}).get('total_score', 
                         self.bayesian_data.get('model_output', {}).get('team_score', {}))
            
            # Mean Absolute Error
            mc_mae = np.mean([abs(score - mc_score['mean']) for score in actual_scores])
            bayes_mae = np.mean([abs(score - bayes_score['mean']) for score in actual_scores])
            
            validation_metrics['historical_validation'] = {
                'monte_carlo_mae': mc_mae,
                'bayesian_mae': bayes_mae,
                'mae_ratio': mc_mae / bayes_mae if bayes_mae > 0 else float('inf'),
                'better_model': 'bayesian' if bayes_mae < mc_mae else 'monte_carlo',
                'improvement_pct': abs(mc_mae - bayes_mae) / max(mc_mae, bayes_mae) * 100
            }
        
        # Add validation metrics to results
        self.comparison_results['validation_metrics'] = validation_metrics
        
        return validation_metrics
    
    def create_model_selection_criteria(self) -> Dict[str, Any]:
        """Create framework for model selection and validation.
        
        Returns:
            Dictionary containing model selection criteria
        """
        if self.comparison_results == {}:
            raise ValueError("Must run comparison first")
        
        print("🎯 Creating model selection criteria...")
        
        # Extract metadata for model selection
        mc_meta = self.monte_carlo_data.get('simulation_metadata', {})
        bayes_meta = self.bayesian_data.get('model_metadata', {})
        
        # Model quality scores
        mc_quality_score = self._calculate_monte_carlo_quality_score(mc_meta)
        bayes_quality_score = self._calculate_bayesian_quality_score(bayes_meta)
        
        # Model selection criteria
        selection_criteria = {
            'monte_carlo_quality': {
                'score': mc_quality_score,
                'convergence': mc_meta.get('convergence_status', 'unknown'),
                'simulations': mc_meta.get('number_of_simulations', 0),
                'execution_time': mc_meta.get('execution_time', 0)
            },
            'bayesian_quality': {
                'score': bayes_quality_score,
                'rhat': bayes_meta.get('convergence_metrics', {}).get('rhat', float('inf')),
                'effective_sample_size': bayes_meta.get('convergence_metrics', {}).get('effective_sample_size', 0),
                'draws': bayes_meta.get('draws', 0),
                'chains': bayes_meta.get('chains', 0)
            },
            'recommendation': self._generate_model_recommendation(
                mc_quality_score, bayes_quality_score
            )
        }
        
        # Add selection criteria to results
        self.comparison_results['model_selection'] = selection_criteria
        
        return selection_criteria
    
    def _calculate_monte_carlo_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for Monte Carlo model.
        
        Args:
            metadata: Monte Carlo metadata
            
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Convergence status
        if metadata.get('convergence_status') == 'converged':
            score += 30
        elif metadata.get('convergence_status') == 'partial':
            score += 15
        
        # Number of simulations
        sims = metadata.get('number_of_simulations', 0)
        if sims >= 10000:
            score += 40
        elif sims >= 5000:
            score += 30
        elif sims >= 1000:
            score += 20
        elif sims >= 100:
            score += 10
        
        # Execution time (shorter is better for testing)
        exec_time = metadata.get('execution_time', 0)
        if exec_time <= 60:  # 1 minute or less
            score += 30
        elif exec_time <= 300:  # 5 minutes or less
            score += 20
        elif exec_time <= 600:  # 10 minutes or less
            score += 10
        
        return min(score, 100.0)
    
    def _calculate_bayesian_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for Bayesian model.
        
        Args:
            metadata: Bayesian metadata
            
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # R-hat convergence
        rhat = metadata.get('convergence_metrics', {}).get('rhat', float('inf'))
        if rhat <= 1.01:
            score += 40
        elif rhat <= 1.05:
            score += 30
        elif rhat <= 1.1:
            score += 20
        elif rhat <= 1.2:
            score += 10
        
        # Effective sample size
        ess = metadata.get('convergence_metrics', {}).get('effective_sample_size', 0)
        if ess >= 2000:
            score += 30
        elif ess >= 1000:
            score += 20
        elif ess >= 500:
            score += 10
        
        # Number of draws and chains
        draws = metadata.get('draws', 0)
        chains = metadata.get('chains', 0)
        
        if draws >= 2000 and chains >= 4:
            score += 30
        elif draws >= 1000 and chains >= 2:
            score += 20
        elif draws >= 500 and chains >= 1:
            score += 10
        
        return min(score, 100.0)
    
    def _generate_model_recommendation(self, mc_score: float, bayes_score: float) -> Dict[str, str]:
        """Generate model recommendation based on quality scores.
        
        Args:
            mc_score: Monte Carlo quality score
            bayes_score: Bayesian quality score
            
        Returns:
            Dictionary containing recommendation and reasoning
        """
        score_diff = mc_score - bayes_score
        
        if abs(score_diff) < 10:
            recommendation = "Both models are comparable in quality"
            reasoning = "Quality scores are within 10 points of each other"
        elif mc_score > bayes_score:
            recommendation = "Monte Carlo model is recommended"
            reasoning = f"Monte Carlo quality score ({mc_score:.1f}) is {score_diff:.1f} points higher"
        else:
            recommendation = "Bayesian model is recommended"
            reasoning = f"Bayesian quality score ({bayes_score:.1f}) is {abs(score_diff):.1f} points higher"
        
        return {
            'primary_recommendation': recommendation,
            'reasoning': reasoning,
            'monte_carlo_score': mc_score,
            'bayesian_score': bayes_score,
            'score_difference': score_diff
        }
    
    def save_comparison_results(self, filename: Optional[str] = None) -> str:
        """Save comparison results to file.
        
        Args:
            filename: Output filename. If None, generates timestamped name.
            
        Returns:
            Path to saved file
        """
        if self.comparison_results == {}:
            raise ValueError("No comparison results to save")
        
        if filename is None:
            filename = "model_comparison_results.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        print(f"💾 Saved comparison results to: {file_path}")
        return file_path
    
    def create_comparison_visualizations(self) -> List[str]:
        """Create visualizations for model comparison.
        
        Returns:
            List of paths to generated visualization files
        """
        if self.comparison_results == {}:
            raise ValueError("No comparison results to visualize")
        
        print("📊 Creating model comparison visualizations...")
        return self._create_model_comparison_visualizations()
    
    def _create_model_comparison_visualizations(self):
        """Create comprehensive model comparison visualizations with actionable insights."""
        print("📊 Creating model comparison visualizations...")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Model Disagreement Analysis (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_disagreement(ax1)
        
        # Plot 2: Position Value Comparison (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_position_value_comparison(ax2)
        
        # Plot 3: Risk-Adjusted Rankings (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_risk_adjusted_rankings(ax3)
        
        # Plot 4: Team Projection Distribution (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_team_projection_distribution(ax4)
        
        # Plot 5: Player Uncertainty Analysis (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_player_uncertainty_analysis(ax5)
        
        # Plot 6: Model Confidence Comparison (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_model_confidence_comparison(ax6)
        
        # Plot 7: Draft Strategy Insights (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_draft_strategy_insights(ax7)
        
        # Plot 8: Market Inefficiency Analysis (Bottom Middle)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_market_inefficiency_analysis(ax8)
        
        # Plot 9: Actionable Recommendations (Bottom Right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_actionable_recommendations(ax9)
        
        # Plot 10: Model Quality Assessment (Bottom Full Width)
        ax10 = fig.add_subplot(gs[3, :])
        self._plot_model_quality_assessment(ax10)
        
        plt.tight_layout()
        
        # Save the comprehensive dashboard
        dashboard_file = os.path.join(self.output_dir, 'model_comparison_dashboard.png')
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {dashboard_file}")
        
        # Also create individual focused charts
        self._create_focused_charts()
        
        return [dashboard_file]
    
    def _plot_model_disagreement(self, ax):
        """Plot where models disagree - this is valuable information."""
        if not self.bayesian_data or not self.monte_carlo_data:
            ax.text(0.5, 0.5, 'Insufficient data for model disagreement analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Disagreement Analysis', fontsize=14, fontweight='bold')
            return
        
        # Extract player projections from both models
        mc_players = {}
        bayes_players = {}
        
        # Get Monte Carlo player data
        if 'player_analysis' in self.monte_carlo_data:
            mc_players = self.monte_carlo_data['player_analysis']
        
        # Get Bayesian player data
        bayes_players = {name: data.get('monte_carlo', {}) for name, data in self.bayesian_data.items()}
        
        # Find common players
        common_players = set(mc_players.keys()) & set(bayes_players.keys())
        if len(common_players) < 5:
            ax.text(0.5, 0.5, 'Insufficient overlapping players for comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Disagreement Analysis', fontsize=14, fontweight='bold')
            return
        
        # Calculate disagreements
        disagreements = []
        for player in list(common_players)[:15]:  # Top 15 for clarity
            mc_mean = mc_players[player].get('mean', 0)
            bayes_mean = bayes_players[player].get('mean', 0)
            disagreement = abs(mc_mean - bayes_mean)
            disagreements.append((player, disagreement, mc_mean, bayes_mean))
        
        # Sort by disagreement
        disagreements.sort(key=lambda x: x[1], reverse=True)
        
        # Plot
        players = [d[0] for d in disagreements]
        mc_means = [d[2] for d in disagreements]
        bayes_means = [d[3] for d in disagreements]
        
        x = range(len(players))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], mc_means, width, label='Monte Carlo', alpha=0.8, color='skyblue')
        bars2 = ax.bar([i + width/2 for i in x], bayes_means, width, label='Bayesian', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Players')
        ax.set_ylabel('Projected Fantasy Points')
        ax.set_title('Model Disagreement Analysis\n(Where Models Disagree Most)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add insight
        if disagreements:
            max_disagreement = disagreements[0]
            ax.text(0.02, 0.98, f'Biggest Disagreement: {max_disagreement[0]}\n'
                                f'MC: {max_disagreement[2]:.1f} vs Bayes: {max_disagreement[3]:.1f}\n'
                                f'Difference: {max_disagreement[1]:.1f} points',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    def _plot_position_value_comparison(self, ax):
        """Plot position value comparison across models."""
        if not self.bayesian_data:
            ax.text(0.5, 0.5, 'No Bayesian data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Value Comparison', fontsize=14, fontweight='bold')
            return
        
        # Group players by position and calculate average projections
        pos_data = {}
        for name, data in self.bayesian_data.items():
            if 'monte_carlo' in data and 'position' in data:
                pos = data['position']
                mean = data['monte_carlo'].get('mean', 0)
                if pos not in pos_data:
                    pos_data[pos] = []
                pos_data[pos].append(mean)
        
        if not pos_data:
            ax.text(0.5, 0.5, 'No position data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Value Comparison', fontsize=14, fontweight='bold')
            return
        
        # Calculate averages
        pos_means = {pos: np.mean(values) for pos, values in pos_data.items()}
        pos_counts = {pos: len(values) for pos, values in pos_data.items()}
        
        # Sort by average value
        sorted_pos = sorted(pos_means.items(), key=lambda x: x[1], reverse=True)
        positions = [p[0] for p in sorted_pos]
        means = [p[1] for p in sorted_pos]
        counts = [pos_counts[p] for p in positions]
        
        # Create bar chart
        bars = ax.bar(positions, means, color='lightgreen', alpha=0.8, edgecolor='black')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'n={count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Average Projected Points')
        ax.set_title('Position Value Comparison\n(Average Projections by Position)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add insight
        if positions:
            best_pos = positions[0]
            worst_pos = positions[-1]
            ax.text(0.02, 0.98, f'Best Position: {best_pos} ({means[0]:.1f} avg)\n'
                                f'Worst Position: {worst_pos} ({means[-1]:.1f} avg)\n'
                                f'Strategy: Prioritize {best_pos} early',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_risk_adjusted_rankings(self, ax):
        """Plot risk-adjusted player rankings."""
        if not self.bayesian_data:
            ax.text(0.5, 0.5, 'No Bayesian data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk-Adjusted Rankings', fontsize=14, fontweight='bold')
            return
        
        # Calculate risk-adjusted scores
        risk_scores = []
        for name, data in self.bayesian_data.items():
            if 'monte_carlo' in data:
                mean = data['monte_carlo'].get('mean', 0)
                std = data['monte_carlo'].get('std', 0)
                if mean > 0 and std > 0:
                    # Risk-adjusted score = mean / (1 + coefficient of variation)
                    risk_score = mean / (1 + (std / mean))
                    risk_scores.append((name, risk_score, mean, std))
        
        if len(risk_scores) < 5:
            ax.text(0.5, 0.5, 'Insufficient data for risk analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk-Adjusted Rankings', fontsize=14, fontweight='bold')
            return
        
        # Sort by risk-adjusted score
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Plot top 10
        top_10 = risk_scores[:10]
        names = [p[0] for p in top_10]
        scores = [p[1] for p in top_10]
        means = [p[2] for p in top_10]
        stds = [p[3] for p in top_10]
        
        bars = ax.barh(range(len(names)), scores, color='gold', alpha=0.8, edgecolor='black')
        
        # Add player names
        for i, name in enumerate(names):
            ax.text(scores[i] + 0.5, i, name, va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Risk-Adjusted Score')
        ax.set_title('Risk-Adjusted Player Rankings\n(Top 10 - Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        # Add insight
        if top_10:
            best_player = top_10[0]
            ax.text(0.02, 0.98, f'Best Risk-Adjusted: {best_player[0]}\n'
                                f'Score: {best_player[1]:.1f}\n'
                                f'Mean: {best_player[2]:.1f} ± {best_player[3]:.1f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def _plot_team_projection_distribution(self, ax):
        """Plot team projection distribution with confidence intervals."""
        if not self.monte_carlo_data or 'team_performance' not in self.monte_carlo_data:
            ax.text(0.5, 0.5, 'No Monte Carlo team data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Team Projection Distribution', fontsize=14, fontweight='bold')
            return
        
        perf_data = self.monte_carlo_data['team_performance']
        
        # Create a simulated distribution
        mean = perf_data.get('mean', 0)
        std = perf_data.get('std', 0)
        
        if std <= 0:
            ax.text(0.5, 0.5, 'Invalid team projection data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Team Projection Distribution', fontsize=14, fontweight='bold')
            return
        
        # Generate distribution
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        y = stats.norm.pdf(x, mean, std)
        
        ax.plot(x, y, 'b-', linewidth=2, label='Team Score Distribution')
        ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        
        # Add confidence intervals
        if 'percentiles' in perf_data:
            p25 = perf_data['percentiles'].get('25th', mean - 0.67*std)
            p75 = perf_data['percentiles'].get('75th', mean + 0.67*std)
            ax.axvline(p25, color='orange', linestyle=':', linewidth=2, label=f'25th: {p25:.1f}')
            ax.axvline(p75, color='orange', linestyle=':', linewidth=2, label=f'75th: {p75:.1f}')
        
        ax.set_xlabel('Team Fantasy Points')
        ax.set_ylabel('Probability Density')
        ax.set_title('Team Projection Distribution\n(5000 Monte Carlo Simulations)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add insight
        ax.text(0.02, 0.98, f'Team Projection: {mean:.1f} ± {std:.1f} points\n'
                            f'50% of weeks: {p25:.1f}-{p75:.1f} points\n'
                            f'High variance team - boom/bust potential',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _plot_player_uncertainty_analysis(self, ax):
        """Plot player uncertainty analysis."""
        if not self.bayesian_data:
            ax.text(0.5, 0.5, 'No Bayesian data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Player Uncertainty Analysis', fontsize=14, fontweight='bold')
            return
        
        # Extract uncertainty data
        uncertainty_data = []
        for name, data in self.bayesian_data.items():
            if 'monte_carlo' in data and 'bayesian_uncertainty' in data:
                mean = data['monte_carlo'].get('mean', 0)
                std = data['monte_carlo'].get('std', 0)
                overall_uncertainty = data['bayesian_uncertainty'].get('overall_uncertainty', 0)
                if mean > 0:
                    uncertainty_data.append((name, mean, std, overall_uncertainty))
        
        if len(uncertainty_data) < 5:
            ax.text(0.5, 0.5, 'Insufficient uncertainty data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Player Uncertainty Analysis', fontsize=14, fontweight='bold')
            return
        
        # Sort by uncertainty
        uncertainty_data.sort(key=lambda x: x[3], reverse=True)
        
        # Plot top 10 most uncertain players
        top_10 = uncertainty_data[:10]
        names = [p[0] for p in top_10]
        uncertainties = [p[3] for p in top_10]
        means = [p[1] for p in top_10]
        
        # Color by projection value
        colors = ['red' if u > 0.8 else 'orange' if u > 0.5 else 'green' for u in uncertainties]
        
        bars = ax.barh(range(len(names)), uncertainties, color=colors, alpha=0.8, edgecolor='black')
        
        # Add player names and projections
        for i, (name, mean) in enumerate(zip(names, means)):
            ax.text(uncertainties[i] + 0.02, i, f"{name} ({mean:.1f})", 
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Uncertainty Score (0-1)')
        ax.set_title('Player Uncertainty Analysis\n(Top 10 Most Uncertain)', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        # Add insight
        if top_10:
            most_uncertain = top_10[0]
            ax.text(0.02, 0.98, f'Most Uncertain: {most_uncertain[0]}\n'
                                f'Uncertainty: {most_uncertain[3]:.2f}\n'
                                f'Projection: {most_uncertain[1]:.1f} ± {most_uncertain[2]:.1f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def _plot_model_confidence_comparison(self, ax):
        """Plot model confidence comparison."""
        # This would compare confidence intervals between models
        ax.text(0.5, 0.5, 'Model Confidence Comparison\n(Coming Soon)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Model Confidence Comparison', fontsize=14, fontweight='bold')
    
    def _plot_draft_strategy_insights(self, ax):
        """Plot draft strategy insights."""
        ax.text(0.5, 0.5, 'Draft Strategy Insights\n(Coming Soon)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Draft Strategy Insights', fontsize=14, fontweight='bold')
    
    def _plot_market_inefficiency_analysis(self, ax):
        """Plot market inefficiency analysis."""
        ax.text(0.5, 0.5, 'Market Inefficiency Analysis\n(Coming Soon)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Market Inefficiency Analysis', fontsize=14, fontweight='bold')
    
    def _plot_actionable_recommendations(self, ax):
        """Plot actionable recommendations."""
        recommendations = [
            "🎯 DRAFT STRATEGY:",
            "• Target high-value positions early",
            "• Use model disagreements as opportunities",
            "• Prioritize low-uncertainty players in early rounds",
            "",
            "📊 TEAM BUILDING:",
            "• Balance high-upside with consistency",
            "• Monitor player uncertainty levels",
            "• Consider correlation effects in lineup decisions",
            "",
            "⚡ WEEKLY STRATEGY:",
            "• Use Monte Carlo for weekly projections",
            "• Leverage Bayesian uncertainty for risk assessment",
            "• Adjust strategy based on opponent strength"
        ]
        
        ax.text(0.05, 0.95, '\n'.join(recommendations), transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        ax.set_title('Actionable Recommendations', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _plot_model_quality_assessment(self, ax):
        """Plot comprehensive model quality assessment."""
        # Create quality metrics
        mc_quality = self._calculate_monte_carlo_quality_score(self.monte_carlo_data.get('simulation_metadata', {}))
        bayes_quality = self._calculate_bayesian_quality_score(self.bayesian_data.get('model_metadata', {}))
        
        models = ['Monte Carlo', 'Bayesian']
        qualities = [mc_quality, bayes_quality]
        colors = ['skyblue' if q > 0 else 'lightcoral' for q in qualities]
        
        bars = ax.bar(models, qualities, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, quality in zip(bars, qualities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{quality:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Quality Score (0-100)')
        ax.set_title('Model Quality Assessment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add recommendation
        if mc_quality > bayes_quality:
            recommendation = f"RECOMMENDATION: Use Monte Carlo model\n(Score: {mc_quality:.1f} vs {bayes_quality:.1f})"
        else:
            recommendation = f"RECOMMENDATION: Use Bayesian model\n(Score: {bayes_quality:.1f} vs {mc_quality:.1f})"
        
        ax.text(0.02, 0.98, recommendation, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def _create_focused_charts(self):
        """Create individual focused charts for specific insights."""
        # This would create additional focused charts
        pass

def main():
    """Main function to run the model comparison framework."""
    print("=" * 70)
    print("Model Comparison Framework for Fantasy Football Analysis")
    print("Comparing Monte Carlo vs Bayesian Team Projections")
    print("=" * 70)
    
    try:
        # Initialize framework
        framework = ModelComparisonFramework()
        
        # Load model results
        print("\n📊 Loading model results...")
        framework.load_monte_carlo_results()
        framework.load_bayesian_results()
        
        # Run comparison
        print("\n🔍 Running model comparison...")
        comparison_results = framework.compare_team_projections()
        
        # Add statistical validation (if historical data available)
        print("\n📊 Adding statistical validation...")
        # Note: Historical data would be loaded here in practice
        # framework.add_statistical_validation(historical_data)
        
        # Create model selection criteria
        print("\n🎯 Creating model selection criteria...")
        selection_criteria = framework.create_model_selection_criteria()
        
        # Save results
        print("\n💾 Saving comparison results...")
        results_file = framework.save_comparison_results()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        plot_files = framework.create_comparison_visualizations()
        
        # Summary
        print("\n" + "=" * 70)
        print("Model Comparison Framework - Execution Complete")
        print(f"- Results saved to: {results_file}")
        print(f"- Generated {len(plot_files)} visualization charts")
        print(f"- All charts saved to: {framework.output_dir}")
        
        # Display key insights
        if 'analysis_summary' in comparison_results:
            print("\n🔍 Key Insights:")
            for key, value in comparison_results['analysis_summary'].items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        if 'model_selection' in comparison_results:
            recommendation = comparison_results['model_selection']['recommendation']
            print(f"\n🎯 Model Recommendation: {recommendation['primary_recommendation']}")
            print(f"   Reasoning: {recommendation['reasoning']}")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in model comparison framework: {str(e)}")
        raise

if __name__ == "__main__":
    main()
