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
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class ModelComparisonFramework:
    """Framework for comparing Monte Carlo and Bayesian team projections."""
    
    def __init__(self, output_dir: str = "results/model_comparison"):
        """Initialize the model comparison framework.
        
        Args:
            output_dir: Directory to save comparison results and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize comparison results
        self.comparison_results = {}
        self.monte_carlo_data = None
        self.bayesian_data = None
        
    def load_monte_carlo_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Load Monte Carlo simulation results.
        
        Args:
            file_path: Path to Monte Carlo results file. If None, searches for latest.
            
        Returns:
            Dictionary containing Monte Carlo results
        """
        if file_path is None:
            # Search for latest Monte Carlo results
            search_patterns = [
                "results/team_aggregation/team_aggregation_results_*.json",
                "results/monte_carlo/monte_carlo_results_*.json"
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
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate Monte Carlo data structure
        if 'monte_carlo_projection' not in data:
            raise ValueError("Invalid Monte Carlo data structure: missing 'monte_carlo_projection'")
        
        self.monte_carlo_data = data
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
            search_patterns = [
                "results/bayesian_model/bayesian_results_*.json",
                "results/bayesian_model/model_output_*.json"
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
        
        # Validate Bayesian data structure
        if 'team_projection' not in data and 'model_output' not in data:
            raise ValueError("Invalid Bayesian data structure: missing 'team_projection' or 'model_output'")
        
        self.bayesian_data = data
        return data
    
    def compare_team_projections(self) -> Dict[str, Any]:
        """Compare team projections between Monte Carlo and Bayesian models.
        
        Returns:
            Dictionary containing comparison metrics and analysis
        """
        if self.monte_carlo_data is None or self.bayesian_data is None:
            raise ValueError("Both Monte Carlo and Bayesian data must be loaded first")
        
        print("🔍 Comparing team projections between models...")
        
        # Extract team score data
        mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
        
        # Handle different Bayesian data structures
        if 'team_projection' in self.bayesian_data:
            bayes_score = self.bayesian_data['team_projection']['total_score']
        else:
            bayes_score = self.bayesian_data['model_output']['team_score']
        
        # Calculate comparison metrics
        comparison_metrics = {
            'mean_difference': abs(mc_score['mean'] - bayes_score['mean']),
            'mean_difference_pct': abs(mc_score['mean'] - bayes_score['mean']) / mc_score['mean'] * 100,
            'std_ratio': mc_score['std'] / bayes_score['std'],
            'uncertainty_difference': abs(mc_score['std'] - bayes_score['std']),
            'confidence_interval_overlap': self._calculate_ci_overlap(
                mc_score['confidence_interval'], 
                bayes_score['confidence_interval']
            ),
            'percentile_correlation': self._calculate_percentile_correlation(
                mc_score['percentiles'], 
                bayes_score['percentiles']
            )
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
    
    def _generate_analysis_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable analysis summary.
        
        Args:
            metrics: Comparison metrics dictionary
            
        Returns:
            Dictionary containing analysis insights
        """
        summary = {}
        
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_results_{timestamp}.json"
        
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
        
        plot_files = []
        
        # 1. Team Score Comparison
        plot1 = self._create_team_score_comparison_chart()
        if plot1:
            plot_files.append(plot1)
        
        # 2. Uncertainty Comparison
        plot2 = self._create_uncertainty_comparison_chart()
        if plot2:
            plot_files.append(plot2)
        
        # 3. Model Quality Comparison
        plot3 = self._create_model_quality_chart()
        if plot3:
            plot_files.append(plot3)
        
        # 4. Player Contribution Comparison
        plot4 = self._create_player_contribution_chart()
        if plot4:
            plot_files.append(plot4)
        
        return plot_files
    
    def _create_team_score_comparison_chart(self) -> Optional[str]:
        """Create team score comparison chart."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Extract data
            mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
            bayes_score = self.bayesian_data.get('team_projection', {}).get('total_score', 
                         self.bayesian_data.get('model_output', {}).get('team_score', {}))
            
            # Chart 1: Score Distribution Comparison
            x_range = np.linspace(min(mc_score['min'], bayes_score['min']) - 2, 
                                max(mc_score['max'], bayes_score['max']) + 2, 100)
            
            # Monte Carlo distribution
            mc_pdf = stats.norm.pdf(x_range, mc_score['mean'], mc_score['std'])
            ax1.plot(x_range, mc_pdf, 'b-', linewidth=2, label='Monte Carlo', alpha=0.7)
            
            # Bayesian distribution
            bayes_pdf = stats.norm.pdf(x_range, bayes_score['mean'], bayes_score['std'])
            ax1.plot(x_range, bayes_pdf, 'r-', linewidth=2, label='Bayesian', alpha=0.7)
            
            ax1.set_title('Team Score Distribution Comparison')
            ax1.set_xlabel('Team Score (Fantasy Points)')
            ax1.set_ylabel('Probability Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Confidence Interval Comparison
            models = ['Monte Carlo', 'Bayesian']
            means = [mc_score['mean'], bayes_score['mean']]
            stds = [mc_score['std'], bayes_score['std']]
            
            y_pos = np.arange(len(models))
            ax2.barh(y_pos, means, xerr=stds, capsize=5, alpha=0.7, 
                    color=['blue', 'red'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(models)
            ax2.set_xlabel('Team Score (Fantasy Points)')
            ax2.set_title('Team Score Comparison with Uncertainty')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax2.text(mean + std + 0.5, i, f'{mean:.1f}±{std:.1f}', 
                        va='center', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            filename = "team_score_comparison.png"
            plot_file = os.path.join(self.output_dir, filename)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {filename}")
            return plot_file
            
        except Exception as e:
            print(f"   ❌ Error creating team score comparison chart: {e}")
            return None
    
    def _create_uncertainty_comparison_chart(self) -> Optional[str]:
        """Create uncertainty comparison chart."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # Extract data
            mc_score = self.monte_carlo_data['monte_carlo_projection']['team_projection']['total_score']
            bayes_score = self.bayesian_data.get('team_projection', {}).get('total_score', 
                         self.bayesian_data.get('model_output', {}).get('team_score', {}))
            
            # Chart 1: Standard Deviation Comparison
            models = ['Monte Carlo', 'Bayesian']
            stds = [mc_score['std'], bayes_score['std']]
            
            bars1 = ax1.bar(models, stds, alpha=0.7, color=['blue', 'red'])
            ax1.set_title('Uncertainty Comparison (Standard Deviation)')
            ax1.set_ylabel('Standard Deviation')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, std in zip(bars1, stds):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{std:.2f}', ha='center', va='bottom')
            
            # Chart 2: Confidence Interval Width Comparison
            mc_ci_width = mc_score['confidence_interval'][1] - mc_score['confidence_interval'][0]
            bayes_ci_width = bayes_score['confidence_interval'][1] - bayes_score['confidence_interval'][0]
            
            ci_widths = [mc_ci_width, bayes_ci_width]
            bars2 = ax2.bar(models, ci_widths, alpha=0.7, color=['blue', 'red'])
            ax2.set_title('Confidence Interval Width Comparison')
            ax2.set_ylabel('CI Width')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, width in zip(bars2, ci_widths):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{width:.2f}', ha='center', va='bottom')
            
            # Chart 3: Percentile Comparison
            percentiles = ['p5', 'p25', 'p50', 'p75', 'p95']
            mc_percentiles = [mc_score['percentiles'][p] for p in percentiles]
            bayes_percentiles = [bayes_score['percentiles'][p] for p in percentiles]
            
            x_pos = np.arange(len(percentiles))
            width = 0.35
            
            ax3.bar(x_pos - width/2, mc_percentiles, width, label='Monte Carlo', 
                   alpha=0.7, color='blue')
            ax3.bar(x_pos + width/2, bayes_percentiles, width, label='Bayesian', 
                   alpha=0.7, color='red')
            
            ax3.set_title('Percentile Comparison')
            ax3.set_xlabel('Percentile')
            ax3.set_ylabel('Team Score')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(percentiles)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Chart 4: Uncertainty Ratio Analysis
            metrics = self.comparison_results['comparison_metrics']
            
            comparison_metrics = [
                metrics['mean_difference'],
                metrics['std_ratio'],
                metrics['confidence_interval_overlap'],
                metrics['percentile_correlation']
            ]
            
            metric_labels = ['Mean Diff', 'Std Ratio', 'CI Overlap', 'Pct Corr']
            
            bars4 = ax4.bar(metric_labels, comparison_metrics, alpha=0.7, color='green')
            ax4.set_title('Comparison Metrics Summary')
            ax4.set_ylabel('Metric Value')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, metric in zip(bars4, comparison_metrics):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{metric:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            filename = "uncertainty_comparison.png"
            plot_file = os.path.join(self.output_dir, filename)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {filename}")
            return plot_file
            
        except Exception as e:
            print(f"   ❌ Error creating uncertainty comparison chart: {e}")
            return None
    
    def _create_model_quality_chart(self) -> Optional[str]:
        """Create model quality comparison chart."""
        try:
            if 'model_selection' not in self.comparison_results:
                print("   ⚠️  No model selection data available")
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            selection = self.comparison_results['model_selection']
            
            # Chart 1: Overall Quality Scores
            models = ['Monte Carlo', 'Bayesian']
            scores = [selection['monte_carlo_quality']['score'], 
                     selection['bayesian_quality']['score']]
            
            bars1 = ax1.bar(models, scores, alpha=0.7, color=['blue', 'red'])
            ax1.set_title('Model Quality Scores')
            ax1.set_ylabel('Quality Score (0-100)')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars1, scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{score:.1f}', ha='center', va='bottom')
            
            # Chart 2: Monte Carlo Quality Breakdown
            mc_quality = selection['monte_carlo_quality']
            mc_metrics = ['Convergence', 'Simulations', 'Execution Time']
            mc_values = [
                30 if mc_quality['convergence'] == 'converged' else 15,
                min(40, mc_quality['simulations'] / 250),  # Scale to 0-40
                min(30, 30 - mc_quality['execution_time'] / 20)  # Scale to 0-30
            ]
            
            bars2 = ax2.bar(mc_metrics, mc_values, alpha=0.7, color='lightblue')
            ax2.set_title('Monte Carlo Quality Breakdown')
            ax2.set_ylabel('Score')
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Bayesian Quality Breakdown
            bayes_quality = selection['bayesian_quality']
            bayes_metrics = ['R-hat', 'Effective Sample Size', 'Draws & Chains']
            bayes_values = [
                min(40, max(0, 40 - (bayes_quality['rhat'] - 1.0) * 100)),
                min(30, bayes_quality['effective_sample_size'] / 66.7),  # Scale to 0-30
                min(30, (bayes_quality['draws'] / 66.7) + (bayes_quality['chains'] * 7.5))  # Scale to 0-30
            ]
            
            bars3 = ax3.bar(bayes_metrics, bayes_values, alpha=0.7, color='lightcoral')
            ax3.set_title('Bayesian Quality Breakdown')
            ax3.set_ylabel('Score')
            ax3.grid(True, alpha=0.3)
            
            # Chart 4: Recommendation Summary
            recommendation = selection['recommendation']
            
            ax4.text(0.1, 0.8, "Primary Recommendation:", fontsize=14, fontweight='bold')
            ax4.text(0.1, 0.7, recommendation['primary_recommendation'], fontsize=12)
            ax4.text(0.1, 0.6, "Reasoning:", fontsize=14, fontweight='bold')
            ax4.text(0.1, 0.5, recommendation['reasoning'], fontsize=12)
            ax4.text(0.1, 0.4, f"Monte Carlo Score: {recommendation['monte_carlo_score']:.1f}", fontsize=12)
            ax4.text(0.1, 0.3, f"Bayesian Score: {recommendation['bayesian_score']:.1f}", fontsize=12)
            ax4.text(0.1, 0.2, f"Score Difference: {recommendation['score_difference']:.1f}", fontsize=12)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Model Selection Recommendation')
            
            plt.tight_layout()
            
            # Save plot
            filename = "model_quality_comparison.png"
            plot_file = os.path.join(self.output_dir, filename)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {filename}")
            return plot_file
            
        except Exception as e:
            print(f"   ❌ Error creating model quality chart: {e}")
            return None
    
    def _create_player_contribution_chart(self) -> Optional[str]:
        """Create player contribution comparison chart."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Extract player data
            mc_players = self.monte_carlo_data['monte_carlo_projection']['player_contributions']
            bayes_players = self.bayesian_data.get('team_projection', {}).get('player_contributions', 
                         self.bayesian_data.get('model_output', {}).get('player_contributions', {}))
            
            if not mc_players or not bayes_players:
                print("   ⚠️  No player contribution data available")
                return None
            
            # Get common players
            common_players = list(set(mc_players.keys()) & set(bayes_players.keys()))
            if not common_players:
                print("   ⚠️  No common players between models")
                return None
            
            # Chart 1: Player Score Comparison
            players = common_players[:10]  # Limit to first 10 players for readability
            
            mc_means = [mc_players[p]['mean'] for p in players]
            bayes_means = [bayes_players[p]['mean'] for p in players]
            
            x_pos = np.arange(len(players))
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, mc_means, width, label='Monte Carlo', 
                           alpha=0.7, color='blue')
            bars2 = ax1.bar(x_pos + width/2, bayes_means, width, label='Bayesian', 
                           alpha=0.7, color='red')
            
            ax1.set_title('Player Score Comparison')
            ax1.set_xlabel('Players')
            ax1.set_ylabel('Fantasy Points')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(players, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Contribution Percentage Comparison
            mc_contribs = [mc_players[p]['contribution_pct'] for p in players]
            bayes_contribs = [bayes_players[p]['contribution_pct'] for p in players]
            
            bars3 = ax2.bar(x_pos - width/2, mc_contribs, width, label='Monte Carlo', 
                           alpha=0.7, color='lightblue')
            bars4 = ax2.bar(x_pos + width/2, bayes_contribs, width, label='Bayesian', 
                           alpha=0.7, color='lightcoral')
            
            ax2.set_title('Player Contribution Percentage Comparison')
            ax2.set_xlabel('Players')
            ax2.set_ylabel('Contribution (%)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(players, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = "player_contribution_comparison.png"
            plot_file = os.path.join(self.output_dir, filename)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {filename}")
            return plot_file
            
        except Exception as e:
            print(f"   ❌ Error creating player contribution chart: {e}")
            return None

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
