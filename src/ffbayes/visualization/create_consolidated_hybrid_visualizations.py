#!/usr/bin/env python3
"""
Create consolidated hybrid visualizations that show all aspects of the Hybrid MC model.
ELIMINATED ALL DUPLICATES - only unique, valuable insights remain.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Import centralized position colors
from ffbayes.visualization.position_colors import POSITION_COLORS, get_legend_elements


class ConsolidatedHybridVisualizer:
    def __init__(self):
        self.baseline_data = None
        self.hybrid_data = None
        self.monte_carlo_data = None
        self.comparison_df = None
        
    def load_model_results(self):
        """Load results from all three models."""
        print("🔄 Loading model results...")
        
        current_year = datetime.now().year
        
        from ffbayes.utils.path_constants import get_hybrid_mc_dir

        # Load baseline model results (now using Hybrid MC as baseline)
        baseline_file = get_hybrid_mc_dir(current_year) / "hybrid_model_results.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            print(f"✅ Baseline (Hybrid MC): {len(self.baseline_data)} players")
        else:
            print("⚠️  Baseline results not found")
            return False
        
        # Load hybrid model results
        hybrid_file = get_hybrid_mc_dir(current_year) / "hybrid_model_results.json"
        if os.path.exists(hybrid_file):
            with open(hybrid_file, 'r') as f:
                self.hybrid_data = json.load(f)
            print(f"✅ Hybrid: {len(self.hybrid_data)} players")
        else:
            print("⚠️  Hybrid results not found")
            return False
        
        # Load Monte Carlo results if available
        from ffbayes.utils.file_naming import find_monte_carlo_file_legacy
        from ffbayes.utils.training_config import get_monte_carlo_training_years
        training_years = get_monte_carlo_training_years()
        mc_file = find_monte_carlo_file_legacy(current_year, training_years)
        if mc_file and mc_file.exists():
            self.monte_carlo_data = pd.read_csv(mc_file, sep='\t')
            print(f"✅ Monte Carlo: {len(self.monte_carlo_data)} players")
            print(f"   📁 Loaded from: {mc_file}")
        else:
            print("⚠️  Monte Carlo results not found")
        
        # Create comparison dataframe
        self._create_comparison_dataframe()
        return True
        
    def _create_comparison_dataframe(self):
        """Create a comprehensive comparison dataframe."""
        print("🔧 Creating comparison dataframe...")
        
        comparisons = []
        
        # Get baseline players from the new Hybrid MC structure
        baseline_players = self.baseline_data
        
        for player in baseline_players.keys():
            if player in self.hybrid_data:
                try:
                    baseline = baseline_players[player]
                    hybrid = self.hybrid_data[player]
                    
                    # Extract values from baseline (Hybrid MC format)
                    baseline_mean = baseline['monte_carlo']['mean']
                    baseline_std = baseline['monte_carlo']['std']
                    
                    # Extract values from hybrid (same data, but for comparison logic)
                    hybrid_mean = hybrid['monte_carlo']['mean']
                    hybrid_std = hybrid['monte_carlo']['std']
                    hybrid_adjusted_std = hybrid['monte_carlo'].get('adjusted_std', hybrid_std)
                    
                    # Since baseline and hybrid are the same data, differences should be minimal
                    mean_diff = hybrid_mean - baseline_mean
                    mean_diff_pct = (mean_diff / baseline_mean) * 100 if baseline_mean != 0 else 0
                    
                    std_improvement = baseline_std - hybrid_adjusted_std
                    std_improvement_pct = (std_improvement / baseline_std) * 100 if baseline_std > 0 else 0
                    
                    # Get uncertainty features
                    bayes = hybrid.get('bayesian_uncertainty', {})
                    overall_uncertainty = bayes.get('overall_uncertainty', 0)
                    data_quality = bayes.get('data_quality_score', 0)
                    consistency = bayes.get('consistency_score', 0)
                    
                    # Get VOR ranking if available
                    vor_validation = hybrid.get('vor_validation', {})
                    vor_rank = vor_validation.get('global_rank', 121)
                    vor_confidence = hybrid.get('vor_match_confidence', 0.0)
                    
                    comparisons.append({
                        'player': player,
                        'position': baseline['position'],
                        'team': baseline.get('team', ''),
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'hybrid_mean': hybrid_mean,
                        'hybrid_std': hybrid_std,
                        'hybrid_adjusted_std': hybrid_adjusted_std,
                        'mean_diff': mean_diff,
                        'mean_diff_pct': mean_diff_pct,
                        'std_improvement': std_improvement,
                        'std_improvement_pct': std_improvement_pct,
                        'overall_uncertainty': overall_uncertainty,
                        'data_quality': data_quality,
                        'consistency': consistency,
                        'vor_rank': vor_rank,
                        'vor_confidence': vor_confidence
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error processing {player}: {e}")
                    continue
        
        self.comparison_df = pd.DataFrame(comparisons)
        print(f"✅ Comparison dataframe created: {len(self.comparison_df)} players")
        
    def create_essential_visualizations(self):
        """Create comprehensive essential visualizations."""
        print("🎨 Creating comprehensive essential visualizations...")
        
        # Create the main comprehensive chart (2x2 instead of 3x3)
        fig = self._create_essential_chart()
        
        # Create VOR validation chart (only if VOR data exists)
        if len(self.comparison_df[self.comparison_df['vor_rank'] != 121]) > 0:
            self._create_vor_validation_chart()
        
        return fig
        
    def _create_essential_chart(self):
        """Create essential 3x3 chart with comprehensive uncertainty insights."""
        print("📊 Creating essential comprehensive chart...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Hybrid MC Model: Essential Insights', fontsize=18, fontweight='bold')
        
        # Panel 1: Uncertainty Improvement vs Prediction Quality (Top Left) - UNIQUE
        ax1 = axes[0, 0]
        self._plot_uncertainty_vs_prediction_quality(ax1)
        ax1.set_title('Uncertainty Improvement vs Prediction Quality', fontweight='bold')
        
        # Panel 2: Uncertainty Improvement Distribution (Top Center) - UNIQUE
        ax2 = axes[0, 1]
        self._plot_uncertainty_improvement_dist(ax2)
        ax2.set_title('Uncertainty Improvement Distribution', fontweight='bold')
        
        # Panel 3: VOR Ranking Validation (Top Right) - UNIQUE
        ax3 = axes[0, 2]
        self._plot_vor_validation(ax3)
        ax3.set_title('VOR Rank vs Improvement', fontweight='bold')
        
        # Panel 4: Position-wise Analysis (Middle Left) - UNIQUE
        ax4 = axes[1, 0]
        self._plot_position_analysis(ax4)
        ax4.set_title('Position-wise Uncertainty Improvement', fontweight='bold')
        
        # Panel 5: Data Quality vs Improvement (Middle Center) - UNIQUE
        ax5 = axes[1, 1]
        self._plot_data_quality_correlation(ax5)
        ax5.set_title('Data Quality vs Improvement', fontweight='bold')
        
        # Panel 6: Risk vs Reward Analysis (Middle Right) - UNIQUE
        ax6 = axes[1, 2]
        self._plot_risk_reward_analysis(ax6)
        ax6.set_title('Risk vs Reward (Uncertainty)', fontweight='bold')
        
        # Panel 7: Top 10 Improvements (Bottom Left) - UNIQUE
        ax7 = axes[2, 0]
        self._plot_top_improvements(ax7)
        ax7.set_title('Top 10 Uncertainty Improvements', fontweight='bold')
        
        # Panel 8: Consistency Analysis (Bottom Center) - UNIQUE
        ax8 = axes[2, 1]
        self._plot_consistency_analysis(ax8)
        ax8.set_title('Consistency vs Improvement', fontweight='bold')
        
        # Panel 9: Summary Statistics (Bottom Right) - UNIQUE
        ax9 = axes[2, 2]
        self._plot_summary_statistics(ax9)
        ax9.set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    def _plot_uncertainty_vs_prediction_quality(self, ax):
        """Plot uncertainty improvement vs prediction quality."""
        # Use average prediction as proxy for quality (higher projections = better players)
        avg_prediction = (self.comparison_df['baseline_mean'] + self.comparison_df['hybrid_mean']) / 2
        uncertainty_improvement = self.comparison_df['std_improvement_pct']
        
        # Color by position to show position-specific patterns
        position_colors = [POSITION_COLORS.get(pos, '#95a5a6') for pos in self.comparison_df['position']]
        
        scatter = ax.scatter(avg_prediction, uncertainty_improvement, 
                           c=position_colors, alpha=0.7, s=50)
        
        ax.set_xlabel('Average Prediction (Fantasy Points)')
        ax.set_ylabel('Uncertainty Improvement (%)')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0% improvement
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
        
        # Add position legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=POSITION_COLORS.get(pos, '#95a5a6'), label=pos) 
                          for pos in self.comparison_df['position'].unique()]
        ax.legend(handles=legend_elements, loc='upper right', title='Positions')
        
        # Calculate correlation between baseline and hybrid predictions
        baseline_pred = self.comparison_df['baseline_mean']
        hybrid_pred = self.comparison_df['hybrid_mean']
        baseline_correlation = np.corrcoef(baseline_pred, hybrid_pred)[0, 1]
        
        # Debug: Print actual correlation and differences
        print(f"🔍 DEBUG: Baseline vs Hybrid correlation: {baseline_correlation:.4f}")
        print(f"🔍 DEBUG: Mean difference: {self.comparison_df['mean_diff'].mean():.4f}")
        print(f"🔍 DEBUG: Std improvement: {self.comparison_df['std_improvement_pct'].mean():.4f}%")
        
        # Add prominent R² display
        if baseline_correlation > 0.95:
            message = f'R² = {baseline_correlation**2:.3f}\nBaseline vs Hybrid MC\nSAME PREDICTIONS!'
            color = 'red'
        elif baseline_correlation > 0.8:
            message = f'R² = {baseline_correlation**2:.3f}\nBaseline vs Hybrid MC\nSIMILAR PREDICTIONS'
            color = 'orange'
        else:
            message = f'R² = {baseline_correlation**2:.3f}\nBaseline vs Hybrid MC\nDIFFERENT PREDICTIONS'
            color = 'green'
        
        ax.text(0.02, 0.98, message, 
                transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9, color='white'))
        
        # Calculate and display correlation for this plot
        plot_correlation = np.corrcoef(avg_prediction, uncertainty_improvement)[0, 1]
        ax.text(0.02, 0.85, f'Plot Correlation: {plot_correlation:.3f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
    def _plot_uncertainty_improvement_dist(self, ax):
        """Plot uncertainty improvement distribution stacked by position."""
        improvements = self.comparison_df['std_improvement_pct']
        positions = self.comparison_df['position']
        
        # Create stacked histogram by position
        position_list = positions.unique()
        position_data = [improvements[positions == pos].values for pos in position_list]
        position_colors_list = [POSITION_COLORS.get(pos, '#95a5a6') for pos in position_list]
        
        ax.hist(position_data, bins=20, alpha=0.8, color=position_colors_list, 
                label=position_list, stacked=True, edgecolor='black', linewidth=0.5)
        
        ax.axvline(improvements.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {improvements.mean():.1f}%')
        ax.set_xlabel('Uncertainty Improvement (%)')
        ax.set_ylabel('Number of Players')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    def _plot_top_improvements(self, ax):
        """Plot top 10 uncertainty improvements."""
        top_improvements = self.comparison_df.nlargest(10, 'std_improvement_pct')
        
        # Color bars by position using consistent colors
        bar_colors = [POSITION_COLORS.get(row['position'], '#95a5a6') for _, row in top_improvements.iterrows()]
        bars = ax.barh(range(len(top_improvements)), top_improvements['std_improvement_pct'], color=bar_colors)
        
        ax.set_yticks(range(len(top_improvements)))
        ax.set_yticklabels([f"{row['player']} ({row['position']})" for _, row in top_improvements.iterrows()])
        ax.set_xlabel('Uncertainty Improvement (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_vor_validation(self, ax):
        """Plot VOR ranking validation."""
        valid_vor = self.comparison_df[self.comparison_df['vor_rank'] != 121]
        if len(valid_vor) > 0:
            # Color code by uncertainty improvement to show the relationship
            scatter = ax.scatter(valid_vor['vor_rank'], valid_vor['std_improvement_pct'], 
                               c=valid_vor['std_improvement_pct'], cmap='RdYlGn', alpha=0.7, s=50)
            ax.set_xlabel('VOR Global Rank')
            ax.set_ylabel('Uncertainty Improvement (%)')
            ax.set_title('VOR Rank vs Improvement (Color = Improvement %)')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to show improvement levels
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Uncertainty Improvement (%)')
        else:
            ax.text(0.5, 0.5, 'No VOR data available', ha='center', va='center', 
                   transform=ax.transAxes)
            
    def _plot_position_analysis(self, ax):
        """Plot position-wise analysis."""
        pos_stats = self.comparison_df.groupby('position')['std_improvement_pct'].agg(['mean', 'count'])
        positions = pos_stats.index
        means = pos_stats['mean']
        counts = pos_stats['count']
        
        bars = ax.bar(positions, means, color=[POSITION_COLORS.get(pos, '#95a5a6') for pos in positions])
        ax.set_xlabel('Position')
        ax.set_ylabel('Average Uncertainty Improvement (%)')
        ax.grid(True, alpha=0.3)
        
        # Add player counts
        for i, (pos, mean, count) in enumerate(zip(positions, means, counts)):
            ax.text(i, mean + 1, f'n={int(count)}', ha='center', va='bottom', fontsize=10)
            
    def _plot_data_quality_correlation(self, ax):
        """Plot data quality vs improvement correlation."""
        # Color code by uncertainty improvement - more meaningful visualization
        scatter = ax.scatter(self.comparison_df['data_quality'], self.comparison_df['std_improvement_pct'], 
                           c=self.comparison_df['std_improvement_pct'], cmap='RdYlGn', alpha=0.7, s=50)
        ax.set_xlabel('Data Quality Score')
        ax.set_ylabel('Uncertainty Improvement (%)')
        ax.set_title('Data Quality vs Improvement (Color = Improvement %)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to show the relationship
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Uncertainty Improvement (%)')
        
    def _plot_risk_reward_analysis(self, ax):
        """Plot risk vs reward analysis."""
        # Color code by improvement percentage to show the risk-reward relationship
        scatter = ax.scatter(self.comparison_df['baseline_std'], self.comparison_df['std_improvement_pct'], 
                           c=self.comparison_df['std_improvement_pct'], cmap='RdYlGn', alpha=0.7, s=50)
        ax.set_xlabel('Baseline Uncertainty (Risk)')
        ax.set_ylabel('Uncertainty Improvement (%)')
        ax.set_title('Risk vs Reward (Color = Improvement %)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to show improvement levels
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Uncertainty Improvement (%)')
        
    def _plot_consistency_analysis(self, ax):
        """Plot consistency analysis."""
        # Color code by uncertainty improvement for better insights
        scatter = ax.scatter(self.comparison_df['consistency'], self.comparison_df['std_improvement_pct'], 
                           c=self.comparison_df['std_improvement_pct'], cmap='RdYlGn', alpha=0.7, s=50)
        ax.set_xlabel('Consistency Score')
        ax.set_ylabel('Uncertainty Improvement (%)')
        ax.set_title('Consistency vs Improvement (Color = Improvement %)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to show the relationship
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Uncertainty Improvement (%)')
        
    def _plot_summary_statistics(self, ax):
        """Plot summary statistics."""
        metrics = ['Uncertainty\nImprovement', 'Prediction\nDifference', 'Players\nImproved', 'Total\nPlayers']
        
        mean_uncertainty_improvement = self.comparison_df['std_improvement_pct'].mean()
        mean_prediction_diff = self.comparison_df['mean_diff_pct'].mean()
        players_with_improvement = len(self.comparison_df[self.comparison_df['std_improvement_pct'] > 0])
        total_players = len(self.comparison_df)
        
        values = [mean_uncertainty_improvement, mean_prediction_diff, players_with_improvement, total_players]
        colors = ['lightgreen', 'lightblue', 'gold', 'lightcoral']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom')
        
        # Add position color legend
        legend_elements = get_legend_elements(self.comparison_df['position'].unique())
        fig = ax.get_figure()
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                  title='Position Colors')
        
    def _create_vor_validation_chart(self):
        """Create VOR validation chart (only if VOR data exists)."""
        print("📊 Creating VOR validation chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VOR Ranking Validation', fontsize=16, fontweight='bold')
        
        # Panel 1: VOR Rank vs Uncertainty Improvement
        ax1 = axes[0, 0]
        valid_vor = self.comparison_df[self.comparison_df['vor_rank'] != 121]
        if len(valid_vor) > 0:
            # Color code by uncertainty improvement to show the relationship
            scatter = ax1.scatter(valid_vor['vor_rank'], valid_vor['std_improvement_pct'], 
                               c=valid_vor['std_improvement_pct'], cmap='RdYlGn', alpha=0.7, s=50)
            ax1.set_xlabel('VOR Global Rank')
            ax1.set_ylabel('Uncertainty Improvement (%)')
            ax1.set_title('VOR Rank vs Improvement (Color = Improvement %)')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar to show improvement levels
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Uncertainty Improvement (%)')
        
        # Panel 2: VOR Confidence Distribution
        ax2 = axes[0, 1]
        valid_confidence = self.comparison_df[self.comparison_df['vor_confidence'] > 0]
        if len(valid_confidence) > 0:
            ax2.hist(valid_confidence['vor_confidence'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax2.set_xlabel('VOR Match Confidence')
            ax2.set_ylabel('Number of Players')
            ax2.set_title('VOR Match Confidence Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Position-wise VOR Coverage
        ax3 = axes[1, 0]
        pos_vor_stats = self.comparison_df.groupby('position').agg({
            'vor_rank': lambda x: (x != 121).sum(),
            'position': 'count'
        })
        pos_vor_stats.columns = ['With_VOR', 'Total']
        pos_vor_stats['Coverage_Pct'] = (pos_vor_stats['With_VOR'] / pos_vor_stats['Total'] * 100).round(1)
        
        bars = ax3.bar(pos_vor_stats.index, pos_vor_stats['Coverage_Pct'], 
                       color=[POSITION_COLORS.get(pos, '#95a5a6') for pos in pos_vor_stats.index])
        ax3.set_xlabel('Position')
        ax3.set_ylabel('VOR Coverage (%)')
        ax3.set_title('Position-wise VOR Coverage')
        ax3.grid(True, alpha=0.3)
        
        # Add coverage percentages
        for i, (pos, row) in enumerate(pos_vor_stats.iterrows()):
            ax3.text(i, row['Coverage_Pct'] + 1, f'{row["Coverage_Pct"]:.1f}%', 
                     ha='center', va='bottom', fontsize=10)
        
        # Panel 4: VOR Rank Distribution
        ax4 = axes[1, 1]
        valid_ranks = self.comparison_df[self.comparison_df['vor_rank'] != 121]['vor_rank']
        if len(valid_ranks) > 0:
            ax4.hist(valid_ranks, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax4.set_xlabel('VOR Global Rank')
            ax4.set_ylabel('Number of Players')
            ax4.set_title('VOR Global Rank Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        from ffbayes.utils.path_constants import get_plots_dir
        output_dir = get_plots_dir(datetime.now().year) / 'hybrid_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chart_path = output_dir / 'vor_validation_chart.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {chart_path}")
        
        # Also save to docs
        docs_dir = Path('docs/images/hybrid_visualizations')
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / 'vor_validation_chart.png'
        fig.savefig(docs_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to docs: {docs_path}")
        
        plt.close(fig)
        
    def save_visualizations(self):
        """Save all visualizations to the consolidated directory."""
        print("💾 Saving essential visualizations...")
        
        # Create output directory
        from ffbayes.utils.path_constants import get_plots_dir
        output_dir = get_plots_dir(datetime.now().year) / 'hybrid_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create essential chart
        fig = self.create_essential_visualizations()
        essential_path = output_dir / 'essential_insights_chart.png'
        fig.savefig(essential_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {essential_path}")
        
        # Also save to docs
        docs_dir = Path('docs/images/hybrid_visualizations')
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / 'essential_insights_chart.png'
        fig.savefig(docs_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to docs: {docs_path}")
        
        plt.close(fig)
        
        print("🎉 Essential visualization creation completed!")
        print(f"📁 Results saved to: {output_dir}")
        print("📚 Also saved to docs for static documentation")
        print("✨ Comprehensive uncertainty analysis completed!")


def main():
    """Main function to run the consolidated visualizer."""
    print("=" * 60)
    print("Creating Comprehensive Hybrid MC Visualizations")
    print("=" * 60)
    
    visualizer = ConsolidatedHybridVisualizer()
    
    if visualizer.load_model_results():
        visualizer.save_visualizations()
    else:
        print("❌ Failed to load data. Please run the models first.")


if __name__ == "__main__":
    main()
