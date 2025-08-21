#!/usr/bin/env python3
"""
Create comprehensive team aggregation visualizations with better output organization.
Addresses QUICK_TEST mode and implements consistent file naming without timestamps.
"""

import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set style for better-looking plots
plt.style.use('default')
# Do not globally override palettes; we color explicitly via POS_PALETTE
# sns.set_palette("husl")

# Shared position palette for consistent colors across plots
POS_PALETTE = {
    'QB': '#4C78A8', 'RB': '#F58518', 'WR': '#54A24B', 'TE': '#B279A2',
    'DST': '#9C755F', 'K': '#E45756', 'UNK': '#7F7F7F'
}

def get_output_directory():
    """Determine output directory based on QUICK_TEST mode."""
    is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
    
    if is_quick_test:
        # Test runs go to a separate directory that can be easily cleaned
        base_dir = "plots/test_runs"
        os.makedirs(base_dir, exist_ok=True)
        return base_dir, True
    else:
        # Production runs go to the organized team_aggregation directory
        base_dir = "plots/team_aggregation"
        os.makedirs(base_dir, exist_ok=True)
        return base_dir, False

def cleanup_old_test_files():
    """Clean up old test files to prevent clutter."""
    test_dir = "plots/test_runs"
    if os.path.exists(test_dir):
        # Remove files older than 1 hour
        current_time = datetime.now()
        for file_path in glob.glob(os.path.join(test_dir, "*.png")):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if (current_time - file_time).total_seconds() > 3600:  # 1 hour
                os.remove(file_path)
                print(f"   🗑️  Cleaned up old test file: {os.path.basename(file_path)}")

def load_latest_team_aggregation_results():
    """Load the most recent team aggregation results."""
    # Look for results in both test and production directories
    search_patterns = [
        "results/team_aggregation/team_aggregation_results_*.json",
        "plots/test_runs/team_aggregation_results_*.json"
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
        raise FileNotFoundError("No team aggregation results found")
    
    print(f"📊 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file

def create_team_score_breakdown_chart(results, output_dir, is_test_mode):
    """Create comprehensive team score breakdown chart."""
    print("📊 Creating team score breakdown chart...")
    
    if 'monte_carlo_projection' not in results or 'player_contributions' not in results['monte_carlo_projection']:
        print("   ⚠️  No Monte Carlo projection data available")
        return None
    
    player_contribs = results['monte_carlo_projection']['player_contributions']
    if not player_contribs:
        print("   ⚠️  No player contribution data available")
        return None
    
    # Extract data and optionally positions
    players = list(player_contribs.keys())
    means = np.array([player_contribs[p]['mean'] for p in players])
    stds = np.array([player_contribs[p]['std'] for p in players])
    contribs = np.array([player_contribs[p]['contribution_pct'] for p in players])
    positions = [player_contribs[p].get('position', 'UNK') for p in players]
    
    # Sort by contribution descending for readability
    order = np.argsort(-contribs)
    players = [players[i] for i in order]
    means = means[order]
    stds = stds[order]
    contribs = contribs[order]
    positions = [positions[i] for i in order]
    
    # Color map by position
    colors = [POS_PALETTE.get(pos, '#7F7F7F') for pos in positions]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Absolute Fantasy Points with slender whiskers
    y_pos = np.arange(len(players))
    bars1 = ax1.barh(y_pos, means, color=colors, alpha=0.75)
    # Draw whiskers for ±std
    ax1.errorbar(means, y_pos, xerr=stds, fmt='none', ecolor='black', elinewidth=1, capsize=3, alpha=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{p} ({pos})" for p, pos in zip(players, positions)])
    ax1.set_xlabel('Fantasy Points (mean ± std)')
    ax1.set_title('Individual Player Fantasy Points')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, means, stds):
        ax1.text(bar.get_width() + max(0.1, 0.01*bar.get_width()), bar.get_y() + bar.get_height()/2, 
                 f'{mean:.1f}±{std:.1f}', va='center', fontsize=9)
    
    # Plot 2: Contribution Percentages (sorted to match left)
    bars2 = ax2.barh(y_pos, contribs, color=colors, alpha=0.75)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(players)
    ax2.set_xlabel('Contribution to Team Total (%)')
    ax2.set_title('Player Contribution Percentages')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, contrib in zip(bars2, contribs):
        ax2.text(bar.get_width() + max(0.1, 0.01*bar.get_width()), bar.get_y() + bar.get_height()/2, 
                 f'{contrib:.1f}%', va='center', fontsize=9)
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Team Score Breakdown (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
    # Add shared legend for positions
    unique_positions = list(dict.fromkeys(positions))
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=pos,
                                  markerfacecolor=POS_PALETTE.get(pos, '#7F7F7F'), markersize=10)
                      for pos in unique_positions]
    ax1.legend(handles=legend_handles, title='Position', loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    # Save with consistent filename (no timestamp for latest version)
    filename = "team_score_breakdown_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return plot_file

def create_position_analysis_chart(results, output_dir, is_test_mode):
    """Create position-based analysis chart."""
    print("📊 Creating position analysis charts...")
    
    if 'monte_carlo_projection' not in results or 'player_contributions' not in results['monte_carlo_projection']:
        print("   ⚠️  No Monte Carlo projection data available")
        return None
    
    player_contribs = results['monte_carlo_projection']['player_contributions']
    if not player_contribs:
        print("   ⚠️  No player contribution data available")
        return None
    
    # Group players by position (use embedded position if present)
    position_data = {}
    for player, data in player_contribs.items():
        pos = data.get('position', 'UNK')
        if pos not in position_data:
            position_data[pos] = {'players': [], 'points': [], 'contribs': []}
        position_data[pos]['players'].append(player)
        position_data[pos]['points'].append(data['mean'])
        position_data[pos]['contribs'].append(data['contribution_pct'])
    
    # Sort positions by total points desc for a more informative view
    pos_order = sorted(position_data.keys(), key=lambda p: sum(position_data[p]['points']), reverse=True)
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    positions = pos_order
    total_points = [sum(position_data[p]['points']) for p in positions]
    avg_points = [np.mean(position_data[p]['points']) if position_data[p]['points'] else 0 for p in positions]
    total_contribs = [sum(position_data[p]['contribs']) for p in positions]
    player_counts = [len(position_data[p]['players']) for p in positions]
    
    # Plot 1: Total Points by Position
    bars1 = ax1.bar(positions, total_points, color=[POS_PALETTE.get(p, '#7F7F7F') for p in positions], alpha=0.9)
    ax1.set_title('Total Fantasy Points by Position')
    ax1.set_ylabel('Total Points')
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, total in zip(bars1, total_points):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{total:.1f}', ha='center', va='bottom')
    
    # Plot 2: Average Points per Player by Position
    bars2 = ax2.bar(positions, avg_points, color=[POS_PALETTE.get(p, '#7F7F7F') for p in positions], alpha=0.9)
    ax2.set_title('Average Fantasy Points per Player by Position')
    ax2.set_ylabel('Average Points')
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, avg in zip(bars2, avg_points):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{avg:.1f}', ha='center', va='bottom')
    
    # Plot 3: Contribution Percentage by Position
    bars3 = ax3.bar(positions, total_contribs, color=[POS_PALETTE.get(p, '#7F7F7F') for p in positions], alpha=0.9)
    ax3.set_title('Total Contribution Percentage by Position')
    ax3.set_ylabel('Contribution (%)')
    ax3.grid(True, axis='y', alpha=0.3)
    for bar, contrib in zip(bars3, total_contribs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{contrib:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Player Count by Position
    bars4 = ax4.bar(positions, player_counts, color=[POS_PALETTE.get(p, '#7F7F7F') for p in positions], alpha=0.9)
    ax4.set_title('Number of Players by Position')
    ax4.set_ylabel('Player Count')
    ax4.grid(True, axis='y', alpha=0.3)
    for bar, count in zip(bars4, player_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{count}', ha='center', va='bottom')
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Position Analysis (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
    # Add legend consistent with position palette
    legend_positions = positions
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', label=pos,
                                  markerfacecolor=POS_PALETTE.get(pos, '#7F7F7F'), markersize=10)
                      for pos in legend_positions]
    ax1.legend(handles=legend_handles, title='Position', loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save with consistent filename
    filename = "position_analysis_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return plot_file

def create_uncertainty_analysis_chart(results, output_dir, is_test_mode):
    """Create uncertainty analysis chart."""
    print("📊 Creating uncertainty analysis charts...")
    
    if 'monte_carlo_projection' not in results or 'player_contributions' not in results['monte_carlo_projection']:
        print("   ⚠️  No Monte Carlo projection data available")
        return None
    
    player_contribs = results['monte_carlo_projection']['player_contributions']
    if not player_contribs:
        print("   ⚠️  No player contribution data available")
        return None
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    players = list(player_contribs.keys())
    means = [player_contribs[p]['mean'] for p in players]
    stds = [player_contribs[p]['std'] for p in players]
    contribs = [player_contribs[p]['contribution_pct'] for p in players]
    positions = [player_contribs[p].get('position', 'UNK') for p in players]
    colors = [POS_PALETTE.get(pos, '#7F7F7F') for pos in positions]
    
    # Calculate coefficient of variation (uncertainty metric)
    cv_values = [std/mean if mean > 0 else 0 for mean, std in zip(means, stds)]
    
    # Plot 1: Uncertainty vs Team Contribution
    scatter1 = ax1.scatter(contribs, cv_values, c=colors, s=100, alpha=0.8, edgecolors='k', linewidths=0.3)
    ax1.set_xlabel('Contribution to Team Total (%)')
    ax1.set_ylabel('Coefficient of Variation (Uncertainty)')
    ax1.set_title('Player Uncertainty vs Team Contribution')
    ax1.grid(True, alpha=0.3)
    
    # Add legend by position
    unique_pos = sorted(set(positions))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=pos,
                                  markerfacecolor=POS_PALETTE.get(pos, '#7F7F7F'), markeredgecolor='k', markersize=8)
                      for pos in unique_pos]
    ax1.legend(handles=legend_handles, title='Position', loc='upper right', fontsize=8)
    
    # Add player labels
    for i, player in enumerate(players):
        ax1.annotate(player, (contribs[i], cv_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Distribution of Player Uncertainty
    ax2.hist(cv_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Coefficient of Variation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Player Uncertainty')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Player Mean vs Standard Deviation
    scatter3 = ax3.scatter(means, stds, c=colors, s=100, alpha=0.8, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('Mean Fantasy Points')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Player Mean vs Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Legend by position
    ax3.legend(handles=legend_handles, title='Position', loc='upper left', fontsize=8)
    
    # Add player labels
    for i, player in enumerate(players):
        ax3.annotate(player, (means[i], stds[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Team Score Confidence Analysis
    if 'team_projection' in results and 'total_score' in results['team_projection']:
        team_score = results['team_projection']['total_score']['mean']
        team_std = results['team_projection']['total_score']['std']
        
        # Create a range around the team score for visualization
        x_range = np.linspace(team_score - 3*team_std, team_score + 3*team_std, 100)
        
        if team_std > 0:
            # Normal distribution if there's actual uncertainty
            y_values = stats.norm.pdf(x_range, team_score, team_std)
            ax4.plot(x_range, y_values, 'b-', linewidth=2, label='Team Score Distribution')
        else:
            # Single point if deterministic (test mode)
            ax4.axvline(x=team_score, color='red', linestyle='--', linewidth=3, 
                       label=f'Deterministic Score: {team_score:.1f}')
            y_values = [1.0]  # Dummy value for single point
        
        # Add confidence intervals and percentiles
        if team_std > 0:
            ci_95_lower = team_score - 1.96 * team_std
            ci_95_upper = team_score + 1.96 * team_std
            p5 = team_score - 1.645 * team_std
            p25 = team_score - 0.675 * team_std
            p75 = team_score + 0.675 * team_std
            p95 = team_score + 1.645 * team_std
            
            ax4.axvline(ci_95_lower, color='green', linestyle=':', linewidth=2, 
                        label=f'95% CI Lower: {ci_95_lower:.1f}')
            ax4.axvline(ci_95_upper, color='green', linestyle=':', linewidth=2, 
                        label=f'95% CI Upper: {ci_95_upper:.1f}')
            ax4.axvline(p5, color='yellow', linestyle='--', linewidth=2, 
                        label=f'P5: {p5:.1f}')
            ax4.axvline(p25, color='yellow', linestyle='--', linewidth=2, 
                        label=f'P25: {p25:.1f}')
            ax4.axvline(p75, color='yellow', linestyle='--', linewidth=2, 
                        label=f'P75: {p75:.1f}')
            ax4.axvline(p95, color='yellow', linestyle='--', linewidth=2, 
                        label=f'P95: {p95:.1f}')
        else:
            # For deterministic scores, show the single value
            ax4.axvline(team_score, color='red', linestyle='--', linewidth=3)
        
        ax4.set_xlabel('Team Score (Fantasy Points)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Team Score Confidence Analysis')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Uncertainty Analysis (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
        # Add warning text
        fig.text(0.5, 0.02, '⚠️  WARNING: Results are deterministic due to QUICK_TEST mode. ' +
                'For realistic uncertainty analysis, run with QUICK_TEST=false', 
                ha='center', fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.tight_layout()
    
    # Save with consistent filename
    filename = "uncertainty_analysis_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return plot_file

def create_comparison_insights_chart(results, output_dir, is_test_mode):
    """Create comparison insights chart."""
    print("📊 Creating comparison insights chart...")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Model Performance Comparison (if available)
    model_comp = results.get('model_comparison')
    # If not embedded, try to load latest model comparison results
    if not model_comp:
        comp_files = sorted(glob.glob('results/model_comparison/model_comparison_results_*.json'))
        if comp_files:
            try:
                with open(comp_files[-1], 'r') as f:
                    comp_data = json.load(f)
                model_comp = {
                    'Bayesian': {'mae': comp_data.get('comparison_metrics', {}).get('bayesian_mae', None)},
                    'Baseline': {'mae': comp_data.get('comparison_metrics', {}).get('baseline_mae', None)}
                }
            except Exception:
                model_comp = None
    if model_comp:
        models = list(model_comp.keys())
        mae_values = [model_comp[m].get('mae', 0) or 0 for m in models]
        
        bars1 = ax1.bar(models, mae_values, alpha=0.7, color=['lightblue', 'lightcoral'])
        ax1.set_title('Model Performance Comparison (MAE)')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mae in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mae:.3f}', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'No model comparison data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Model Performance Comparison')
    
    # Plot 2: Team Statistics Summary (annotated metrics + CI)
    ax2.axis('off')
    if 'team_projection' in results:
        team_stats = results['team_projection']['total_score']
        mean = team_stats['mean']; std = team_stats['std']
        vmin = team_stats.get('min', mean); vmax = team_stats.get('max', mean)
        ci = team_stats.get('confidence_interval', [mean - 1.96*std, mean + 1.96*std])
        percentiles = team_stats.get('percentiles', {})
        lines = [
            f"Mean: {mean:.1f} points",
            f"Std Dev: {std:.1f} points",
            f"Min/Max: {vmin:.1f} / {vmax:.1f}",
            f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]",
            f"Percentiles: P5 {percentiles.get('p5', 0):.1f}, P50 {percentiles.get('p50', 0):.1f}, P95 {percentiles.get('p95', 0):.1f}",
        ]
        ax2.text(0.02, 0.95, 'Team Score Summary', fontsize=14, fontweight='bold', va='top')
        for i, text in enumerate(lines):
            ax2.text(0.02, 0.9 - i*0.12, text, fontsize=12, va='top')
    else:
        ax2.text(0.5, 0.5, 'No team projection data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Team Score Statistics')
    
    # Plot 3: Roster Coverage Analysis pie
    if 'roster_analysis' in results:
        coverage = results['roster_analysis'].get('roster_coverage_percentage', 0)
        missing_players = results['roster_analysis'].get('missing_players_count', 0)
        total_players = results['roster_analysis'].get('total_roster_size', 0)
        labels = ['Covered Players', 'Missing Players']
        sizes = [coverage, 100 - coverage]
        colors = ['lightgreen', 'lightcoral']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Roster Coverage Analysis\n({total_players - missing_players}/{total_players} players)')
    else:
        ax3.text(0.5, 0.5, 'No roster analysis data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Roster Coverage Analysis')
    
    # Plot 4: Key Insights and Recommendations (unchanged)
    insights = []
    if 'model_comparison' in results:
        mae_ratio = results['model_comparison'].get('mae_ratio', 1.0)
        if mae_ratio > 1.2:
            insights.append(f"Monte Carlo shows {mae_ratio:.1f}x higher error than Bayesian")
        elif mae_ratio < 0.8:
            insights.append(f"Bayesian shows {1/mae_ratio:.1f}x higher error than Monte Carlo")
        else:
            insights.append("Both models show similar performance")
    if 'roster_analysis' in results:
        coverage = results['roster_analysis'].get('roster_coverage_percentage', 0)
        if coverage < 80:
            insights.append(f"⚠️  Low roster coverage: {coverage:.1f}%")
        elif coverage < 95:
            insights.append(f"⚠️  Moderate roster coverage: {coverage:.1f}%")
        else:
            insights.append(f"✅ Excellent roster coverage: {coverage:.1f}%")
    if 'team_projection' in results:
        team_score = results['team_projection']['total_score']['mean']
        if team_score > 120:
            insights.append(f"🎯 High-scoring team projection: {team_score:.1f} points")
        elif team_score < 100:
            insights.append(f"📉 Low-scoring team projection: {team_score:.1f} points")
        else:
            insights.append(f"📊 Moderate team projection: {team_score:.1f} points")
    if not insights:
        insights = [
            "📊 Team aggregation completed successfully",
            "📈 Individual player predictions aggregated to team level",
            "🎯 Uncertainty properly propagated from individual to team",
            "📋 Results ready for draft strategy analysis"
        ]
    ax4.text(0.05, 0.95, 'Key Insights & Recommendations:', 
             transform=ax4.transAxes, fontsize=14, fontweight='bold')
    for i, insight in enumerate(insights):
        y_pos = 0.85 - (i * 0.15)
        ax4.text(0.05, y_pos, f"• {insight}", 
                transform=ax4.transAxes, fontsize=11, wrap=True)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Actionable Insights')
    
    if is_test_mode:
        fig.suptitle('Comparison Insights (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
    plt.tight_layout()
    filename = "comparison_insights_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")
    return plot_file

def main():
    """Main function to create all visualizations."""
    print("=" * 70)
    print("Creating Comprehensive Team Aggregation Visualizations")
    print("Generating additional insights and analysis charts")
    print("=" * 70)
    
    try:
        # Get output directory and test mode status
        output_dir, is_test_mode = get_output_directory()
        
        # Clean up old test files if in test mode
        if is_test_mode:
            cleanup_old_test_files()
            print(f"📁 Test mode detected - saving to: {output_dir}")
            print("   ⚠️  Results will be deterministic due to QUICK_TEST mode")
        else:
            print(f"📁 Production mode - saving to: {output_dir}")
        
        # Load results
        results, source_file = load_latest_team_aggregation_results()
        
        # Create all visualizations
        plot_files = []
        
        # Generate charts
        plot1 = create_team_score_breakdown_chart(results, output_dir, is_test_mode)
        if plot1:
            plot_files.append(plot1)
        
        plot2 = create_position_analysis_chart(results, output_dir, is_test_mode)
        if plot2:
            plot_files.append(plot2)
        
        plot3 = create_uncertainty_analysis_chart(results, output_dir, is_test_mode)
        if plot3:
            plot_files.append(plot3)
        
        plot4 = create_comparison_insights_chart(results, output_dir, is_test_mode)
        if plot4:
            plot_files.append(plot4)
        
        print("\n" + "=" * 70)
        print("Visualization Summary:")
        print(f"- Generated {len(plot_files)} comprehensive charts")
        print(f"- All plots saved to: {output_dir}")
        
        if is_test_mode:
            print("- ⚠️  QUICK_TEST mode: Results are deterministic for testing")
            print("- 📁 Test files will be automatically cleaned up after 1 hour")
            print("- 🎯 For production analysis, run with QUICK_TEST=false")
        else:
            print("- ✅ Production mode: Full uncertainty analysis enabled")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main()
