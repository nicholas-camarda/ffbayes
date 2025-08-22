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
    """Create comprehensive position analysis chart with proper coloring and insights."""
    print("📊 Creating position analysis chart...")
    
    if 'monte_carlo_projection' not in results or 'player_contributions' not in results['monte_carlo_projection']:
        print("   ⚠️  No Monte Carlo projection data available")
        return None
    
    player_contribs = results['monte_carlo_projection']['player_contributions']
    if not player_contribs:
        print("   ⚠️  No player contribution data available")
        return None
    
    # Extract and organize data by position
    position_data = {}
    for player, data in player_contribs.items():
        pos = data.get('position', 'UNK')
        if pos not in position_data:
            position_data[pos] = {'players': [], 'means': [], 'stds': [], 'contribs': []}
        
        position_data[pos]['players'].append(player)
        position_data[pos]['means'].append(data['mean'])
        position_data[pos]['stds'].append(data['std'])
        position_data[pos]['contribs'].append(data['contribution_pct'])
    
    # Create comprehensive position analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Position Analysis & Team Construction Insights', fontsize=18, fontweight='bold')
    
    positions = list(position_data.keys())
    colors = [POS_PALETTE.get(pos, '#7F7F7F') for pos in positions]
    
    # Plot 1: Total Points by Position (Stacked Bar)
    total_points_by_pos = [sum(position_data[pos]['means']) for pos in positions]
    bars1 = ax1.bar(positions, total_points_by_pos, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Total Fantasy Points by Position', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Points', fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels and insights
    for bar, total, pos in zip(bars1, total_points_by_pos, positions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{total:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add position insights
        player_count = len(position_data[pos]['players'])
        avg_points = total / player_count if player_count > 0 else 0
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{player_count} players\navg: {avg_points:.1f}', 
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Plot 2: Average Points per Player by Position
    avg_points_by_pos = [np.mean(position_data[pos]['means']) for pos in positions]
    bars2 = ax2.bar(positions, avg_points_by_pos, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Average Fantasy Points per Player by Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Points per Player', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, avg, pos in zip(bars2, avg_points_by_pos, positions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{avg:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 3: Position Contribution to Team Total
    total_contrib_by_pos = [sum(position_data[pos]['contribs']) for pos in positions]
    bars3 = ax3.bar(positions, total_contrib_by_pos, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Position Contribution to Team Total (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Contribution (%)', fontsize=12)
    ax3.grid(True, axis='y', alpha=0.3)
    
    for bar, contrib, pos in zip(bars3, total_contrib_by_pos, positions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{contrib:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 4: Position Efficiency (Points per Roster Spot)
    efficiency_by_pos = [total_points_by_pos[i] / len(position_data[pos]['players']) if len(position_data[pos]['players']) > 0 else 0 
                        for i, pos in enumerate(positions)]
    bars4 = ax4.bar(positions, efficiency_by_pos, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Position Efficiency (Points per Roster Spot)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Points per Roster Spot', fontsize=12)
    ax4.grid(True, axis='y', alpha=0.3)
    
    for bar, eff, pos in zip(bars4, efficiency_by_pos, positions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{eff:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add comprehensive legend with position colors
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=POS_PALETTE.get(pos, '#7F7F7F'), 
                                   edgecolor='black', linewidth=1, label=f'{pos} ({len(position_data[pos]["players"])} players)')
                      for pos in positions]
    ax1.legend(handles=legend_elements, title='Position Breakdown', loc='upper right', fontsize=10)
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Position Analysis (QUICK_TEST MODE - Limited Data)', 
                    fontsize=18, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with consistent filename
    filename = "position_analysis_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return plot_file

def create_uncertainty_analysis_chart(results, output_dir, is_test_mode):
    """Create meaningful uncertainty analysis chart with actionable insights."""
    print("📊 Creating uncertainty analysis charts...")
    
    if 'monte_carlo_projection' not in results or 'player_contributions' not in results['monte_carlo_projection']:
        print("   ⚠️  No Monte Carlo projection data available")
        return None
    
    player_contribs = results['monte_carlo_projection']['player_contributions']
    if not player_contribs:
        print("   ⚠️  No player contribution data available")
        return None
    
    # Create comprehensive uncertainty analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Player Uncertainty Analysis & Risk Assessment', fontsize=18, fontweight='bold')
    
    players = list(player_contribs.keys())
    means = [player_contribs[p]['mean'] for p in players]
    stds = [player_contribs[p]['std'] for p in players]
    contribs = [player_contribs[p]['contribution_pct'] for p in players]
    positions = [player_contribs[p].get('position', 'UNK') for p in players]
    colors = [POS_PALETTE.get(pos, '#7F7F7F') for pos in positions]
    
    # Calculate meaningful uncertainty metrics
    cv_values = [std/mean if mean > 0 else 0 for mean, std in zip(means, stds)]  # Coefficient of variation
    risk_scores = [std * contrib/100 for std, contrib in zip(stds, contribs)]  # Risk = uncertainty × impact
    
    # Plot 1: Player Risk vs Impact Matrix
    scatter1 = ax1.scatter(contribs, risk_scores, c=colors, s=150, alpha=0.8, edgecolors='black', linewidths=1)
    ax1.set_xlabel('Team Impact (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Risk Score (Uncertainty × Impact)', fontsize=12, fontweight='bold')
    ax1.set_title('Player Risk vs Impact Matrix', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add risk zones
    ax1.axhline(y=np.mean(risk_scores), color='red', linestyle='--', alpha=0.7, label='Average Risk')
    ax1.axvline(x=np.mean(contribs), color='blue', linestyle='--', alpha=0.7, label='Average Impact')
    
    # Add player labels with risk insights
    for i, player in enumerate(players):
        risk_level = "HIGH" if risk_scores[i] > np.mean(risk_scores) else "LOW"
        ax1.annotate(f'{player}\n({risk_level})', (contribs[i], risk_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Plot 2: Uncertainty Distribution by Position
    pos_uncertainty = {}
    for i, pos in enumerate(positions):
        if pos not in pos_uncertainty:
            pos_uncertainty[pos] = []
        pos_uncertainty[pos].append(cv_values[i])
    
    pos_labels = list(pos_uncertainty.keys())
    pos_means = [np.mean(pos_uncertainty[pos]) for pos in pos_labels]
    pos_colors = [POS_PALETTE.get(pos, '#7F7F7F') for pos in pos_labels]
    
    bars2 = ax2.bar(pos_labels, pos_means, color=pos_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Average Uncertainty by Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (Uncertainty)', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, mean, pos in zip(bars2, pos_means, pos_labels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{mean:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 3: Player Reliability Ranking
    reliability_scores = [1/(1+cv) for cv in cv_values]  # Higher = more reliable
    sorted_indices = np.argsort(reliability_scores)[::-1]  # Sort by reliability descending
    
    sorted_players = [players[i] for i in sorted_indices]
    sorted_reliability = [reliability_scores[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars3 = ax3.bar(range(len(sorted_players)), sorted_reliability, color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Player Reliability Ranking (Higher = More Predictable)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Reliability Score', fontsize=12)
    ax3.set_xlabel('Players (Ranked by Reliability)', fontsize=12)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add player labels
    ax3.set_xticks(range(len(sorted_players)))
    ax3.set_xticklabels(sorted_players, rotation=45, ha='right', fontsize=9)
    
    # Plot 4: Uncertainty vs Expected Performance
    scatter4 = ax4.scatter(means, stds, c=colors, s=150, alpha=0.8, edgecolors='black', linewidths=1)
    ax4.set_xlabel('Expected Points', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Standard Deviation (Uncertainty)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance vs Uncertainty Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(means, stds, 1)
    p = np.poly1d(z)
    ax4.plot(means, p(means), "r--", alpha=0.8, label=f'Trend: σ = {z[0]:.2f}μ + {z[1]:.2f}')
    
    # Add player labels
    for i, player in enumerate(players):
        ax4.annotate(player, (means[i], stds[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Add comprehensive legend
    unique_pos = sorted(set(positions))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=pos,
                                  markerfacecolor=POS_PALETTE.get(pos, '#7F7F7F'), markeredgecolor='black', markersize=8)
                      for pos in unique_pos]
    ax1.legend(handles=legend_handles, title='Position', loc='upper right', fontsize=10)
    ax4.legend(loc='upper left', fontsize=10)
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Player Uncertainty Analysis (QUICK_TEST MODE - Limited Data)', 
                    fontsize=18, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with consistent filename
    filename = "uncertainty_analysis_latest.png"
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return plot_file

def create_comparison_insights_chart(results, output_dir, is_test_mode):
    """Create meaningful comparison insights chart with actionable analysis."""
    print("📊 Creating comparison insights chart...")
    
    # Create comprehensive comparison dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Team Construction Insights & Model Performance Analysis', fontsize=18, fontweight='bold')
    
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
    
    if model_comp and any(model_comp[m].get('mae') for m in model_comp):
        models = list(model_comp.keys())
        mae_values = [model_comp[m].get('mae', 0) or 0 for m in models]
        
        # Color based on performance (lower MAE = better)
        colors = ['lightgreen' if mae == min(mae_values) else 'lightcoral' for mae in mae_values]
        
        bars1 = ax1.bar(models, mae_values, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        ax1.set_title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error (Lower = Better)', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels and performance insights
        for bar, mae, model in zip(bars1, mae_values, models):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mae:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Add performance insight
            if mae == min(mae_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                        'BEST', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Add improvement calculation
        if len(mae_values) == 2:
            improvement = ((max(mae_values) - min(mae_values)) / max(mae_values)) * 100
            ax1.text(0.5, 0.9, f'Improvement: {improvement:.1f}%', 
                    transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    else:
        ax1.text(0.5, 0.5, 'No model comparison data available\nRun model comparison first', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    
    # Plot 2: Team Construction Analysis
    ax2.axis('off')
    if 'team_projection' in results and 'total_score' in results['team_projection']:
        team_stats = results['team_projection']['total_score']
        mean = team_stats.get('mean', 0)
        std = team_stats.get('std', 0)
        vmin = team_stats.get('min', mean - 2*std)
        vmax = team_stats.get('max', mean + 2*std)
        
        # Calculate confidence intervals
        ci_95 = [mean - 1.96*std, mean + 1.96*std] if std > 0 else [mean, mean]
        ci_68 = [mean - std, mean + std] if std > 0 else [mean, mean]
        
        # Team construction insights
        insights = []
        insights.append(f"🎯 Team Projection: {mean:.1f} ± {std:.1f} points")
        insights.append(f"📊 95% Confidence: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")
        insights.append(f"📈 68% Confidence: [{ci_68[0]:.1f}, {ci_68[1]:.1f}]")
        insights.append(f"📉 Score Range: {vmin:.1f} - {vmax:.1f}")
        
        # Add team strength assessment
        if mean > 120:
            team_strength = "🏆 ELITE TEAM"
            strength_color = "darkgreen"
        elif mean > 110:
            team_strength = "🥇 STRONG TEAM"
            strength_color = "green"
        elif mean > 100:
            team_strength = "🥈 COMPETITIVE TEAM"
            strength_color = "orange"
        else:
            team_strength = "🥉 DEVELOPING TEAM"
            strength_color = "red"
        
        insights.append(f"\n{team_strength}")
        
        # Display insights
        ax2.text(0.02, 0.95, 'Team Construction Analysis', fontsize=16, fontweight='bold', va='top')
        for i, insight in enumerate(insights):
            color = strength_color if 'TEAM' in insight else 'black'
            fontweight = 'bold' if 'TEAM' in insight else 'normal'
            ax2.text(0.02, 0.9 - i*0.08, insight, fontsize=12, va='top', color=color, fontweight=fontweight)
    else:
        ax2.text(0.5, 0.5, 'No team projection data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
        ax2.set_title('Team Construction Analysis', fontsize=14, fontweight='bold')
    
    # Plot 3: Roster Coverage & Quality Analysis
    if 'roster_analysis' in results:
        coverage = results['roster_analysis'].get('roster_coverage_percentage', 0)
        missing_players = results['roster_analysis'].get('missing_players_count', 0)
        total_players = results['roster_analysis'].get('total_roster_size', 0)
        
        # Roster quality assessment
        if coverage >= 95:
            quality = "🏆 EXCELLENT"
            quality_color = "darkgreen"
        elif coverage >= 85:
            quality = "🥇 GOOD"
            quality_color = "green"
        elif coverage >= 75:
            quality = "🥈 FAIR"
            quality_color = "orange"
        else:
            quality = "🥉 POOR"
            quality_color = "red"
        
        # Create enhanced pie chart
        labels = ['Covered Players', 'Missing Players']
        sizes = [coverage, 100 - coverage]
        colors_pie = ['lightgreen', 'lightcoral']
        explode = (0.05, 0)  # Slight emphasis on covered players
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90, explode=explode)
        
        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax3.set_title(f'Roster Coverage Analysis\n{quality} ({total_players - missing_players}/{total_players} players)', 
                     fontsize=14, fontweight='bold', color=quality_color)
    else:
        ax3.text(0.5, 0.5, 'No roster analysis data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
        ax3.set_title('Roster Coverage Analysis', fontsize=14, fontweight='bold')
    
    # Plot 4: Key Insights and Strategic Recommendations
    insights = []
    
    # Model performance insights
    if model_comp and any(model_comp[m].get('mae') for m in model_comp):
        mae_values = [model_comp[m].get('mae', 0) or 0 for m in model_comp]
        if len(mae_values) == 2:
            best_model = models[mae_values.index(min(mae_values))]
            improvement = ((max(mae_values) - min(mae_values)) / max(mae_values)) * 100
            insights.append(f"📊 {best_model} model performs best (MAE: {min(mae_values):.3f})")
            if improvement > 5:
                insights.append(f"🎯 Significant improvement: {improvement:.1f}% over baseline")
            else:
                insights.append(f"⚠️  Minimal improvement: {improvement:.1f}% over baseline")
    
    # Team construction insights
    if 'team_projection' in results and 'total_score' in results['team_projection']:
        mean = results['team_projection']['total_score']['mean']
        if mean > 120:
            insights.append(f"🏆 Elite team projection: {mean:.1f} points")
            insights.append("💡 Consider conservative draft strategy")
        elif mean > 110:
            insights.append(f"🥇 Strong team projection: {mean:.1f} points")
            insights.append("💡 Balanced approach recommended")
        elif mean > 100:
            insights.append(f"🥈 Competitive team: {mean:.1f} points")
            insights.append("💡 Focus on high-upside players")
        else:
            insights.append(f"🥉 Developing team: {mean:.1f} points")
            insights.append("💡 Aggressive draft strategy needed")
    
    # Roster insights
    if 'roster_analysis' in results:
        coverage = results['roster_analysis'].get('roster_coverage_percentage', 0)
        if coverage < 80:
            insights.append(f"⚠️  Low roster coverage: {coverage:.1f}%")
            insights.append("💡 Focus on filling missing positions")
        elif coverage < 95:
            insights.append(f"⚠️  Moderate coverage: {coverage:.1f}%")
            insights.append("💡 Consider position-specific strategies")
        else:
            insights.append(f"✅ Excellent coverage: {coverage:.1f}%")
            insights.append("💡 Optimize for best available players")
    
    # Add general insights if none generated
    if not insights:
        insights = [
            "📊 Team aggregation completed successfully",
            "📈 Individual player predictions aggregated to team level",
            "🎯 Use insights to inform draft strategy",
            "💡 Monitor model performance for improvements"
        ]
    
    # Display insights
    ax4.axis('off')
    ax4.text(0.02, 0.95, 'Strategic Insights & Recommendations', fontsize=16, fontweight='bold', va='top')
    
    for i, insight in enumerate(insights):
        # Color code different types of insights
        if '🏆' in insight or '🥇' in insight:
            color = 'darkgreen'
        elif '⚠️' in insight:
            color = 'darkorange'
        elif '💡' in insight:
            color = 'darkblue'
        else:
            color = 'black'
        
        ax4.text(0.02, 0.9 - i*0.06, insight, fontsize=11, va='top', color=color, fontweight='bold')
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Team Construction Insights (QUICK_TEST MODE - Limited Data)', 
                    fontsize=18, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with consistent filename
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
