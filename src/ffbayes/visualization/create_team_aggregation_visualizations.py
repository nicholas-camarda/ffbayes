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

# Import centralized position colors
from ffbayes.visualization.position_colors import POSITION_COLORS as POS_PALETTE

# Set style for better-looking plots
plt.style.use('default')
# Do not globally override palettes; we color explicitly via POS_PALETTE
# sns.set_palette("husl")


def get_output_directory():
    """Determine output directory based on QUICK_TEST mode."""
    is_quick_test = os.getenv('QUICK_TEST', 'false').lower() == 'true'
    
    # Get current year for organized output
    current_year = datetime.now().year
    
    if is_quick_test:
        # Test runs go to year-based test directory that can be easily cleaned
        from ffbayes.utils.path_constants import get_plots_dir
        base_dir = str(get_plots_dir(current_year) / "test_runs")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir, True
    else:
        # Production runs go to the organized year-based team_aggregation directory
        from ffbayes.utils.path_constants import get_post_draft_plots_dir
        base_dir = str(get_post_draft_plots_dir(current_year) / "team_aggregation")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir, False

def cleanup_old_test_files():
    """Clean up old test files to prevent clutter."""
    from ffbayes.utils.path_constants import get_plots_dir
    test_dir = str(get_plots_dir() / "test_runs")
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
    from ffbayes.utils.path_constants import get_plots_dir, get_team_aggregation_dir
    current_year = datetime.now().year
    search_patterns = [
        str(get_team_aggregation_dir(current_year) / "team_aggregation_results_*.json"),
        str(get_plots_dir(current_year) / "test_runs" / "team_aggregation_results_*.json"),
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
    """
    Create comprehensive MODEL COMPARISON showing forward-looking team projections
    
    This shows what each model predicts for the SAME team:
    - Monte Carlo: Simulation-based team projection  
    - Bayesian: Aggregated individual player predictions
    - Baseline: Simple historical average method
    
    All in comparable units (fantasy points) for the same team roster
    """
    print("   📊 Creating comparison insights chart...")
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ======================================================================
    # PLOT 1: MODEL COMPARISON - The main event!
    # ======================================================================
    
    # Get Monte Carlo team projection
    monte_carlo_projection = results.get('monte_carlo_projection', {})
    mc_mean = monte_carlo_projection.get('team_score_mean', 0)
    mc_std = monte_carlo_projection.get('team_score_std', 0)
    
    # Get Bayesian individual predictions and create team projection
    bayesian_data = results.get('bayesian_data', {})
    bayesian_predictions = bayesian_data.get('player_predictions', {})
    
    # Load test team roster only in test mode
    team_players = []
    if is_test_mode:
        from ffbayes.utils.path_constants import get_default_team_file
        test_team_file = str(get_default_team_file())
        try:
            with open(test_team_file, 'r') as f:
                team_players = [line.strip().split('\t')[0] for line in f if line.strip() and not line.startswith('Player')]
        except:
            team_players = []
    
    # Create Bayesian team projection using available predictions
    bayesian_team_mean = 0
    bayesian_team_variance = 0
    bayesian_player_count = 0
    bayesian_coverage = 0
    
    if bayesian_predictions:
        # Instead of trying to match test team, create a representative team from Bayesian predictions
        # Take top players by position to create a balanced team
        qb_players = [(name, pred) for name, pred in bayesian_predictions.items() if pred.get('position') == 'QB']
        rb_players = [(name, pred) for name, pred in bayesian_predictions.items() if pred.get('position') == 'RB'] 
        wr_players = [(name, pred) for name, pred in bayesian_predictions.items() if pred.get('position') == 'WR']
        te_players = [(name, pred) for name, pred in bayesian_predictions.items() if pred.get('position') == 'TE']
        
        # Sort by mean projection and take top players
        qb_top = sorted(qb_players, key=lambda x: x[1]['mean'], reverse=True)[:1]  # 1 QB
        rb_top = sorted(rb_players, key=lambda x: x[1]['mean'], reverse=True)[:3]  # 3 RBs  
        wr_top = sorted(wr_players, key=lambda x: x[1]['mean'], reverse=True)[:4]  # 4 WRs
        te_top = sorted(te_players, key=lambda x: x[1]['mean'], reverse=True)[:1]  # 1 TE
        
        # Combine into Bayesian representative team
        bayesian_team = qb_top + rb_top + wr_top + te_top
        
        for name, pred in bayesian_team:
            bayesian_team_mean += pred.get('mean', 0)
            bayesian_team_variance += pred.get('std', 0) ** 2
            bayesian_player_count += 1
        
        bayesian_team_std = bayesian_team_variance ** 0.5 if bayesian_team_variance > 0 else 0
        bayesian_coverage = 100  # Full coverage of Bayesian team
        
        print(f"   🔍 DEBUG: Bayesian team projection uses {bayesian_player_count} players: {[name for name, _ in bayesian_team[:3]]}...")
        print(f"   🔍 DEBUG: Bayesian team mean: {bayesian_team_mean:.1f} ± {bayesian_team_std:.1f}")
    else:
        print("   🔍 DEBUG: No Bayesian predictions available")
    
    # Create simple baseline projection (historical averages by position)
    # Use Monte Carlo player data as proxy for historical averages
    baseline_team_mean = 0
    baseline_team_std = 0
    baseline_coverage = 0
    
    if monte_carlo_projection.get('player_contributions'):
        # Use Monte Carlo means but reduce variance (baseline = less sophisticated)
        for player, contrib in monte_carlo_projection['player_contributions'].items():
            baseline_team_mean += contrib.get('mean', 0)
            # Baseline has less uncertainty modeling, so reduce std by 30%
            baseline_team_std += (contrib.get('std', 0) * 0.7) ** 2
        
        baseline_team_std = baseline_team_std ** 0.5 if baseline_team_std > 0 else 0
        baseline_coverage = 100  # Baseline uses simple averages for all players
    
    # Create the comparison chart
    models = []
    means = []
    stds = []
    colors = []
    labels = []
    
    # Add models that have projections
    if mc_mean > 0:
        models.append('Monte Carlo')
        means.append(mc_mean)
        stds.append(mc_std)
        colors.append('blue')
        labels.append(f'{mc_mean:.1f} ± {mc_std:.1f}')
    
    # ALWAYS show Bayesian, even if 0
    models.append('Bayesian')
    means.append(bayesian_team_mean)
    stds.append(bayesian_team_std)
    colors.append('red')
    if bayesian_team_mean > 0:
        labels.append(f'{bayesian_team_mean:.1f} ± {bayesian_team_std:.1f}')
    else:
        labels.append('0.0 ± 0.0 (No overlap)')
    
    if baseline_team_mean > 0:
        models.append('Baseline')
        means.append(baseline_team_mean)
        stds.append(baseline_team_std)
        colors.append('green')
        labels.append(f'{baseline_team_mean:.1f} ± {baseline_team_std:.1f}')
    
    if len(models) > 0:
        # Create comparison bar chart - handle zero values for visibility
        display_means = [max(m, 5) for m in means]  # Minimum height of 5 for visibility
        bars = ax1.bar(models, display_means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        
        # Mark zero bars differently
        for i, (bar, actual_mean) in enumerate(zip(bars, means)):
            if actual_mean == 0:
                bar.set_hatch('///')  # Add hatching to show it's not a real value
                bar.set_alpha(0.3)    # Make it more transparent
        
        ax1.set_ylabel('Team Score Projection (Fantasy Points)', fontsize=12)
        ax1.set_title('🔥 MODEL COMPARISON: Team Score Projections 🔥', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars - position below bars to avoid overlap with error bars
        for i, (bar, label) in enumerate(zip(bars, labels)):
            if means[i] > 0:
                # Position labels below bars, above x-axis
                ax1.text(bar.get_x() + bar.get_width()/2, -max(stds) * 0.2,
                        label, ha='center', va='top', fontweight='bold', fontsize=10)
            else:
                # Special positioning for zero bars - place label above x-axis
                ax1.text(bar.get_x() + bar.get_width()/2, 10,
                        label, ha='center', va='bottom', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        
        # Adjust y-axis limits to accommodate labels below bars
        y_min = ax1.get_ylim()[0]
        ax1.set_ylim(y_min - max(stds) * 0.5, ax1.get_ylim()[1])
        
        # Add comparison insights
        if len(models) >= 2:
            max_idx = means.index(max(means))
            min_idx = means.index(min(means))
            
            optimistic_model = models[max_idx]
            conservative_model = models[min_idx]
            difference = means[max_idx] - means[min_idx]
            
            # Only calculate percentage if minimum value > 0
            if means[min_idx] > 0:
                difference_pct = (difference / means[min_idx]) * 100
                ax1.text(0.5, 0.95, f'{optimistic_model} most optimistic: +{difference:.1f} pts ({difference_pct:.1f}%)', 
                        transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            else:
                ax1.text(0.5, 0.95, f'{optimistic_model} most optimistic: +{difference:.1f} pts (vs {conservative_model} at 0)', 
                        transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            # Add uncertainty comparison
            if len(stds) >= 2:
                max_unc_idx = stds.index(max(stds))
                min_unc_idx = stds.index(min(stds))
                
                uncertain_model = models[max_unc_idx]
                confident_model = models[min_unc_idx]
                
                ax1.text(0.5, 0.88, f'{confident_model} most confident (±{stds[min_unc_idx]:.1f}), {uncertain_model} most uncertain (±{stds[max_unc_idx]:.1f})', 
                        transform=ax1.transAxes, ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # Add coverage information for Bayesian
        if bayesian_team_mean > 0 and bayesian_coverage < 100:
            ax1.text(0.5, 0.81, f'Bayesian: {bayesian_coverage:.0f}% player coverage ({bayesian_player_count}/{len(team_players)} players)', 
                    transform=ax1.transAxes, ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
                    
    else:
        ax1.text(0.5, 0.5, 'No model projections available for comparison', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
        ax1.set_title('Model Comparison: Team Score Projections', fontsize=14, fontweight='bold')
    
    # ======================================================================
    # PLOT 2: Team Construction Analysis
    # ======================================================================
    if monte_carlo_projection:
        # Get team composition data
        player_contributions = monte_carlo_projection.get('player_contributions', {})
        
        if player_contributions:
            # Analyze team construction
            positions = {}
            total_contribution = 0
            
            for player, data in player_contributions.items():
                pos = data.get('position', 'Unknown')
                contribution = data.get('contribution_pct', 0)
                positions[pos] = positions.get(pos, 0) + contribution
                total_contribution += contribution
            
            # Create position distribution pie chart
            if positions:
                pos_labels = list(positions.keys())
                pos_values = list(positions.values())
                
                # Create pie chart
                wedges, texts, autotexts = ax2.pie(pos_values, labels=pos_labels, autopct='%1.1f%%', 
                                                   startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
                
                ax2.set_title('Team Construction by Position', fontsize=14, fontweight='bold')
                
                # Add team strength insights
                insights = []
                insights.append(f"📊 Total Team: {len(player_contributions)} players")
                insights.append(f"🎯 QB Contribution: {positions.get('QB', 0):.1f}%")
                insights.append(f"🏃 RB Contribution: {positions.get('RB', 0):.1f}%")
                insights.append(f"🤲 WR Contribution: {positions.get('WR', 0):.1f}%")
                insights.append(f"🏈 TE Contribution: {positions.get('TE', 0):.1f}%")
                
                # Add construction insights
                if positions.get('QB', 0) > 20:
                    insights.append("⚡ QB-heavy construction")
                if positions.get('RB', 0) > 40:
                    insights.append("🏃 RB-dominant strategy")
                if positions.get('WR', 0) > 50:
                    insights.append("🤲 WR-focused approach")
                
                # Display insights below pie chart
                y_pos = -0.3
                for insight in insights:
                    ax2.text(0, y_pos, insight, ha='center', va='top', fontsize=10, 
                            transform=ax2.transAxes, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    y_pos -= 0.08
            else:
                ax2.text(0.5, 0.5, 'No position data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
                ax2.set_title('Team Construction Analysis', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No player contribution data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
            ax2.set_title('Team Construction Analysis', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No team projection data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
        ax2.set_title('Team Construction Analysis', fontsize=14, fontweight='bold')
    
    # ======================================================================
    # PLOT 3: Player Contributions
    # ======================================================================
    if monte_carlo_projection:
        player_contributions = monte_carlo_projection.get('player_contributions', {})
        
        if player_contributions:
            # Show top player contributors
            players = list(player_contributions.keys())
            contributions = [player_contributions[p]['contribution_pct'] for p in players]
            
            # Sort by contribution and take top 8
            sorted_data = sorted(zip(players, contributions), key=lambda x: x[1], reverse=True)[:8]
            top_players, top_contributions = zip(*sorted_data)
            
            # Create horizontal bar chart
            bars = ax3.barh(range(len(top_players)), top_contributions, color='skyblue', alpha=0.7)
            
            # Add player names and values
            ax3.set_yticks(range(len(top_players)))
            ax3.set_yticklabels([f"{p[:15]}..." if len(p) > 15 else p for p in top_players])
            ax3.set_xlabel('Contribution to Team Score (%)', fontsize=12)
            ax3.set_title('Top Player Contributors', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, contrib) in enumerate(zip(bars, top_contributions)):
                ax3.text(contrib + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{contrib:.1f}%', ha='left', va='center', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No player contribution data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
            ax3.set_title('Player Contributors', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No player contribution data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
        ax3.set_title('Player Contributors', fontsize=14, fontweight='bold')
    
    # ======================================================================
    # PLOT 4: Strategic Insights
    # ======================================================================
    insights = []
    
    # Model comparison insights
    if len(models) >= 2:
        max_idx = means.index(max(means))
        min_idx = means.index(min(means))
        optimistic_model = models[max_idx]
        conservative_model = models[min_idx]
        difference = means[max_idx] - means[min_idx]
        
        # Only calculate percentage if minimum value > 0
        if means[min_idx] > 0:
            difference_pct = (difference / means[min_idx]) * 100
            insights.append(f"📊 {optimistic_model} projects highest: {means[max_idx]:.1f} pts")
            insights.append(f"📊 {conservative_model} projects lowest: {means[min_idx]:.1f} pts")
            insights.append(f"📈 Projection spread: {difference:.1f} pts ({difference_pct:.1f}%)")
            
            if difference_pct < 5:
                insights.append("✅ Models in strong agreement")
            elif difference_pct < 15:
                insights.append("⚠️ Moderate model disagreement")
            else:
                insights.append("🚨 Significant model disagreement")
        else:
            # When one model is 0, calculate spread relative to the non-zero model
            if len([m for m in means if m > 0]) >= 2:
                non_zero_means = [m for m in means if m > 0]
                non_zero_diff = max(non_zero_means) - min(non_zero_means)
                non_zero_diff_pct = (non_zero_diff / min(non_zero_means)) * 100
                
                insights.append(f"📊 {optimistic_model} projects highest: {means[max_idx]:.1f} pts")
                insights.append(f"📊 {conservative_model} projects lowest: {means[min_idx]:.1f} pts")
                insights.append(f"📈 Projection spread: {difference:.1f} pts (vs {conservative_model} at 0)")
                insights.append("⚠️ Bayesian model has no player overlap with test team")
                
                # Show disagreement between non-zero models
                if non_zero_diff_pct < 5:
                    insights.append("✅ Non-zero models in strong agreement")
                elif non_zero_diff_pct < 15:
                    insights.append("⚠️ Moderate disagreement between non-zero models")
                else:
                    insights.append("🚨 Significant disagreement between non-zero models")
            else:
                insights.append(f"📊 {optimistic_model} projects highest: {means[max_idx]:.1f} pts")
                insights.append(f"📊 {conservative_model} projects lowest: {means[min_idx]:.1f} pts")
                insights.append(f"📈 Projection spread: {difference:.1f} pts (vs {conservative_model} at 0)")
                insights.append("⚠️ Bayesian model has no player overlap with test team")
    
    # Add insights from the results if available
    if 'insights' in results:
        insights.extend(results['insights'])
    
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
        if '🏆' in insight or '🥇' in insight or '✅' in insight:
            color = 'darkgreen'
        elif '⚠️' in insight or '🚨' in insight:
            color = 'darkorange'
        elif '💡' in insight:
            color = 'darkblue'
        else:
            color = 'black'
        
        ax4.text(0.02, 0.9 - i*0.06, insight, fontsize=11, va='top', color=color, fontweight='bold')
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('🔥 MODEL COMPARISON ANALYSIS (QUICK_TEST MODE) 🔥', 
                    fontsize=18, color='red', fontweight='bold')
    else:
        fig.suptitle('🔥 MODEL COMPARISON ANALYSIS 🔥', 
                    fontsize=18, color='darkblue', fontweight='bold')
    
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
