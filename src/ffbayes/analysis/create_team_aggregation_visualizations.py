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
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

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
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    players = list(player_contribs.keys())
    means = [player_contribs[p]['mean'] for p in players]
    stds = [player_contribs[p]['std'] for p in players]
    contribs = [player_contribs[p]['contribution_pct'] for p in players]
    
    # Plot 1: Absolute Fantasy Points with Error Bars
    y_pos = np.arange(len(players))
    bars1 = ax1.barh(y_pos, means, xerr=stds, capsize=5, alpha=0.7, color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(players)
    ax1.set_xlabel('Fantasy Points')
    ax1.set_title('Individual Player Fantasy Points')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, means, stds)):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{mean:.1f}±{std:.1f}', va='center', fontsize=9)
    
    # Plot 2: Contribution Percentages
    bars2 = ax2.barh(y_pos, contribs, alpha=0.7, color='lightcoral')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(players)
    ax2.set_xlabel('Contribution to Team Total (%)')
    ax2.set_title('Player Contribution Percentages')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, contrib) in enumerate(zip(bars2, contribs)):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{contrib:.1f}%', va='center', fontsize=9)
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Team Score Breakdown (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
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
    
    # Group players by position
    position_data = {}
    for player, data in player_contribs.items():
        # Extract position from player name (assuming format: "Player Name (POS)")
        if '(' in player and ')' in player:
            pos = player.split('(')[-1].split(')')[0].strip()
        else:
            # Default to 'UNK' if position not found
            pos = 'UNK'
        
        if pos not in position_data:
            position_data[pos] = {'players': [], 'points': [], 'contribs': []}
        
        position_data[pos]['players'].append(player)
        position_data[pos]['points'].append(data['mean'])
        position_data[pos]['contribs'].append(data['contribution_pct'])
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    positions = list(position_data.keys())
    
    # Plot 1: Total Points by Position
    total_points = [sum(position_data[pos]['points']) for pos in positions]
    bars1 = ax1.bar(positions, total_points, alpha=0.7, color='lightblue')
    ax1.set_title('Total Fantasy Points by Position')
    ax1.set_ylabel('Total Points')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, total in zip(bars1, total_points):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{total:.1f}', ha='center', va='bottom')
    
    # Plot 2: Average Points per Player by Position
    avg_points = [np.mean(position_data[pos]['points']) for pos in positions]
    bars2 = ax2.bar(positions, avg_points, alpha=0.7, color='lightgreen')
    ax2.set_title('Average Fantasy Points per Player by Position')
    ax2.set_ylabel('Average Points')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, avg in zip(bars2, avg_points):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{avg:.1f}', ha='center', va='bottom')
    
    # Plot 3: Contribution Percentages by Position
    total_contribs = [sum(position_data[pos]['contribs']) for pos in positions]
    bars3 = ax3.bar(positions, total_contribs, alpha=0.7, color='lightcoral')
    ax3.set_title('Total Contribution Percentage by Position')
    ax3.set_ylabel('Contribution (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, contrib in zip(bars3, total_contribs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{contrib:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Player Count by Position
    player_counts = [len(position_data[pos]['players']) for pos in positions]
    bars4 = ax4.bar(positions, player_counts, alpha=0.7, color='gold')
    ax4.set_title('Number of Players by Position')
    ax4.set_ylabel('Player Count')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars4, player_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count}', ha='center', va='bottom')
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Position Analysis (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
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
    
    # Calculate coefficient of variation (uncertainty metric)
    cv_values = [std/mean if mean > 0 else 0 for mean, std in zip(means, stds)]
    
    # Plot 1: Uncertainty vs Team Contribution
    scatter1 = ax1.scatter(contribs, cv_values, c=means, cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Contribution to Team Total (%)')
    ax1.set_ylabel('Coefficient of Variation (Uncertainty)')
    ax1.set_title('Player Uncertainty vs Team Contribution')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Fantasy Points')
    
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
    scatter3 = ax3.scatter(means, stds, c=contribs, cmap='plasma', s=100, alpha=0.7)
    ax3.set_xlabel('Mean Fantasy Points')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Player Mean vs Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Contribution (%)')
    
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
    if 'model_comparison' in results:
        models = list(results['model_comparison'].keys())
        mae_values = [results['model_comparison'][m].get('mae', 0) for m in models]
        
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
    
    # Plot 2: Team Statistics Summary
    if 'team_projection' in results:
        team_stats = results['team_projection']['total_score']
        stats_labels = ['Mean', 'Std Dev', 'Min', 'Max']
        stats_values = [team_stats['mean'], team_stats['std'], 
                       team_stats.get('min', team_stats['mean']), 
                       team_stats.get('max', team_stats['mean'])]
        
        bars2 = ax2.bar(stats_labels, stats_values, alpha=0.7, color='lightgreen')
        ax2.set_title('Team Score Statistics')
        ax2.set_ylabel('Fantasy Points')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, stats_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No team projection data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Team Score Statistics')
    
    # Plot 3: Roster Coverage Analysis
    if 'roster_analysis' in results:
        coverage = results['roster_analysis'].get('roster_coverage_percentage', 0)
        missing_players = results['roster_analysis'].get('missing_players_count', 0)
        total_players = results['roster_analysis'].get('total_roster_size', 0)
        
        # Create pie chart for coverage
        labels = ['Covered Players', 'Missing Players']
        sizes = [coverage, 100 - coverage]
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Roster Coverage Analysis\n({total_players - missing_players}/{total_players} players)')
    else:
        ax3.text(0.5, 0.5, 'No roster analysis data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Roster Coverage Analysis')
    
    # Plot 4: Key Insights and Recommendations
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
    
    # Add default insights if none available
    if not insights:
        insights = [
            "📊 Team aggregation completed successfully",
            "📈 Individual player predictions aggregated to team level",
            "🎯 Uncertainty properly propagated from individual to team",
            "📋 Results ready for draft strategy analysis"
        ]
    
    # Create text box with insights
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
    
    # Add test mode indicator if applicable
    if is_test_mode:
        fig.suptitle('Comparison Insights (QUICK_TEST MODE - Deterministic Results)', 
                    fontsize=16, color='red', fontweight='bold')
    
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
