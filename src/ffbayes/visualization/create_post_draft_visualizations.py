#!/usr/bin/env python3
"""
Create post-draft visualizations for fantasy football team analysis.
Generates charts that help analyze your drafted team's strengths and weaknesses.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Import centralized position colors
from ffbayes.visualization.position_colors import \
    POSITION_COLORS as POS_PALETTE


def get_output_directory():
    """Determine output directory for post-draft visualizations."""
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_post_draft_plots_dir
    base_dir = str(get_post_draft_plots_dir(current_year))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def load_draft_team():
    """Load the user's drafted team."""
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_teams_dir
    team_file = str(get_teams_dir() / f"drafted_team_{current_year}.tsv")
    
    if not os.path.exists(team_file):
        raise FileNotFoundError(f"Draft team file not found: {team_file}")
    
    print(f"🏈 Loading draft team from: {team_file}")
    team_df = pd.read_csv(team_file, sep='\t')
    
    # Support user's format (POS, PLAYER, BYE) and standard format (Name, Position, Team)
    if 'Name' not in team_df.columns and 'PLAYER' in team_df.columns:
        team_df = team_df.rename(columns={'PLAYER': 'Name'})
    if 'Position' not in team_df.columns and 'POS' in team_df.columns:
        team_df = team_df.rename(columns={'POS': 'Position'})
    
    # Infer Team if missing using unified dataset lookup
    if 'Team' not in team_df.columns:
        print("🔎 Inferring Team column from unified dataset...")
        try:
            from ffbayes.data_pipeline.unified_data_loader import \
                load_unified_dataset
            unified_df = load_unified_dataset('datasets')
            # Use the most recent team per player
            latest_team = (unified_df.sort_values(['Name', 'Season'])
                                     .groupby('Name')
                                     .tail(1)[['Name', 'Tm']]
                                     .rename(columns={'Tm': 'Team'}))
            team_df = team_df.merge(latest_team, on='Name', how='left')
        except Exception as e:
            print(f"⚠️  Could not infer Team from unified dataset: {e}")
            team_df['Team'] = None
    
    # Validate required columns
    required_columns = ['Name', 'Position', 'Team']
    missing_columns = [col for col in required_columns if col not in team_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return team_df


def load_team_analysis_results():
    """Load team aggregation and analysis results."""
    current_year = datetime.now().year
    
    from ffbayes.utils.path_constants import get_team_aggregation_dir

    # Team aggregation results
    team_agg_file = str(get_team_aggregation_dir(current_year) / "team_analysis_results.json")
    if not os.path.exists(team_agg_file):
        raise FileNotFoundError(f"Team analysis results not found: {team_agg_file}")
    
    print(f"📊 Loading team analysis from: {team_agg_file}")
    with open(team_agg_file, 'r') as f:
        return json.load(f)


def load_monte_carlo_results():
    """Load Monte Carlo validation results."""
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_monte_carlo_dir
    mc_json = str(get_monte_carlo_dir(current_year) / "monte_carlo_validation.json")
    
    if os.path.exists(mc_json):
        print(f"🎲 Loading Monte Carlo results from: {mc_json}")
        with open(mc_json, 'r') as f:
            return json.load(f)
    
    # Fallback: find latest TSV and compute a comprehensive summary
    mc_dir = str(get_monte_carlo_dir(current_year))
    import glob
    tsvs = glob.glob(os.path.join(mc_dir, "*.tsv"))
    if not tsvs:
        print("⚠️  No Monte Carlo results found - skipping validation analysis")
        return None
    latest = max(tsvs, key=os.path.getmtime)
    print(f"🎲 Loading Monte Carlo results from TSV: {latest}")
    import pandas as pd
    df = pd.read_csv(latest, sep='\t', index_col=None)
    
    # Create comprehensive summary
    summary = {}
    if 'Total' in df.columns:
        total_scores = df['Total']
        summary['team_performance'] = {
            'mean': float(total_scores.mean()),
            'std': float(total_scores.std()),
            'min': float(total_scores.min()),
            'max': float(total_scores.max()),
            'simulation_count': len(total_scores),
            'percentiles': {
                '10th': float(total_scores.quantile(0.1)),
                '25th': float(total_scores.quantile(0.25)),
                '50th': float(total_scores.quantile(0.5)),
                '75th': float(total_scores.quantile(0.75)),
                '90th': float(total_scores.quantile(0.9))
            }
        }
        
        # Add player-level analysis
        player_cols = [col for col in df.columns if col != 'Total']
        if player_cols:
            summary['player_analysis'] = {}
            for player in player_cols:
                player_scores = df[player]
                mean_val = float(player_scores.mean())
                
                summary['player_analysis'][player] = {
                    'mean': mean_val,
                    'std': float(player_scores.std()),
                    'min': float(player_scores.min()),
                    'max': float(player_scores.max()),
                    'consistency': float(1 - (player_scores.std() / player_scores.mean())) if player_scores.mean() > 0 else 0
                }
    
    return summary


def create_team_composition_chart(team_df, output_dir):
    """Create team composition analysis chart."""
    print("🏈 Creating team composition chart...")
    
    # Position distribution
    position_counts = team_df['Position'].value_counts()
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    
    colors = [POS_PALETTE.get(pos, '#666666') for pos in position_counts.index]
    wedges, texts, autotexts = plt.pie(position_counts.values, 
                                       labels=position_counts.index,
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90)
    
    plt.title('Team Composition by Position', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    # Add meaningful insights instead of overlapping text
    total_players = len(team_df)
    insights = []
    insights.append(f"Total Players: {total_players}")
    
    # Add position insights
    if 'WR' in position_counts:
        insights.append(f"WR Depth: {position_counts['WR']} players")
    if 'RB' in position_counts:
        insights.append(f"RB Depth: {position_counts['RB']} players")
    if 'QB' in position_counts:
        insights.append(f"QB: {position_counts['QB']} starter")
    if 'TE' in position_counts:
        insights.append(f"TE: {position_counts['TE']} starter")
    
        # Add team assessment
    if position_counts.get('WR', 0) >= 5:
        insights.append("Strong WR depth")
    elif position_counts.get('WR', 0) <= 3:
        insights.append("Weak WR depth")
    
    if position_counts.get('RB', 0) >= 3:
        insights.append("Good RB depth")
    elif position_counts.get('RB', 0) <= 2:
        insights.append("Thin RB depth")

    # Display insights in a readable format
    insight_text = '\n'.join(insights)
    plt.text(0.02, 0.98, insight_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save plot
    plot_file = os.path.join(output_dir, 'team_composition.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_team_strength_analysis(team_analysis, output_dir):
    """Create team strength analysis chart with actionable insights."""
    print("💪 Creating team strength analysis...")
    
    # Extract team strength metrics
    if 'team_strength' not in team_analysis:
        print("   ⚠️  No team strength data available")
        return None
    
    strength_data = team_analysis['team_strength']
    
    # Create comprehensive strength analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Position Strength Overview
    categories = list(strength_data.keys())
    values = list(strength_data.values())
    
    # Color code by strength level
    colors = []
    for value in values:
        if value >= 15:  # Elite
            colors.append('#2E8B57')  # Sea Green
        elif value >= 10:  # Good
            colors.append('#4682B4')  # Steel Blue
        elif value >= 5:  # Average
            colors.append('#FFA500')  # Orange
        else:  # Weak
            colors.append('#DC143C')  # Crimson
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Fantasy Points')
    ax1.set_title('Position Strength Analysis', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels and strength assessments
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add strength assessment
        if value >= 15:
            strength = "ELITE"
        elif value >= 10:
            strength = "GOOD"
        elif value >= 5:
            strength = "AVG"
        else:
            strength = "WEAK"
        
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                strength, ha='center', va='center', fontweight='bold', 
                color='white', fontsize=10)
    
    # Plot 2: Team Strengths vs Weaknesses
    strengths = []
    weaknesses = []
    for pos, value in strength_data.items():
        if value >= 10:
            strengths.append((pos, value))
        else:
            weaknesses.append((pos, value))
    
    # Create comparison
    if strengths and weaknesses:
        strong_pos = [s[0] for s in strengths]
        strong_vals = [s[1] for s in strengths]
        weak_pos = [w[0] for w in weaknesses]
        weak_vals = [w[1] for w in weaknesses]
        
        x = range(len(strong_pos) + len(weak_pos))
        all_pos = strong_pos + weak_pos
        all_vals = strong_vals + weak_vals
        all_colors = ['#2E8B57'] * len(strong_pos) + ['#DC143C'] * len(weak_pos)
        
        bars = ax2.bar(x, all_vals, color=all_colors, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_pos, rotation=45, ha='right')
        ax2.set_ylabel('Average Fantasy Points')
        ax2.set_title('Strengths vs Weaknesses', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2E8B57', label='Strength (>=10 pts)'),
                          Patch(facecolor='#DC143C', label='Weakness (<10 pts)')]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    # Plot 3: Team Composition Analysis
    if 'player_projections' in team_analysis:
        projections = team_analysis['player_projections']
        
        # Group by position
        pos_data = {}
        for player, data in projections.items():
            if 'position' in data:
                pos = data['position']
                if pos not in pos_data:
                    pos_data[pos] = []
                pos_data[pos].append(data['mean'])
        
        # Create box plot
        if pos_data:
            pos_labels = list(pos_data.keys())
            pos_values = list(pos_data.values())
            
            bp = ax3.boxplot(pos_values, labels=pos_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax3.set_ylabel('Fantasy Points')
            ax3.set_title('Player Distribution by Position', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Actionable Insights
    insights = []
    
    # Analyze strengths
    strong_positions = [pos for pos, val in strength_data.items() if val >= 10]
    if strong_positions:
        insights.append(f"✅ STRENGTHS: {', '.join(strong_positions)}")
    
    # Analyze weaknesses
    weak_positions = [pos for pos, val in strength_data.items() if val < 8]
    if weak_positions:
        insights.append(f"⚠️  WEAKNESSES: {', '.join(weak_positions)}")
    
    # Overall assessment
    avg_strength = sum(strength_data.values()) / len(strength_data)
    if avg_strength >= 12:
        overall = "STRONG TEAM"
        color = '#2E8B57'
    elif avg_strength >= 8:
        overall = "AVERAGE TEAM"
        color = '#FFA500'
    else:
        overall = "NEEDS IMPROVEMENT"
        color = '#DC143C'
    
    insights.append(f"📊 OVERALL: {overall} ({avg_strength:.1f} avg)")
    
    # Add insights text
    ax4.text(0.05, 0.95, '\n'.join(insights), transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax4.set_title('Team Assessment & Recommendations', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'team_strength_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_player_performance_projections(team_df, team_analysis, output_dir):
    """Create individual player performance projections with actionable insights."""
    print("📈 Creating player performance projections...")
    
    # Extract player projections
    if 'player_projections' not in team_analysis:
        print("   ⚠️  No player projection data available")
        return None
    
    projections = team_analysis['player_projections']
    
    # Create dataframe for plotting
    plot_data = []
    for player_name, player_data in projections.items():
        if 'mean' in player_data and 'std' in player_data:
            # Get position from team_df
            player_row = team_df[team_df['Name'] == player_name]
            position = player_row['Position'].iloc[0] if not player_row.empty else 'Unknown'
            
            # Calculate risk score
            risk_score = player_data['std'] / player_data['mean'] if player_data['mean'] > 0 else 0
            
            plot_data.append({
                'Name': player_name,
                'Mean': player_data['mean'],
                'Std': player_data['std'],
                'Position': position,
                'Risk': risk_score,
                'Team': player_row['Team'].iloc[0] if not player_row.empty else 'Unknown'
            })
    
    if not plot_data:
        print("   ⚠️  No projection data available")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create comprehensive projection analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Player Projections by Position
    # Sort by mean projection
    plot_df_sorted = plot_df.sort_values('Mean', ascending=True)
    
    # Color by position
    colors = [POS_PALETTE.get(pos, '#666666') for pos in plot_df_sorted['Position']]
    
    # Create horizontal bar chart
    bars = ax1.barh(range(len(plot_df_sorted)), plot_df_sorted['Mean'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    # Add error bars
    for i, (_, row) in enumerate(plot_df_sorted.iterrows()):
        ax1.errorbar(row['Mean'], i, xerr=row['Std'], fmt='none', 
                     color='black', capsize=3, capthick=1)
    
    # Add player names
    for i, (_, row) in enumerate(plot_df_sorted.iterrows()):
        ax1.text(row['Mean'] + row['Std'] + 1, i, f"{row['Name']} ({row['Position']})", 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Projected Fantasy Points')
    ax1.set_ylabel('Players')
    ax1.set_title('Player Performance Projections (Sorted by Expected Points)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add position legend
    unique_positions = plot_df_sorted['Position'].unique()
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor=POS_PALETTE.get(pos, '#666666'), 
                                markersize=10, label=pos)
                      for pos in unique_positions]
    ax1.legend(handles=legend_elements, title='Position', loc='lower right')
    
    # Plot 2: Risk vs Reward Analysis
    # Color code by risk level
    risk_colors = []
    for risk in plot_df['Risk']:
        if risk <= 0.3:  # Low risk
            risk_colors.append('#2E8B57')  # Green
        elif risk <= 0.6:  # Medium risk
            risk_colors.append('#FFA500')  # Orange
        else:  # High risk
            risk_colors.append('#DC143C')  # Red
    
    scatter = ax2.scatter(plot_df['Mean'], plot_df['Risk'], 
                         c=risk_colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add player labels
    for _, row in plot_df.iterrows():
        ax2.annotate(row['Name'], (row['Mean'], row['Risk']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Expected Fantasy Points')
    ax2.set_ylabel('Risk Score (Std Dev / Mean)')
    ax2.set_title('Risk vs Reward Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add risk zones
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Low Risk')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium Risk')
    ax2.legend()
    
    # Plot 3: Position Breakdown
    pos_means = plot_df.groupby('Position')['Mean'].mean().sort_values(ascending=True)
    pos_counts = plot_df.groupby('Position').size()
    
    bars = ax3.barh(pos_means.index, pos_means.values, 
                   color=[POS_PALETTE.get(pos, '#666666') for pos in pos_means.index],
                   alpha=0.8, edgecolor='black')
    
    # Add count labels
    for i, (pos, mean_val) in enumerate(pos_means.items()):
        count = pos_counts[pos]
        ax3.text(mean_val + 0.5, i, f'n={count}', va='center', fontweight='bold')
    
    ax3.set_xlabel('Average Fantasy Points')
    ax3.set_title('Position Group Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Key Insights
    insights = []
    
    # Top performers
    top_3 = plot_df.nlargest(3, 'Mean')
    insights.append("TOP PERFORMERS:")
    for _, player in top_3.iterrows():
        insights.append(f"  • {player['Name']}: {player['Mean']:.1f} ± {player['Std']:.1f} pts")
    
    # High risk players
    high_risk = plot_df[plot_df['Risk'] > 0.6]
    if not high_risk.empty:
        insights.append("\n⚠️  HIGH RISK PLAYERS:")
        for _, player in high_risk.iterrows():
            insights.append(f"  • {player['Name']}: Risk = {player['Risk']:.2f}")
    
    # Position insights
    best_pos = pos_means.idxmax()
    worst_pos = pos_means.idxmin()
    insights.append("\n📊 POSITION INSIGHTS:")
    insights.append(f"  • Strongest: {best_pos} ({pos_means[best_pos]:.1f} avg)")
    insights.append(f"  • Weakest: {worst_pos} ({pos_means[worst_pos]:.1f} avg)")
    
    # Team assessment
    total_projection = plot_df['Mean'].sum()
    insights.append(f"\n🎯 TEAM TOTAL: {total_projection:.1f} projected points")
    
    # Add insights text
    ax4.text(0.05, 0.95, '\n'.join(insights), transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax4.set_title('Player Analysis Insights', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'player_performance_projections.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_monte_carlo_validation_chart(mc_results, output_dir):
    """Create Monte Carlo validation analysis chart with actionable insights."""
    if not mc_results:
        print("   ⚠️  Skipping Monte Carlo analysis - no data")
        return None
    
    print("🎲 Creating Monte Carlo validation chart...")
    
    # Extract validation metrics
    if 'team_performance' not in mc_results:
        print("   ⚠️  No team performance data available")
        return None
    
    performance_data = mc_results['team_performance']
    
    # Create comprehensive performance analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Team Score Distribution with Insights
    # Load the actual simulation data for histogram
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_monte_carlo_dir
    mc_dir = str(get_monte_carlo_dir(current_year))
    import glob
    tsvs = glob.glob(os.path.join(mc_dir, "*.tsv"))
    if tsvs:
        latest = max(tsvs, key=os.path.getmtime)
        df = pd.read_csv(latest, sep='\t')
        if 'Total' in df.columns:
            total_scores = df['Total']
            
            # Create histogram with percentiles
            ax1.hist(total_scores, bins=50, alpha=0.7, edgecolor='black', color='lightgreen', density=True)
            
            # Add percentile lines
            mean_score = performance_data['mean']
            std_score = performance_data['std']
            percentiles = performance_data['percentiles']
            
            ax1.axvline(mean_score, color='red', linestyle='--', linewidth=3, 
                       label=f'Mean: {mean_score:.1f} pts')
            ax1.axvline(percentiles['25th'], color='orange', linestyle=':', linewidth=2,
                       label=f'25th percentile: {percentiles["25th"]:.1f} pts')
            ax1.axvline(percentiles['75th'], color='orange', linestyle=':', linewidth=2,
                       label=f'75th percentile: {percentiles["75th"]:.1f} pts')
            
            # Add insights text
            ax1.text(0.02, 0.98, f'Team Performance Insights:\n'
                                 f'• 50% of weeks: {percentiles["25th"]:.1f}-{percentiles["75th"]:.1f} pts\n'
                                 f'• High variance: ±{std_score:.1f} pts\n'
                                 f'• Range: {performance_data["min"]:.1f}-{performance_data["max"]:.1f} pts',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_xlabel('Team Fantasy Points')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Team Performance Distribution (5000 Simulations)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance Percentile Analysis with Context
    if 'percentiles' in performance_data:
        percentiles = performance_data['percentiles']
        percentile_values = list(percentiles.values())
        percentile_labels = list(percentiles.keys())
        
        # Color code by performance level
        colors = []
        for value in percentile_values:
            if value >= 130:  # Great performance
                colors.append('#2E8B57')  # Green
            elif value >= 110:  # Good performance
                colors.append('#4682B4')  # Blue
            elif value >= 90:  # Average performance
                colors.append('#FFA500')  # Orange
            else:  # Poor performance
                colors.append('#DC143C')  # Red
        
        bars = ax2.bar(percentile_labels, percentile_values, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Fantasy Points')
        ax2.set_title('Weekly Performance Expectations\n(What to Expect Each Week)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels with context
        for i, (bar, value) in enumerate(zip(bars, percentile_values)):
            height = bar.get_height()
            
            # Add context labels
            if i == 0:  # 10th percentile
                context = "BAD WEEK\n(Likely Loss)"
            elif i == 1:  # 25th percentile
                context = "BELOW AVERAGE\n(Probably Loss)"
            elif i == 2:  # 50th percentile
                context = "AVERAGE WEEK\n(50/50 Win)"
            elif i == 3:  # 75th percentile
                context = "GOOD WEEK\n(Likely Win)"
            else:  # 90th percentile
                context = "GREAT WEEK\n(Probably Win)"
            
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:.1f} pts\n{context}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8)
        
        # Add overall insight
        mean_val = performance_data['mean']
        ax2.text(0.02, 0.98, f'Team Assessment:\n'
                              f'• Average week: {mean_val:.1f} points\n'
                              f'• High variance team\n'
                              f'• Boom/bust potential',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 3: Player Contribution Analysis
    if 'player_analysis' in mc_results:
        player_data = mc_results['player_analysis']
        # Get top 8 contributors
        player_means = {name: data['mean'] for name, data in player_data.items()}
        top_players = sorted(player_means.items(), key=lambda x: x[1], reverse=True)[:8]
        
        names = [p[0] for p in top_players]
        means = [p[1] for p in top_players]
        
        bars = ax3.barh(names, means, color='lightcoral', alpha=0.8)
        ax3.set_xlabel('Average Fantasy Points')
        ax3.set_title('Top Player Contributors', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{mean:.1f}', ha='left', va='center', fontweight='bold')
    
    # Plot 4: Risk Assessment
    if 'player_analysis' in mc_results:
        player_data = mc_results['player_analysis']
        # Calculate risk metrics
        risk_data = []
        for name, data in player_data.items():
            if data['mean'] > 0:
                risk_score = data['std'] / data['mean']  # Coefficient of variation
                risk_data.append({
                    'name': name,
                    'mean': data['mean'],
                    'risk': risk_score,
                    'consistency': data['consistency']
                })
        
        if risk_data:
            # Sort by risk (high to low)
            risk_data.sort(key=lambda x: x['risk'], reverse=True)
            top_risky = risk_data[:6]
            
            names = [p['name'] for p in top_risky]
            risks = [p['risk'] for p in top_risky]
            
            bars = ax4.barh(names, risks, color='gold', alpha=0.8)
            ax4.set_xlabel('Risk Score (Std Dev / Mean)')
            ax4.set_title('Player Risk Assessment (High Variance)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add risk labels
            for bar, risk in zip(bars, risks):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{risk:.2f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'monte_carlo_validation.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_team_summary_dashboard(team_df, team_analysis, mc_results, output_dir):
    """Create comprehensive team summary dashboard."""
    print("📋 Creating team summary dashboard...")
    
    # Create summary statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Position Distribution
    position_counts = team_df['Position'].value_counts()
    colors = [POS_PALETTE.get(pos, '#666666') for pos in position_counts.index]
    
    bars1 = ax1.bar(position_counts.index, position_counts.values, color=colors, alpha=0.8)
    ax1.set_ylabel('Number of Players')
    ax1.set_title('Team Position Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, position_counts.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Team Strength Overview
    if 'team_strength' in team_analysis:
        strength_data = team_analysis['team_strength']
        categories = list(strength_data.keys())
        values = list(strength_data.values())
        
        bars2 = ax2.bar(categories, values, alpha=0.8, color='lightblue')
        ax2.set_ylabel('Strength Score')
        ax2.set_title('Team Strength Overview')
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Player Count by Team
    team_counts = team_df['Team'].value_counts().head(10)
    bars3 = ax3.bar(range(len(team_counts)), team_counts.values, alpha=0.8, color='lightcoral')
    ax3.set_xlabel('NFL Team')
    ax3.set_ylabel('Number of Players')
    ax3.set_title('Player Count by NFL Team (Top 10)')
    ax3.set_xticks(range(len(team_counts)))
    ax3.set_xticklabels(team_counts.index, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Monte Carlo Results Summary
    if mc_results and 'team_performance' in mc_results:
        perf_data = mc_results['team_performance']
        mean_val = perf_data.get('mean_score', perf_data.get('mean', 'N/A'))
        std_val = perf_data.get('std_score', perf_data.get('std', 'N/A'))
        mean_txt = f"{float(mean_val):.1f}" if isinstance(mean_val, (int, float)) else str(mean_val)
        std_txt = f"{float(std_val):.1f}" if isinstance(std_val, (int, float)) else str(std_val)
        ax4.text(0.1, 0.8, f"Mean Score: {mean_txt}", 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.6, f"Std Dev: {std_txt}", 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.4, f"Simulations: {perf_data.get('simulation_count', 'N/A')}", 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.set_title('Monte Carlo Validation Summary')
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, 'No Monte Carlo data available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=16)
        ax4.set_title('Monte Carlo Validation Summary')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'team_summary_dashboard.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def main():
    """Main function to create all post-draft visualizations."""
    print("=" * 70)
    print("Creating Post-Draft Fantasy Football Visualizations")
    print("=" * 70)
    
    try:
        # Get output directory
        output_dir = get_output_directory()
        print(f"📁 Output directory: {output_dir}")
        
        # Load data
        team_df = load_draft_team()
        team_analysis = load_team_analysis_results()
        mc_results = load_monte_carlo_results()
        
        print(f"🏈 Loaded draft team: {len(team_df)} players")
        print(f"📊 Loaded team analysis: {len(team_analysis)} metrics")
        if mc_results:
            print(f"🎲 Loaded Monte Carlo data: {len(mc_results)} results")
        
        # Create visualizations
        plot_files = []
        
        # 1. Team composition
        plot_files.append(create_team_composition_chart(team_df, output_dir))
        
        # 2. Team strength analysis
        strength_plot = create_team_strength_analysis(team_analysis, output_dir)
        if strength_plot:
            plot_files.append(strength_plot)
        
        # 3. Player projections
        projections_plot = create_player_performance_projections(team_df, team_analysis, output_dir)
        if projections_plot:
            plot_files.append(projections_plot)
        
        # 4. Monte Carlo validation
        mc_plot = create_monte_carlo_validation_chart(mc_results, output_dir)
        if mc_plot:
            plot_files.append(mc_plot)
        
        # 5. Comprehensive dashboard
        plot_files.append(create_team_summary_dashboard(team_df, team_analysis, mc_results, output_dir))
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 Post-Draft Visualizations Complete!")
        print("=" * 70)
        print(f"✅ Generated {len(plot_files)} visualization files:")
        for plot_file in plot_files:
            if plot_file:
                print(f"   📊 {os.path.basename(plot_file)}")
        
        print(f"\n📁 All files saved to: {output_dir}")
        print("🏈 Use these visualizations to analyze your drafted team!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
