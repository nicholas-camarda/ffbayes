#!/usr/bin/env python3
"""
Create comprehensive pre-draft visualizations for fantasy football draft strategy.
Generates charts that help with draft decision-making before the draft occurs.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import centralized position colors
from ffbayes.visualization.position_colors import POSITION_COLORS as POS_PALETTE

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")




def get_output_directory():
    """Determine output directory for pre-draft visualizations."""
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_pre_draft_plots_dir
    base_dir = str(get_pre_draft_plots_dir(current_year) / "visualizations")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def load_vor_data():
    """Load VOR data from the latest VOR strategy file."""
    current_year = datetime.now().year
    
    # Look for VOR strategy files
    from ffbayes.utils.vor_filename_generator import get_vor_strategy_path
    vor_files = [
        get_vor_strategy_path(current_year)
    ]
    
    for vor_file in vor_files:
        if os.path.exists(vor_file):
            print(f"📊 Loading VOR data from: {vor_file}")
            df = pd.read_excel(vor_file)
            
            # Standardize column names
            from ffbayes.utils.column_standards import standardize_columns
            df_std = standardize_columns(df, 'vor_strategy')
            
            return df_std
    
    raise FileNotFoundError("No VOR strategy file found")


def load_bayesian_strategy():
    """Load Bayesian draft strategy data."""
    from ffbayes.utils.strategy_path_generator import get_bayesian_strategy_path
    bayes_file = get_bayesian_strategy_path()
    
    if not os.path.exists(bayes_file):
        raise FileNotFoundError("No Bayesian strategy file found")
    
    print(f"🧠 Loading Bayesian strategy from: {bayes_file}")
    with open(bayes_file, 'r') as f:
        return json.load(f)


def load_hybrid_mc_results():
    """Load Hybrid MC model results for uncertainty analysis."""
    from ffbayes.utils.strategy_path_generator import get_hybrid_mc_results_path
    mc_file = get_hybrid_mc_results_path()
    
    if not os.path.exists(mc_file):
        print("⚠️  No Hybrid MC results found - skipping uncertainty analysis")
        return None
    
    print(f"🎲 Loading Hybrid MC results from: {mc_file}")
    with open(mc_file, 'r') as f:
        return json.load(f)


# REMOVED: create_vor_vs_bayesian_comparison function
# This plot was useless - just showed a diagonal line with no meaningful insights
# The comparison between VOR and Bayesian strategies should be done in the model comparison framework


def create_position_distribution_chart(vor_data, output_dir):
    """Create position distribution analysis chart."""
    print("📊 Creating position distribution chart...")
    
    # Analyze position distribution in top tiers
    top_50 = vor_data.head(50)
    top_100 = vor_data.head(100)
    top_120 = vor_data.head(120)
    
    position_counts = {}
    for tier_name, tier_data in [('Top 50', top_50), ('Top 100', top_100), ('Top 120', top_120)]:
        tier_counts = tier_data['Position'].value_counts()
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos not in position_counts:
                position_counts[pos] = {}
            position_counts[pos][tier_name] = tier_counts.get(pos, 0)
    
    # Create stacked bar chart
    positions = list(position_counts.keys())
    tiers = ['Top 50', 'Top 100', 'Top 120']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = range(len(positions))
    width = 0.25
    
    for i, tier in enumerate(tiers):
        values = [position_counts[pos][tier] for pos in positions]
        colors = [POS_PALETTE.get(pos, '#666666') for pos in positions]
        
        bars = ax.bar([xi + i * width for xi in x], values, width, 
                     label=tier, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Number of Players')
    ax.set_title('Position Distribution Across Draft Tiers')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(positions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = os.path.join(output_dir, 'position_distribution_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_draft_position_strategy_chart(bayes_data, output_dir):
    """Create draft position strategy visualization for position 10."""
    print("🎯 Creating draft position strategy chart...")
    
    # Extract strategy data
    strategy = bayes_data.get('strategy', {})
    
    # Create draft flow visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define draft rounds and picks for position 10
    rounds = list(range(1, 17))  # 16 rounds
    picks = [10, 11, 30, 31, 50, 51, 70, 71, 90, 91, 110, 111, 130, 131, 150, 151]
    
    # Color positions
    position_colors = {'QB': '#FF6B6B', 'RB': '#4ECDC4', 'WR': '#45B7D1', 'TE': '#96CEB4'}
    
    # Plot draft flow
    for i, (round_num, pick_num) in enumerate(zip(rounds, picks)):
        if str(pick_num) in strategy:
            pick_data = strategy[str(pick_num)]
            
            # Get primary targets
            primary_targets = pick_data.get('primary_targets', [])
            if primary_targets:
                # Use first primary target for visualization
                target = primary_targets[0]
                
                # Try to get position from VOR data
                position = 'Unknown'
                try:
                    vor_data = load_vor_data()
                    vor_match = vor_data[vor_data['Name'] == target]
                    if not vor_match.empty:
                        position = vor_match.iloc[0]['Position']
                except:
                    pass
                
                color = position_colors.get(position, '#666666')
                
                # Plot pick
                ax.scatter(round_num, pick_num, c=color, s=200, alpha=0.8, edgecolors='black')
                
                # Add player name
                ax.annotate(target, (round_num, pick_num), 
                           xytext=(0, 10), textcoords='offset points', 
                           ha='center', fontsize=8, fontweight='bold')
                
                # Add position label
                ax.annotate(position, (round_num, pick_num), 
                           xytext=(0, -15), textcoords='offset points', 
                           ha='center', fontsize=7, color=color, fontweight='bold')
    
    # Customize chart
    ax.set_xlabel('Draft Round')
    ax.set_ylabel('Overall Pick Number')
    ax.set_title('Draft Strategy for Position 10 (Snake Draft)')
    ax.grid(True, alpha=0.3)
    
    # Add position legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=pos)
                      for pos, color in position_colors.items()]
    ax.legend(handles=legend_elements, title='Position', loc='upper right')
    
    # Skip saving low-value draft_position_strategy plot
    # plot_file = os.path.join(output_dir, 'draft_position_strategy.png')
    # plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return None


def create_uncertainty_analysis_chart(mc_data, output_dir):
    """Create uncertainty analysis chart from Hybrid MC results."""
    if not mc_data:
        print("   ⚠️  Skipping uncertainty analysis - no MC data")
        return None
    
    print("🎲 Creating uncertainty analysis chart...")
    
    # Extract uncertainty data
    uncertainty_data = []
    for player_name, player_data in list(mc_data.items())[:50]:  # Top 50 players
        if 'monte_carlo' in player_data:
            mc_stats = player_data['monte_carlo']
            if 'mean' in mc_stats and 'std' in mc_stats:
                uncertainty_data.append({
                    'Name': player_name,
                    'Mean': mc_stats['mean'],
                    'Std': mc_stats['std'],
                    'CV': mc_stats['std'] / mc_stats['mean'] if mc_stats['mean'] > 0 else 0
                })
    
    if not uncertainty_data:
        print("   ⚠️  No uncertainty data available")
        return None
    
    uncertainty_df = pd.DataFrame(uncertainty_data)
    
    # Create uncertainty visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Coefficient of Variation (Uncertainty vs Mean)
    ax1.scatter(uncertainty_df['Mean'], uncertainty_df['CV'], alpha=0.7, s=50)
    ax1.set_xlabel('Projected Points (Mean)')
    ax1.set_ylabel('Coefficient of Variation (Std/Mean)')
    ax1.set_title('Uncertainty vs Projected Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add player names for high uncertainty players
    high_uncertainty = uncertainty_df.nlargest(10, 'CV')
    for _, row in high_uncertainty.iterrows():
        ax1.annotate(row['Name'], (row['Mean'], row['CV']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Standard Deviation Distribution
    ax2.hist(uncertainty_df['Std'], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Number of Players')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_position_value_vs_risk_chart(mc_data, vor_data, output_dir):
    """Plot positional value (expected points) vs risk (CV) using Hybrid MC results.
    Uses VOR data to map player -> position and aggregates by position.
    """
    if not mc_data:
        print("   ⚠️  Skipping value vs risk - no MC data")
        return None

    print("📈 Creating positional value vs risk chart...")

    # Build per-player stats from MC
    rows = []
    for player_name, pdata in mc_data.items():
        mc = pdata.get('monte_carlo') or {}
        mean = mc.get('mean')
        std = mc.get('std')
        if mean is None or std is None or mean <= 0:
            continue
        # Map position from VOR data if available
        pos = None
        try:
            match = vor_data[vor_data['Name'] == player_name]
            if not match.empty:
                pos = match.iloc[0]['Position']
        except Exception:
            pass
        if not pos:
            pos = 'Unknown'
        rows.append({'Name': player_name, 'Position': pos, 'Mean': float(mean), 'Std': float(std)})

    if not rows:
        print("   ⚠️  No overlapping MC/VOR data for value vs risk")
        return None

    df = pd.DataFrame(rows)
    df['CV'] = df['Std'] / df['Mean']

    # Aggregate by position: average mean and average CV
    agg = df.groupby('Position', as_index=False).agg({'Mean': 'mean', 'CV': 'mean', 'Name': 'count'})
    agg = agg.rename(columns={'Name': 'Count'})

    # Plot
    plt.figure(figsize=(12, 8))
    for _, row in agg.iterrows():
        color = POS_PALETTE.get(row['Position'], '#666666')
        plt.scatter(row['Mean'], row['CV'], s=80 + 5 * row['Count'], color=color, edgecolors='black', alpha=0.8)
        plt.annotate(f"{row['Position']} (n={int(row['Count'])})", (row['Mean'], row['CV']),
                     xytext=(6, 6), textcoords='offset points', fontsize=9, fontweight='bold')

    # Axes and guides
    plt.xlabel('Expected Fantasy Points (Mean)')
    plt.ylabel('Risk (Coefficient of Variation: Std/Mean)')
    plt.title('Positional Value vs Risk (Hybrid MC)')
    plt.grid(True, alpha=0.3)

    # Quadrant lines for guidance
    if len(agg) > 0:
        plt.axvline(x=agg['Mean'].median(), color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=agg['CV'].median(), color='gray', linestyle='--', alpha=0.5)
        plt.text(0.02, 0.95, 'Top-Left: High Value, Low Risk\nBottom-Right: Low Value, High Risk',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plot_file = os.path.join(output_dir, 'positional_value_vs_risk.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def create_draft_summary_dashboard(vor_data, bayes_data, output_dir):
    """Create a comprehensive draft summary dashboard."""
    print("📋 Creating draft summary dashboard...")
    
    # Create summary statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Top 20 VOR Rankings
    top_20 = vor_data.head(20)
    colors = [POS_PALETTE.get(pos, '#666666') for pos in top_20['Position']]
    
    bars1 = ax1.barh(range(len(top_20)), top_20['VOR_Rank'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['Name'], fontsize=8)
    ax1.set_xlabel('VOR Value')
    ax1.set_title('Top 20 Players by VOR')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, top_20['VOR_Rank'])):
        ax1.text(value + 1, i, f'{value:.1f}', va='center', fontweight='bold')
    
    # Plot 2: Position Value Distribution
    position_vor = vor_data.groupby('Position')['VOR_Rank'].agg(['mean', 'count']).reset_index()
    colors = [POS_PALETTE.get(pos, '#666666') for pos in position_vor['Position']]
    
    bars2 = ax2.bar(position_vor['Position'], position_vor['mean'], color=colors, alpha=0.8)
    ax2.set_ylabel('Average VOR Value')
    ax2.set_title('Average VOR Value by Position')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars2, position_vor['count'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Bayesian Strategy Summary
    strategy_summary = {}
    for pick_num, pick_data in bayes_data.get('strategy', {}).items():
        primary_count = len(pick_data.get('primary_targets', []))
        backup_count = len(pick_data.get('backup_options', []))
        strategy_summary[pick_num] = primary_count + backup_count
    
    ax3.bar(strategy_summary.keys(), strategy_summary.values(), alpha=0.8, color='skyblue')
    ax3.set_xlabel('Pick Number')
    ax3.set_ylabel('Number of Options')
    ax3.set_title('Bayesian Strategy: Options per Pick')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: VOR Value Drop-off
    vor_values = vor_data['VOR_Rank'].values
    drop_offs = [vor_values[i] - vor_values[i+1] for i in range(len(vor_values)-1)]
    
    ax4.plot(range(1, len(drop_offs)+1), drop_offs, marker='o', alpha=0.7)
    ax4.set_xlabel('Player Rank')
    ax4.set_ylabel('VOR Value Drop-off')
    ax4.set_title('VOR Value Drop-off Between Consecutive Players')
    ax4.grid(True, alpha=0.3)
    
    # Highlight major drop-offs
    major_drops = [(i+1, drop) for i, drop in enumerate(drop_offs) if drop > 10]
    for rank, drop in major_drops[:5]:  # Top 5 major drops
        ax4.annotate(f'Rank {rank}', (rank, drop), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'draft_summary_dashboard.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_file}")
    return plot_file


def main():
    """Main function to create all pre-draft visualizations."""
    print("=" * 70)
    print("Creating Pre-Draft Fantasy Football Visualizations")
    print("=" * 70)
    
    try:
        # Get output directory
        output_dir = get_output_directory()
        print(f"📁 Output directory: {output_dir}")
        
        # Load data
        vor_data = load_vor_data()
        bayes_data = load_bayesian_strategy()
        mc_data = load_hybrid_mc_results()
        
        print(f"📊 Loaded VOR data: {len(vor_data)} players")
        print(f"🧠 Loaded Bayesian strategy: {len(bayes_data.get('strategy', {}))} picks")
        if mc_data:
            print(f"🎲 Loaded Hybrid MC data: {len(mc_data)} players")
        
        # Create visualizations
        plot_files = []
        
        # 1. Strategy comparison
        # REMOVED: Useless VOR vs Bayesian comparison plot
        
        # 2. Position distribution
        plot_files.append(create_position_distribution_chart(vor_data, output_dir))
        
        # 3. Draft position strategy (skipped - low descriptive value)
        _ = create_draft_position_strategy_chart(bayes_data, output_dir)
        
        # 4. Uncertainty analysis (if available)
        uncertainty_plot = create_uncertainty_analysis_chart(mc_data, output_dir)
        if uncertainty_plot:
            plot_files.append(uncertainty_plot)
        
        # 4b. Positional value vs risk (if MC available)
        val_risk_plot = create_position_value_vs_risk_chart(mc_data, vor_data, output_dir)
        if val_risk_plot:
            plot_files.append(val_risk_plot)
        
        # 5. Comprehensive dashboard
        plot_files.append(create_draft_summary_dashboard(vor_data, bayes_data, output_dir))
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 Pre-Draft Visualizations Complete!")
        print("=" * 70)
        print(f"✅ Generated {len(plot_files)} visualization files:")
        for plot_file in plot_files:
            if plot_file:
                print(f"   📊 {os.path.basename(plot_file)}")
        
        print(f"\n📁 All files saved to: {output_dir}")
        print("🎯 Use these visualizations to inform your draft strategy!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
