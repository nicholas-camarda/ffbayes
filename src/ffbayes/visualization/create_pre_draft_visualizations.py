#!/usr/bin/env python3
"""
Create comprehensive pre-draft visualizations for fantasy football draft strategy.
Generates charts that help with draft decision-making before the draft occurs.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ffbayes.utils.path_constants import (get_pre_draft_plots_dir,
                                          get_unified_dataset_csv_path,
                                          get_user_config_file)
from ffbayes.utils.strategy_path_generator import (
    get_bayesian_strategy_path,
    get_hybrid_mc_results_path,
)
from ffbayes.visualization.base_plots import ValidationPlot
from ffbayes.visualization.model_performance_dashboard import ModelPerformanceDashboard
from ffbayes.visualization.uncertainty_overview_unified import UncertaintyOverviewUnified


def get_output_directory():
    """Determine output directory for pre-draft visualizations."""
    current_year = datetime.now().year
    base_dir = str(get_pre_draft_plots_dir(current_year) / "visualizations")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_unified_dataset():
    """Load unified dataset CSV as the single source of truth for viz."""
    csv_path = str(get_unified_dataset_csv_path())
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Unified dataset CSV not found: {csv_path}")
    print(f"📊 Loading unified dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def load_bayesian_strategy():
    """Load Bayesian draft strategy data."""
    bayes_file = get_bayesian_strategy_path()
    
    if not os.path.exists(bayes_file):
        raise FileNotFoundError("No Bayesian strategy file found")
    
    print(f"🧠 Loading Bayesian strategy from: {bayes_file}")
    with open(bayes_file, 'r') as f:
        return json.load(f)

def load_hybrid_mc_results():
    """Load Hybrid MC model results for uncertainty analysis."""
    mc_file = get_hybrid_mc_results_path()
    
    if not os.path.exists(mc_file):
        print("⚠️  No Hybrid MC results found - skipping uncertainty analysis")
        return None
    
    print(f"🎲 Loading Hybrid MC results from: {mc_file}")
    with open(mc_file, 'r') as f:
        return json.load(f)

# ============================================================================
# PLOT 1: MODEL PERFORMANCE DASHBOARD
# ============================================================================

def create_model_performance_dashboard(model_data, output_dir):
    """Create the model performance dashboard from real model inputs."""
    print("📊 Creating Model Performance Dashboard...")

    if isinstance(model_data, pd.DataFrame):
        model_payload = {'unified': model_data}
    elif isinstance(model_data, dict):
        model_payload = model_data
    else:
        model_payload = {}

    if not model_payload:
        print("❌ No model data available - cannot create dashboard without real data")
        return None

    current_year = datetime.now().year
    dashboard = ModelPerformanceDashboard(output_dir)

    fig = dashboard.create_plot(model_payload)
    success = dashboard.save_plot(fig, f'model_performance_dashboard_{current_year}', ['html', 'png'])
    
    if success:
        print(f"✅ Model Performance Dashboard saved to: {output_dir}")
    else:
        print("❌ Failed to save Model Performance Dashboard")
    
    return fig


def create_draft_value_heatmap_from_unified(unified_df: pd.DataFrame, output_dir: str):
    """Create and save the draft value heatmap figure from unified dataset."""

    # Determine position and value columns robustly
    if 'Position' not in unified_df.columns:
        raise KeyError("Unified dataset missing required column: 'Position'")
    pos_series = unified_df['Position']

    # Choose value column preference: vor_value -> RAV -> VOR (if present)
    value_col_name = None
    if 'vor_value' in unified_df.columns:
        value_col_name = 'vor_value'
    elif 'RAV' in unified_df.columns:
        value_col_name = 'RAV'
    elif 'VOR' in unified_df.columns:
        value_col_name = 'VOR'
    else:
        raise KeyError("Unified dataset missing required value column: one of ['vor_value', 'RAV', 'VOR']")

    value_series = pd.to_numeric(unified_df[value_col_name], errors='coerce')

    # Determine rank: prefer vor_global_rank else rank by chosen value
    if 'vor_global_rank' in unified_df.columns:
        rank_series = pd.to_numeric(unified_df['vor_global_rank'], errors='coerce')
    else:
        rank_series = value_series.rank(ascending=False, method='first')

    vor_like = pd.DataFrame({'Position': pos_series, 'Value': value_series, 'VALUERANK': rank_series})

    league_size = 10
    try:
        cfg_path = get_user_config_file()
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            league_size = int(cfg.get('league_settings', {}).get('league_size', league_size))
    except Exception:
        pass

    dfh = vor_like.dropna(subset=['Position', 'Value']).copy()
    dfh['round'] = (dfh['VALUERANK'] - 1) // league_size + 1
    dfh = dfh[dfh['round'] <= 17]
    pivot = (
        dfh.groupby(['Position', 'round'])['Value']
           .mean()
           .reset_index()
           .pivot(index='Position', columns='round', values='Value')
           .fillna(0)
    )
    colorbar_title = f"Mean {('VOR' if value_col_name == 'vor_value' or value_col_name == 'VOR' else 'RAV')}"
    heat = go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale='Viridis',
                      colorbar=dict(title=colorbar_title))
    fig2 = make_subplots(rows=1, cols=1, subplot_titles=("Draft Value Heatmap",))
    fig2.add_trace(heat, row=1, col=1)
    fig2.update_xaxes(title_text="Round", row=1, col=1)
    fig2.update_yaxes(title_text="Position", row=1, col=1)
    current_year = datetime.now().year
    fig2.update_layout(title=f"Draft Value Heatmap {current_year}")

    # Save outputs
    from pathlib import Path as _Path
    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f'draft_value_heatmap_{current_year}.html'
    png_path = out_dir / f'draft_value_heatmap_{current_year}.png'
    fig2.write_html(str(html_path))
    try:
        fig2.write_image(str(png_path))
    except Exception as e:
        # Propagate error to enforce environment correctness
        raise
    return fig2


def _prepare_model_data(raw_data, model_type):
    """
    Prepare model data for visualization using only real data.
    No simulation or placeholder generation.
    """
    if raw_data is None or raw_data.empty:
        return None

    df = raw_data.copy()
    
    # Map actual column names to expected names
    column_mapping = {
        'PLAYER': 'player',
        'POS': 'position',
        'FPTS': 'projected_points',
        'VOR': 'VOR',
        'VALUERANK': 'VALUERANK'
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Only work with actual data - do not simulate anything
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('UNKNOWN')
    
    # Only clean existing data - do not fill missing values with fake data
    for col in ['VOR', 'projected_points', 'VALUERANK']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to create all pre-draft visualizations."""
    print("🚀 Starting pre-draft visualization creation...")
    current_year = datetime.now().year
    
    try:
        # Load single-source data
        unified_df = load_unified_dataset()
        
        # Load proper VOR data for draft strategy analysis
        vor_data = None
        try:
            from ffbayes.utils.path_constants import get_vor_strategy_dir
            from ffbayes.utils.vor_filename_generator import get_vor_csv_filename

            vor_dir = get_vor_strategy_dir(current_year)
            vor_csv = vor_dir / get_vor_csv_filename(current_year)
            if not vor_csv.exists():
                candidates = sorted(vor_dir.glob("snake-draft_ppr-*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                vor_csv = candidates[0] if candidates else vor_csv
            if vor_csv.exists():
                vor_data = pd.read_csv(vor_csv)
                print(f"📊 Loaded VOR strategy data: {len(vor_data)} players")
            else:
                print("⚠️  No VOR strategy file found - using unified dataset for analysis")
        except Exception as e:
            print(f"⚠️  Could not load VOR data: {e}")
        
        # Get output directory
        output_dir = get_output_directory()
        
        # Create Plot 1: Model Performance Dashboard (using proper VOR data if available)
        print("\n" + "="*60)
        print("PLOT 1: MODEL PERFORMANCE DASHBOARD")
        print("="*60)
        
        # Use VOR data if available, otherwise use unified dataset
        if vor_data is not None:
            # Create model data with proper VOR structure
            model_data = {
                'vor': vor_data,  # Use actual VOR data
                'unified': unified_df  # Keep unified for additional context
            }
        else:
            model_data = {'unified': unified_df}

        fig1 = create_model_performance_dashboard(model_data, output_dir)
        
        print("\n✅ Plot 1 completed successfully!")
        print("📊 This plot shows:")
        print("   - MAE by position (QB vs RB vs WR vs TE accuracy)")
        print("   - MAE by draft round (early vs late pick predictability)")
        print("   - Model confidence vs actual accuracy correlation")
        print("   - R² values for different model comparisons")
        print("   - Mean prediction accuracy (predicted vs actual scatter plots)")
        print("   - Monte Carlo validation metrics")
        
        print("\n🎯 Actionable insights from this plot:")
        print("   - Which positions are most predictable")
        print("   - Which draft rounds have highest uncertainty")
        print("   - How much to trust your model confidence")
        print("   - Which model performs best (VOR vs Bayesian vs Hybrid)")
        print("   - Whether your Monte Carlo means are accurate")
        print("   - If your Bayesian models are converging properly")
        
        print("\n📁 Files created:")
        print(f"   - {os.path.join(output_dir, f'model_performance_dashboard_{current_year}.html')}")
        print(f"   - {os.path.join(output_dir, f'model_performance_dashboard_{current_year}.png')}")
        
        # Create Plot 2: Draft Value Heatmap (focused figure)
        print("\n" + "="*60)
        print("PLOT 2: DRAFT VALUE HEATMAP")
        print("="*60)

        # Build VOR-like frame from VOR data if available, otherwise from unified
        if vor_data is not None:
            # Use proper VOR data
            vor_like = vor_data.copy()
            # Map column names to expected format
            vor_like['Position'] = vor_like['POS']
            vor_like['VOR'] = vor_like['VOR']
            vor_like['VALUERANK'] = vor_like['VALUERANK']
        else:
            # Fallback to unified dataset (but this won't have proper VOR)
            vor_like = pd.DataFrame()
            if 'Position' in unified_df.columns:
                vor_like['Position'] = unified_df['Position']
            if 'vor_value' in unified_df.columns:
                vor_like['VOR'] = pd.to_numeric(unified_df['vor_value'], errors='coerce')
            if 'vor_global_rank' in unified_df.columns:
                vor_like['VALUERANK'] = pd.to_numeric(unified_df['vor_global_rank'], errors='coerce')
            else:
                vor_like['VALUERANK'] = vor_like['VOR'].rank(ascending=False, method='first')

        # League size from config
        league_size = 10
        try:
            cfg_path = get_user_config_file()
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                league_size = int(cfg.get('league_settings', {}).get('league_size', league_size))
        except Exception:
            pass

        dfh = vor_like.dropna(subset=['Position', 'VOR']).copy()
        dfh['round'] = (dfh['VALUERANK'] - 1) // league_size + 1
        dfh = dfh[dfh['round'] <= 17]
        pivot = (
            dfh.groupby(['Position', 'round'])['VOR']
               .mean()
               .reset_index()
               .pivot(index='Position', columns='round', values='VOR')
               .fillna(0)
        )
        heat = go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale='Viridis',
                          colorbar=dict(title='Mean VOR'))
        fig2 = make_subplots(rows=1, cols=1, subplot_titles=("Draft Value Heatmap",))
        fig2.add_trace(heat, row=1, col=1)
        fig2.update_xaxes(title_text="Round", row=1, col=1)
        fig2.update_yaxes(title_text="Position", row=1, col=1)
        fig2.update_layout(title=f"Draft Value Heatmap {current_year}")
        try:
            vp = ValidationPlot("Draft Value Heatmap", output_dir)
            vp.save_plot(fig2, f'draft_value_heatmap_{current_year}', ['html', 'png'])
            print(f"✅ Draft Value Heatmap saved to: {output_dir}")
        except Exception as e:
            print(f"❌ Failed to save Draft Value Heatmap: {e}")

        # Create Plot 3: Uncertainty Overview (unified-only, minimal)
        print("\n" + "="*60)
        print("PLOT 3: UNCERTAINTY OVERVIEW (UNIFIED)")
        print("="*60)
        ov = UncertaintyOverviewUnified(output_dir)
        fig3 = ov.create_plot(unified_df)
        ov.save_plot(fig3, f'uncertainty_overview_{current_year}', ['html', 'png'])
        print(f"✅ Uncertainty Overview saved to: {output_dir}")

        print("\n" + "="*60)
        print("PLOTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error creating pre-draft visualizations: {e}")
        raise

if __name__ == "__main__":
    main()
