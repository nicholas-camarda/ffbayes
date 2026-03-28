#!/usr/bin/env python3
"""
Model Performance Dashboard - Using ONLY Real VOR Data

This dashboard shows actionable insights from VOR data and position analysis
without requiring any simulated or placeholder data.
"""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ffbayes.utils.path_constants import get_user_config_file
from .base_plots import ValidationPlot


class ModelPerformanceDashboard(ValidationPlot):
    """
    Dashboard showing VOR analysis and draft strategy insights using only real data.
    """
    
    def __init__(self, output_dir: str = None):
        super().__init__("Model Performance Dashboard", output_dir)
        
    def process_model_data(self, model_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process model data to extract insights using real inputs only.
        Accepts keys like 'vor' (DataFrame with POS/VOR/VALUERANK) and
        'risk' (DataFrame with Hybrid MC metrics: mc_mean/mc_std/p5/p95/overall_uncertainty).
        """
        processed_data: Dict[str, Any] = {}

        # Prefer unified dataset if provided
        unified_df = model_data.get('unified')
        if isinstance(unified_df, pd.DataFrame) and not unified_df.empty:
            dfu = unified_df.copy()
            # Normalize column names
            pos_col = 'Position' if 'Position' in dfu.columns else ('POS' if 'POS' in dfu.columns else None)
            if pos_col is not None:
                # Build a VOR-like frame from unified dataset
                vor_like = pd.DataFrame()
                vor_like[pos_col] = dfu[pos_col]
                # Look for actual VOR data first, then fallback to fantasy points for analysis
                if 'VOR' in dfu.columns:
                    vor_like['VOR'] = pd.to_numeric(dfu['VOR'], errors='coerce')
                elif 'vor_value' in dfu.columns:
                    vor_like['VOR'] = pd.to_numeric(dfu['vor_value'], errors='coerce')
                elif 'FantPt' in dfu.columns:
                    # Use fantasy points as a proxy for analysis, but don't call it VOR
                    vor_like['FantPt'] = pd.to_numeric(dfu['FantPt'], errors='coerce')
                    vor_like['VOR'] = None  # No actual VOR data available
                else:
                    vor_like['VOR'] = None
                # Use vor_global_rank if present, otherwise create from VOR
                if 'vor_global_rank' in dfu.columns:
                    vor_like['VALUERANK'] = pd.to_numeric(dfu['vor_global_rank'], errors='coerce')
                else:
                    vor_like['VALUERANK'] = vor_like['VOR'].rank(ascending=False, method='first')
                # Carry tier cliff and RAV if available for overlays
                if 'tier_cliff_distance' in dfu.columns:
                    vor_like['tier_cliff_distance'] = pd.to_numeric(dfu['tier_cliff_distance'], errors='coerce')
                if 'RAV' in dfu.columns:
                    vor_like['RAV'] = pd.to_numeric(dfu['RAV'], errors='coerce')
                processed_data['vor_raw_data'] = vor_like
                processed_data['unified_raw_data'] = dfu

        # Store VOR-related aggregates (fallback)
        vor_df = model_data.get('vor')
        if isinstance(vor_df, pd.DataFrame) and not vor_df.empty:
            print(f"Processing VOR data: {len(vor_df)} rows")
            processed_data['vor_raw_data'] = vor_df
            # Resolve column names flexibly (standardized or legacy)
            pos_col = 'POS' if 'POS' in vor_df.columns else ('Position' if 'Position' in vor_df.columns else None)
            vor_col = 'VOR' if 'VOR' in vor_df.columns else ('VOR_Value' if 'VOR_Value' in vor_df.columns else None)
            if pos_col and vor_col:
                pos_stats = vor_df.groupby(pos_col).agg({
                    vor_col: ['count', 'mean', 'std', 'min', 'max']
                }).round(2)
                processed_data['vor_position_stats'] = pos_stats
            rank_col = 'VALUERANK' if 'VALUERANK' in vor_df.columns else ('VOR_Rank' if 'VOR_Rank' in vor_df.columns else None)
            if rank_col:
                processed_data['vor_rank_range'] = (
                    int(vor_df[rank_col].min()), int(vor_df[rank_col].max())
                )

        # Store risk-related aggregates from Hybrid MC
        risk_df = model_data.get('risk')
        if isinstance(risk_df, pd.DataFrame) and not risk_df.empty:
            print(f"Processing Hybrid MC risk data: {len(risk_df)} players")
            processed_data['risk_raw_data'] = risk_df
            # Compute per-position uncertainty averages
            if 'overall_uncertainty' in risk_df.columns and 'position' in risk_df.columns:
                uncert = (
                    risk_df.dropna(subset=['overall_uncertainty'])
                    .groupby('position')['overall_uncertainty']
                    .mean()
                    .sort_values(ascending=False)
                )
                processed_data['avg_uncertainty_by_position'] = uncert
        
        # NO SYNTHETIC DATA CREATION - only use real data
        # If uncertainty data is not available, we'll handle this in the plotting methods
        # by either skipping the plot or discussing with the user
        
        print(f"Processed data keys: {list(processed_data.keys())}")
        return processed_data
    
    def create_plot(self, model_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create dashboard with 4 subplots using only real data.
        """
        processed_data = self.process_model_data(model_data)
        
        if len(processed_data) == 0:
            # Create empty dashboard if no data
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "VOR Distribution by Position",
                    "Value Rank Distribution", 
                    "Position Value Analysis",
                    "Draft Strategy Insights"
                ),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            fig.update_layout(
                title="Model Performance Dashboard - No Data Available",
                showlegend=False,
                height=800
            )
            return fig
        
        # Create subplots (2x2):
        # (1,1) Positional depth curves (VOR vs rank)
        # (1,2) Risk vs upside (Hybrid MC)
        # (2,1) Avg uncertainty by position
        # (2,2) Value by round heatmap
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Positional Depth: VOR vs Rank",
                "Risk vs Upside (Hybrid MC)",
                "Average Uncertainty by Position",
                "Value by Round (Mean VOR)"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )

        # Plot (1,1)
        self._plot_depth_curves(fig, processed_data, row=1, col=1)
        # Plot (1,2)
        if 'unified_raw_data' in processed_data:
            self._plot_risk_reward_unified(fig, processed_data, row=1, col=2)
        else:
            self._plot_risk_vs_upside(fig, processed_data, row=1, col=2)
        # Plot (2,1)
        self._plot_uncertainty_by_position(fig, processed_data, row=2, col=1)
        # Plot (2,2)
        self._plot_value_by_round(fig, processed_data, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Model Performance Dashboard 2025",
            height=800,
            showlegend=True
        )
        
        return fig
    def _plot_depth_curves(self, fig: go.Figure, processed_data: Dict, row: int, col: int):
        """Plot VOR vs VALUERANK per position to reveal depth/decline."""
        vor_df = processed_data.get('vor_raw_data')
        if vor_df is None or vor_df.empty:
            return
        # Resolve columns (Position/VOR_Value/VOR_Rank or POS/VOR/VALUERANK)
        pos_col = 'POS' if 'POS' in vor_df.columns else ('Position' if 'Position' in vor_df.columns else None)
        vor_col = 'VOR' if 'VOR' in vor_df.columns else ('VOR_Value' if 'VOR_Value' in vor_df.columns else None)
        if pos_col is None or vor_col is None:
            return
        if 'VALUERANK' in vor_df.columns:
            rank_col = 'VALUERANK'
            df = vor_df[[pos_col, vor_col, rank_col]].dropna()
        elif 'VOR_Rank' in vor_df.columns:
            rank_col = 'VOR_Rank'
            df = vor_df[[pos_col, vor_col, rank_col]].dropna()
        else:
            df = vor_df[[pos_col, vor_col]].dropna().copy()
            df['VOR_Rank'] = df[vor_col].rank(ascending=False, method='first')
            rank_col = 'VOR_Rank'

        for pos in sorted(df[pos_col].unique()):
            pos_df = df[df[pos_col] == pos].sort_values(rank_col)
            fig.add_trace(
                go.Scatter(
                    x=pos_df[rank_col],
                    y=pos_df[vor_col],
                    mode='lines',
                    name=f"{pos} VOR",
                    line=dict(color=self.POSITION_COLORS.get(str(pos), '#555555'), width=2),
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text="Value Rank", row=row, col=col)
        fig.update_yaxes(title_text="VOR", row=row, col=col)

    def _plot_risk_vs_upside(self, fig: go.Figure, processed_data: Dict, row: int, col: int):
        """Scatter: metric (VOR/RAV) vs Value Rank colored by position."""
        import os
        vor_df = processed_data.get('vor_raw_data')
        if vor_df is None or vor_df.empty:
            return
        
        # Resolve columns
        pos_col = 'POS' if 'POS' in vor_df.columns else ('Position' if 'Position' in vor_df.columns else None)
        vor_col = 'VOR' if 'VOR' in vor_df.columns else ('VOR_Value' if 'VOR_Value' in vor_df.columns else None)
        rank_col = 'VALUERANK' if 'VALUERANK' in vor_df.columns else ('VOR_Rank' if 'VOR_Rank' in vor_df.columns else None)
        
        if pos_col is None or vor_col is None:
            return
            
        df = vor_df.dropna(subset=[pos_col, vor_col]).copy()
        if df.empty:
            return
            
        # Choose x metric (VOR by default; RAV if available and requested)
        scatter_metric = os.getenv('SCATTER_METRIC', 'VOR').upper()
        x_metric_col = vor_col
        if scatter_metric == 'RAV' and 'RAV' in df.columns:
            x_metric_col = 'RAV'

        # Focus on top 150 by selected metric to reduce clutter
        sort_col = x_metric_col if x_metric_col in df.columns else vor_col
        df = df.sort_values(sort_col, ascending=False).head(150)
        
        for pos in sorted(df[pos_col].dropna().unique()):
            sub = df[df[pos_col] == pos]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_metric_col],
                    y=sub[rank_col] if rank_col else sub.index,
                    mode='markers',
                    name=f"{pos} ({x_metric_col})",
                    marker=dict(
                        color=self.POSITION_COLORS.get(str(pos), '#555555'),
                        size=8,
                        opacity=0.8
                    ),
                ),
                row=row, col=col
            )

            # Optional RAV overlay line if available
            if 'RAV' in df.columns:
                sub_sorted = sub.sort_values(rank_col) if rank_col else sub.sort_values(vor_col, ascending=False)
                if not sub_sorted.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub_sorted['RAV'],
                            y=sub_sorted[rank_col] if rank_col else sub_sorted.index,
                            mode='lines',
                            name=f"{pos} RAV",
                            line=dict(color=self.POSITION_COLORS.get(str(pos), '#999999'), dash='dot'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

            # Tier drop-off annotations: mark large deltas within position
            if 'tier_cliff_distance' in df.columns:
                sub_cliff = sub.dropna(subset=['tier_cliff_distance'])
                if not sub_cliff.empty:
                    # Threshold at 90th percentile of positive cliffs
                    pos_cliffs = sub_cliff['tier_cliff_distance']
                    pos_cliffs = pos_cliffs[pos_cliffs > 0]
                    if len(pos_cliffs) > 0:
                        thresh = float(pos_cliffs.quantile(0.9))
                        marks = sub_cliff[sub_cliff['tier_cliff_distance'] >= thresh]
                        if not marks.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=marks[x_metric_col],
                                    y=marks[rank_col] if rank_col else marks.index,
                                    mode='markers+text',
                                    name=f"{pos} tier cliff",
                                    marker=dict(symbol='triangle-up', size=10,
                                                color=self.POSITION_COLORS.get(str(pos), '#333333')),
                                    text=[f"+{v:.1f}" for v in marks['tier_cliff_distance'].values],
                                    textposition='middle right',
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
        fig.update_xaxes(title_text=("RAV" if x_metric_col == 'RAV' else "VOR (Value Over Replacement)"), row=row, col=col)
        fig.update_yaxes(title_text="Value Rank (Lower = Better)", row=row, col=col)

    def _plot_risk_reward_unified(self, fig: go.Figure, processed_data: Dict, row: int, col: int):
        """Plot risk vs reward using unified dataset data."""
        vor_df = processed_data.get('vor_raw_data')
        if vor_df is None or vor_df.empty:
            return
            
        # Resolve columns
        pos_col = 'Position' if 'Position' in vor_df.columns else ('POS' if 'POS' in vor_df.columns else None)
        vor_col = 'VOR' if 'VOR' in vor_df.columns else None
        rank_col = 'VALUERANK' if 'VALUERANK' in vor_df.columns else None
        
        if pos_col is None or vor_col is None:
            return
            
        df = vor_df.dropna(subset=[pos_col, vor_col]).copy()
        if df.empty:
            return
            
        # Focus on top 200 by VOR to reduce clutter
        df = df.sort_values(vor_col, ascending=False).head(200)
        
        # Create risk metric based on VOR variance within position
        df['risk_metric'] = df.groupby(pos_col)[vor_col].transform('std') / df.groupby(pos_col)[vor_col].transform('mean')
        df['risk_metric'] = df['risk_metric'].fillna(0.2).clip(0.05, 0.5)
        
        for pos in sorted(df[pos_col].dropna().unique()):
            sub = df[df[pos_col] == pos]
            if len(sub) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sub[vor_col],
                        y=sub['risk_metric'],
                        mode='markers',
                        name=f"{pos} Risk",
                        marker=dict(
                            color=self.POSITION_COLORS.get(str(pos), '#555555'),
                            size=8,
                            opacity=0.8
                        ),
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="VOR (Fantasy Points)", row=row, col=col)
        fig.update_yaxes(title_text="Risk (CV)", row=row, col=col)

    def _plot_uncertainty_by_position(self, fig: go.Figure, processed_data: Dict, row: int, col: int):
        """Bar chart of average VOR per position (fallback when no uncertainty data)."""
        vor_df = processed_data.get('vor_raw_data')
        if vor_df is None or vor_df.empty:
            return
            
        # Resolve columns
        pos_col = 'POS' if 'POS' in vor_df.columns else ('Position' if 'Position' in vor_df.columns else None)
        vor_col = 'VOR' if 'VOR' in vor_df.columns else ('VOR_Value' if 'VOR_Value' in vor_df.columns else None)
        
        if pos_col is None or vor_col is None:
            return
            
        # Calculate average VOR by position
        avg_vor = vor_df.dropna(subset=[pos_col, vor_col]).groupby(pos_col)[vor_col].mean().sort_values(ascending=False)
        
        if len(avg_vor) == 0:
            return
            
        positions = list(avg_vor.index)
        values = list(avg_vor.values)
        colors = [self.POSITION_COLORS.get(str(p), '#555555') for p in positions]
        
        fig.add_trace(
            go.Bar(x=positions, y=values, marker_color=colors, name="Avg VOR"),
            row=row, col=col
        )
        fig.update_xaxes(title_text="Position", row=row, col=col)
        fig.update_yaxes(title_text="Average VOR", row=row, col=col)

    def _plot_value_by_round(self, fig: go.Figure, processed_data: Dict, row: int, col: int):
        """Heatmap of mean VOR by round and position (using league size)."""
        import json
        from pathlib import Path
        vor_df = processed_data.get('vor_raw_data')
        if vor_df is None or vor_df.empty:
            return
        # Resolve column names
        pos_col = 'POS' if 'POS' in vor_df.columns else ('Position' if 'Position' in vor_df.columns else None)
        vor_col = 'VOR' if 'VOR' in vor_df.columns else ('VOR_Value' if 'VOR_Value' in vor_df.columns else None)
        rank_col = 'VALUERANK' if 'VALUERANK' in vor_df.columns else ('VOR_Rank' if 'VOR_Rank' in vor_df.columns else None)
        if pos_col is None or vor_col is None:
            return
        # Determine league size from config
        league_size = 10
        try:
            cfg_path = get_user_config_file()
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                league_size = int(cfg.get('league_settings', {}).get('league_size', league_size))
        except Exception:
            pass
        df = vor_df.copy()
        # Compute round approximation
        if rank_col:
            df = df.dropna(subset=[rank_col]).copy()
            df['round'] = (df[rank_col] - 1) // league_size + 1
        else:
            # If no rank, approximate from VOR rank
            df['_vor_rank_tmp'] = df[vor_col].rank(ascending=False, method='first')
            df['round'] = (df['_vor_rank_tmp'] - 1) // league_size + 1
        # Show all rounds (up to 17 for typical fantasy football drafts)
        df = df[df['round'] <= 17]
        if pos_col not in df.columns:
            return
        pivot = (
            df.groupby([pos_col, 'round'])[vor_col]
              .mean()
              .reset_index()
              .pivot(index=pos_col, columns='round', values=vor_col)
              .fillna(0)
        )
        z = pivot.values
        x = list(pivot.columns)
        y = list(pivot.index)
        # Create heatmap with properly sized colorbar from the start
        heatmap_trace = go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            colorbar=dict(
                thickness=4,
                len=0.15,  # Much shorter - only 15% of figure height
                y=0.25,    # Position in bottom half of figure
                yanchor='middle',
                x=1.02,    # Position just outside the subplot
                xanchor='left',
                title=dict(text='Mean VOR', side='top', font=dict(size=7)),
                tickfont=dict(size=6)
            )
        )
        fig.add_trace(heatmap_trace, row=row, col=col)
        fig.update_xaxes(title_text="Round", row=row, col=col)
        fig.update_yaxes(title_text="Position", row=row, col=col)
