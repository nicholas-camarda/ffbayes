#!/usr/bin/env python3
"""
Uncertainty Overview (Unified) - Minimal visualization using unified dataset only.

Panels:
- Histogram of consistency_score_latest by position (risk proxy)
- Scatter of consistency_score_latest vs floor_ceiling_spread_latest (validation proxy)
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_plots import ValidationPlot


class UncertaintyOverviewUnified(ValidationPlot):
    def __init__(self, output_dir: str | None = None):
        super().__init__("Uncertainty Overview (Unified)", output_dir)

    def create_plot(self, unified_df: pd.DataFrame) -> go.Figure:
        df = unified_df.copy()
        if df.empty:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Risk Histogram", "Risk vs Spread"))
            fig.update_layout(title="Uncertainty Overview - No Data")
            return fig

        req_cols = ['Position', 'consistency_score_latest', 'floor_ceiling_spread_latest']
        for c in req_cols:
            if c not in df.columns:
                df[c] = None

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Risk (CV) by Position", "Risk vs Spread + Calibration"))

        # Panel 1: Histograms per position
        pos_vals = df['Position'].dropna().unique()
        for pos in sorted(pos_vals):
            sub = df[df['Position'] == pos]
            fig.add_trace(
                go.Histogram(x=sub['consistency_score_latest'], name=str(pos), opacity=0.6),
                row=1, col=1
            )
        fig.update_xaxes(title_text="Consistency (CV)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # Panel 2: Scatter Risk vs Spread
        fig.add_trace(
            go.Scatter(
                x=df['consistency_score_latest'],
                y=df['floor_ceiling_spread_latest'],
                mode='markers',
                name='Risk vs Spread',
                opacity=0.5
            ),
            row=1, col=2
        )
        # Simple calibration: bin CV and plot mean spread per bin as line
        try:
            bins = pd.qcut(df['consistency_score_latest'], q=8, duplicates='drop')
            cal = df.groupby(bins)['floor_ceiling_spread_latest'].mean().reset_index()
            # Use bin midpoints for x
            x_vals = [(b.left + b.right) / 2 for b in cal['consistency_score_latest'].cat.categories]
            y_vals = cal['floor_ceiling_spread_latest'].values
            fig.add_trace(
                go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='Calibration',
                           line=dict(color='firebrick'), marker=dict(size=6)),
                row=1, col=2
            )
        except Exception:
            pass
        fig.update_xaxes(title_text="Consistency (CV)", row=1, col=2)
        fig.update_yaxes(title_text="Floor-Ceiling Spread", row=1, col=2)

        fig.update_layout(title="Uncertainty Overview (Unified)", barmode='overlay', height=500)
        return fig
