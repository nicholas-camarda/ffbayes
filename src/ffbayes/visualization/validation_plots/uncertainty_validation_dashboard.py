#!/usr/bin/env python3
"""
Uncertainty Validation Dashboard - Using ONLY Real Hybrid MC Data

This dashboard shows uncertainty validation using real Monte Carlo and Bayesian data
without any synthetic or placeholder data.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_plots import ValidationPlot


class UncertaintyValidationDashboard(ValidationPlot):
    """
    Dashboard showing uncertainty validation using real Hybrid MC data.
    """
    
    def __init__(self, output_dir: str = None):
        super().__init__("Uncertainty Validation Dashboard", output_dir)
    
    def create_plot(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create uncertainty validation dashboard using real data.
        
        Args:
            data: Dictionary with 'risk' key containing Hybrid MC results
            
        Returns:
            Plotly figure object
        """
        # Extract Hybrid MC data
        risk_df = data.get('risk')
        if risk_df is None or risk_df.empty:
            # Create empty dashboard if no data
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Uncertainty Distribution by Position",
                    "Monte Carlo Confidence Intervals",
                    "Bayesian Uncertainty Components",
                    "VOR Validation vs Uncertainty"
                ),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            fig.update_layout(
                title="Uncertainty Validation Dashboard - No Hybrid MC Data Available",
                showlegend=False,
                height=800
            )
            return fig
        
        # Check if we have the required uncertainty data
        required_cols = ['overall_uncertainty', 'position']
        missing_cols = [col for col in required_cols if col not in risk_df.columns]
        
        if missing_cols:
            # Create dashboard showing what data is missing
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Missing Data",
                    "Available Columns",
                    "Data Summary",
                    "Next Steps"
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Show missing columns
            fig.add_trace(
                go.Bar(x=['Missing'], y=[len(missing_cols)], name="Missing Columns"),
                row=1, col=1
            )
            
            # Show available columns
            fig.add_trace(
                go.Bar(x=['Available'], y=[len(risk_df.columns)], name="Available Columns"),
                row=1, col=2
            )
            
            # Show data size
            fig.add_trace(
                go.Bar(x=['Records'], y=[len(risk_df)], name="Data Records"),
                row=2, col=1
            )
            
            # Show sample of available columns
            sample_cols = list(risk_df.columns)[:10]
            fig.add_trace(
                go.Bar(x=sample_cols, y=[1] * len(sample_cols), name="Sample Columns"),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"Uncertainty Validation Dashboard - Missing: {', '.join(missing_cols)}",
                height=800
            )
            return fig
        
        # Create subplots (2x2):
        # (1,1) Uncertainty distribution by position
        # (1,2) Monte Carlo confidence intervals
        # (2,1) Bayesian uncertainty components
        # (2,2) VOR validation vs uncertainty
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Uncertainty Distribution by Position",
                "Monte Carlo Confidence Intervals",
                "Bayesian Uncertainty Components",
                "VOR Validation vs Uncertainty"
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plot (1,1): Uncertainty distribution by position
        self._plot_uncertainty_distribution(fig, risk_df, row=1, col=1)
        
        # Plot (1,2): Monte Carlo confidence intervals
        self._plot_mc_confidence_intervals(fig, risk_df, row=1, col=2)
        
        # Plot (2,1): Bayesian uncertainty components
        self._plot_bayesian_components(fig, risk_df, row=2, col=1)
        
        # Plot (2,2): VOR validation vs uncertainty
        self._plot_vor_validation(fig, risk_df, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Uncertainty Validation Dashboard 2025",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _plot_uncertainty_distribution(self, fig: go.Figure, risk_df: pd.DataFrame, row: int, col: int):
        """Plot uncertainty distribution by position."""
        if 'overall_uncertainty' not in risk_df.columns or 'position' not in risk_df.columns:
            return
        
        # Filter out NaN values
        df = risk_df.dropna(subset=['overall_uncertainty', 'position'])
        
        if df.empty:
            return
        
        # Create histogram for each position
        for pos in sorted(df['position'].unique()):
            pos_data = df[df['position'] == pos]['overall_uncertainty']
            if len(pos_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=pos_data,
                        name=f"{pos}",
                        opacity=0.7,
                        marker_color=self.POSITION_COLORS.get(str(pos), '#555555')
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Overall Uncertainty", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _plot_mc_confidence_intervals(self, fig: go.Figure, risk_df: pd.DataFrame, row: int, col: int):
        """Plot Monte Carlo confidence intervals."""
        # Check if we have Monte Carlo data
        mc_cols = [col for col in risk_df.columns if 'monte_carlo' in col.lower()]
        if not mc_cols:
            return
        
        # Try to extract Monte Carlo data
        df = risk_df.copy()
        
        # Look for Monte Carlo results in the data
        if 'monte_carlo' in df.columns and isinstance(df['monte_carlo'].iloc[0], dict):
            # Extract Monte Carlo data from JSON-like structure
            mc_data = []
            for _, row_data in df.iterrows():
                if isinstance(row_data['monte_carlo'], dict):
                    mc = row_data['monte_carlo']
                    if 'mean' in mc and 'std' in mc:
                        mc_data.append({
                            'position': row_data.get('position', 'Unknown'),
                            'mean': mc['mean'],
                            'std': mc['std'],
                            'p5': mc.get('percentiles', {}).get('p5', mc['mean'] - mc['std']),
                            'p95': mc.get('percentiles', {}).get('p95', mc['mean'] + mc['std'])
                        })
            
            if mc_data:
                mc_df = pd.DataFrame(mc_data)
                
                # Plot confidence intervals
                for pos in sorted(mc_df['position'].unique()):
                    pos_data = mc_df[mc_df['position'] == pos]
                    if len(pos_data) > 0:
                        # Plot mean with error bars
                        fig.add_trace(
                            go.Scatter(
                                x=pos_data['mean'],
                                y=pos_data['std'],
                                mode='markers',
                                name=f"{pos} MC",
                                marker=dict(
                                    color=self.POSITION_COLORS.get(str(pos), '#555555'),
                                    size=8
                                ),
                                error_x=dict(
                                    array=pos_data['std'],
                                    visible=True
                                )
                            ),
                            row=row, col=col
                        )
        
        fig.update_xaxes(title_text="Monte Carlo Mean", row=row, col=col)
        fig.update_yaxes(title_text="Standard Deviation", row=row, col=col)
    
    def _plot_bayesian_components(self, fig: go.Figure, risk_df: pd.DataFrame, row: int, col: int):
        """Plot Bayesian uncertainty components."""
        # Check for Bayesian uncertainty data
        bayes_cols = [col for col in risk_df.columns if 'bayesian' in col.lower()]
        if not bayes_cols:
            return
        
        # Try to extract Bayesian uncertainty data
        df = risk_df.copy()
        
        if 'bayesian_uncertainty' in df.columns and isinstance(df['bayesian_uncertainty'].iloc[0], dict):
            # Extract Bayesian uncertainty components
            bayes_data = []
            for _, row_data in df.iterrows():
                if isinstance(row_data['bayesian_uncertainty'], dict):
                    bayes = row_data['bayesian_uncertainty']
                    if 'overall_uncertainty' in bayes:
                        bayes_data.append({
                            'position': row_data.get('position', 'Unknown'),
                            'overall': bayes.get('overall_uncertainty', 0),
                            'consistency': bayes.get('consistency_score', 0),
                            'trend': bayes.get('trend_uncertainty', 0),
                            'position_unc': bayes.get('position_uncertainty', 0)
                        })
            
            if bayes_data:
                bayes_df = pd.DataFrame(bayes_data)
                
                # Calculate average components by position
                avg_components = bayes_df.groupby('position').agg({
                    'overall': 'mean',
                    'consistency': 'mean',
                    'trend': 'mean',
                    'position_unc': 'mean'
                }).reset_index()
                
                if not avg_components.empty:
                    # Create stacked bar chart
                    positions = avg_components['position'].tolist()
                    
                    fig.add_trace(
                        go.Bar(
                            x=positions,
                            y=avg_components['consistency'],
                            name='Consistency',
                            marker_color='lightblue'
                        ),
                        row=row, col=col
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=positions,
                            y=avg_components['trend'],
                            name='Trend',
                            marker_color='lightgreen'
                        ),
                        row=row, col=col
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=positions,
                            y=avg_components['position_unc'],
                            name='Position',
                            marker_color='lightcoral'
                        ),
                        row=row, col=col
                    )
        
        fig.update_xaxes(title_text="Position", row=row, col=col)
        fig.update_yaxes(title_text="Uncertainty Score", row=row, col=col)
    
    def _plot_vor_validation(self, fig: go.Figure, risk_df: pd.DataFrame, row: int, col: int):
        """Plot VOR validation vs uncertainty."""
        # Check for VOR validation data
        vor_cols = [col for col in risk_df.columns if 'vor' in col.lower()]
        if not vor_cols:
            return
        
        # Try to extract VOR validation data
        df = risk_df.copy()
        
        if 'vor_validation' in df.columns and isinstance(df['vor_validation'].iloc[0], dict):
            # Extract VOR validation data
            vor_data = []
            for _, row_data in df.iterrows():
                if isinstance(row_data['vor_validation'], dict):
                    vor = row_data['vor_validation']
                    if 'global_rank' in vor and 'overall_uncertainty' in row_data:
                        vor_data.append({
                            'position': row_data.get('position', 'Unknown'),
                            'global_rank': vor['global_rank'],
                            'uncertainty': row_data['overall_uncertainty'],
                            'tier': vor.get('rank_tier', 'Unknown')
                        })
            
            if vor_data:
                vor_df = pd.DataFrame(vor_data)
                
                # Plot VOR rank vs uncertainty
                for pos in sorted(vor_df['position'].unique()):
                    pos_data = vor_df[vor_df['position'] == pos]
                    if len(pos_data) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=pos_data['global_rank'],
                                y=pos_data['uncertainty'],
                                mode='markers',
                                name=f"{pos} VOR",
                                marker=dict(
                                    color=self.POSITION_COLORS.get(str(pos), '#555555'),
                                    size=8
                                )
                            ),
                            row=row, col=col
                        )
        
        fig.update_xaxes(title_text="VOR Global Rank", row=row, col=col)
        fig.update_yaxes(title_text="Overall Uncertainty", row=row, col=col)

