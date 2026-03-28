#!/usr/bin/env python3
"""
Draft Value Heatmap - Using ONLY Real VOR Data

This plot shows draft value by round and position using real VOR data
without any synthetic or placeholder data.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from ffbayes.utils.path_constants import get_user_config_file
from .base_plots import DraftStrategyPlot


class DraftValueHeatmap(DraftStrategyPlot):
    """
    Heatmap showing draft value by round and position using real VOR data.
    """
    
    def __init__(self, output_dir: str = None):
        super().__init__("Draft Value Heatmap", output_dir)
    
    def get_required_columns(self) -> list[str]:
        """Return list of required columns for this plot type."""
        return ['Position', 'VOR', 'VALUERANK']
    
    def create_plot(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create draft value heatmap using real data.
        
        Args:
            data: Dictionary with 'vor' key containing DataFrame with VOR data
            
        Returns:
            Plotly figure object
        """
        # Extract VOR data
        vor_df = data.get('vor')
        if vor_df is None or vor_df.empty:
            # Create empty plot if no data
            fig = go.Figure()
            fig.update_layout(
                title="Draft Value Heatmap - No VOR Data Available",
                annotations=[dict(text="No VOR data available for heatmap", showarrow=False)]
            )
            return fig
        
        # Validate required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in vor_df.columns]
        
        if missing_cols:
            # Create plot showing what data is missing
            fig = go.Figure()
            fig.update_layout(
                title="Draft Value Heatmap - Missing Required Data",
                annotations=[dict(
                    text=f"Missing columns: {', '.join(missing_cols)}<br>Available columns: {', '.join(vor_df.columns)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
        
        # Get league size from config or use default
        league_size = self._get_league_size()
        
        # Create round approximation from VALUERANK
        df = vor_df.copy()
        df['round'] = (df['VALUERANK'] - 1) // league_size + 1
        
        # Limit to reasonable draft rounds (1-17)
        df = df[df['round'] <= 17]
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Draft Value Heatmap - No Valid Round Data",
                annotations=[dict(text="No valid round data after processing", showarrow=False)]
            )
            return fig
        
        # Create pivot table for heatmap
        pivot = (
            df.groupby(['Position', 'round'])['VOR']
            .mean()
            .reset_index()
            .pivot(index='Position', columns='round', values='VOR')
            .fillna(0)
        )
        
        if pivot.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Draft Value Heatmap - No Valid Pivot Data",
                annotations=[dict(text="Unable to create pivot table from data", showarrow=False)]
            )
            return fig
        
        # Create heatmap
        z = pivot.values
        x = list(pivot.columns)
        y = list(pivot.index)
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            colorbar=dict(
                title="Mean VOR",
                thickness=15,
                len=0.5,
                y=0.5,
                yanchor='middle'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Draft Value Heatmap by Round and Position",
            xaxis_title="Draft Round",
            yaxis_title="Position",
            height=600
        )
        
        return fig
    
    def _get_league_size(self) -> int:
        """Get league size from config or use default."""
        try:
            import json
            
            config_path = get_user_config_file()
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    league_size = config.get('league_settings', {}).get('league_size', 12)
                    return int(league_size)
        except Exception:
            pass
        
        # Default to 12-team league if config not available
        return 12
