#!/usr/bin/env python3
"""
Risk-Reward Scatter Plot - Using ONLY Real Data

This plot shows risk vs reward using real VOR and uncertainty data
without any synthetic or placeholder data.
"""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from .base_plots import DraftStrategyPlot


class RiskRewardScatter(DraftStrategyPlot):
    """
    Scatter plot showing risk vs reward using real data.
    """
    
    def __init__(self, output_dir: str = None):
        super().__init__("Risk-Reward Scatter Plot", output_dir)
    
    def get_required_columns(self) -> list[str]:
        """Return list of required columns for this plot type."""
        return ['Position', 'VOR']
    
    def create_plot(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create risk-reward scatter plot using real data.
        
        Args:
            data: Dictionary with 'vor' and 'risk' keys containing DataFrames
            
        Returns:
            Plotly figure object
        """
        # Extract VOR data
        vor_df = data.get('vor')
        risk_df = data.get('risk')
        
        if vor_df is None or vor_df.empty:
            # Create empty plot if no VOR data
            fig = go.Figure()
            fig.update_layout(
                title="Risk-Reward Scatter - No VOR Data Available",
                annotations=[dict(text="No VOR data available for scatter plot", showarrow=False)]
            )
            return fig
        
        # Check required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in vor_df.columns]
        
        if missing_cols:
            # Create plot showing what data is missing
            fig = go.Figure()
            fig.update_layout(
                title="Risk-Reward Scatter - Missing Required Data",
                annotations=[dict(
                    text=f"Missing columns: {', '.join(missing_cols)}<br>Available columns: {', '.join(vor_df.columns)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
        
        # Create base scatter plot with VOR data
        df = vor_df.copy()
        
        # Determine what risk metrics we have available
        risk_metrics = self._identify_available_risk_metrics(df, risk_df)
        
        if not risk_metrics:
            # If no risk metrics available, create simple VOR vs rank plot
            fig = self._create_simple_vor_plot(df)
        else:
            # Create risk-reward scatter with available metrics
            fig = self._create_risk_reward_plot(df, risk_df, risk_metrics)
        
        return fig
    
    def _identify_available_risk_metrics(self, vor_df: pd.DataFrame, risk_df: pd.DataFrame) -> list[str]:
        """Identify what risk metrics are available in the data."""
        available_metrics = []
        
        # Check VOR DataFrame for potential risk metrics
        vor_cols = vor_df.columns.tolist()
        
        # Look for coefficient of variation (CV) or similar
        if 'VOR' in vor_cols:
            # Calculate CV from VOR data if we have multiple values per position
            if 'Position' in vor_cols:
                # Group by position and calculate CV
                pos_stats = vor_df.groupby('Position')['VOR'].agg(['mean', 'std']).reset_index()
                pos_stats['cv'] = pos_stats['std'] / pos_stats['mean']
                if not pos_stats['cv'].isna().all():
                    available_metrics.append('cv_from_vor')
        
        # Check risk DataFrame for uncertainty metrics
        if risk_df is not None and not risk_df.empty:
            risk_cols = risk_df.columns.tolist()
            
            # Look for uncertainty columns
            if 'overall_uncertainty' in risk_cols:
                available_metrics.append('overall_uncertainty')
            if 'consistency_score' in risk_cols:
                available_metrics.append('consistency_score')
            if 'trend_uncertainty' in risk_cols:
                available_metrics.append('trend_uncertainty')
            
            # Look for Monte Carlo data
            if 'monte_carlo' in risk_cols:
                available_metrics.append('monte_carlo_std')
        
        return available_metrics
    
    def _create_simple_vor_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create simple VOR vs rank plot when no risk metrics available."""
        # Use VALUERANK if available, otherwise create rank from VOR
        if 'VALUERANK' in df.columns:
            rank_col = 'VALUERANK'
        elif 'VOR_Rank' in df.columns:
            rank_col = 'VOR_Rank'
        else:
            df['VOR_Rank'] = df['VOR'].rank(ascending=False, method='first')
            rank_col = 'VOR_Rank'
        
        # Create scatter plot by position
        fig = go.Figure()
        
        for pos in sorted(df['Position'].unique()):
            pos_data = df[df['Position'] == pos]
            if len(pos_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pos_data[rank_col],
                        y=pos_data['VOR'],
                        mode='markers',
                        name=f"{pos}",
                        marker=dict(
                            color=self.POSITION_COLORS.get(str(pos), '#555555'),
                            size=8,
                            opacity=0.8
                        ),
                        text=pos_data.get('PLAYER', pos_data.index),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Rank: %{x}<br>' +
                                    'VOR: %{y:.2f}<br>' +
                                    f'Position: {pos}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title="VOR vs Rank (No Risk Metrics Available)",
            xaxis_title="Value Rank (Lower = Better)",
            yaxis_title="VOR (Value Over Replacement)",
            height=600
        )
        
        return fig
    
    def _create_risk_reward_plot(self, vor_df: pd.DataFrame, risk_df: pd.DataFrame, risk_metrics: list[str]) -> go.Figure:
        """Create risk-reward scatter plot with available risk metrics."""
        fig = go.Figure()
        
        # Merge VOR and risk data if possible
        if risk_df is not None and not risk_df.empty:
            # Try to merge on player names or IDs
            merged_df = self._merge_vor_and_risk_data(vor_df, risk_df)
        else:
            merged_df = vor_df.copy()
        
        # Determine x and y axes based on available metrics
        x_metric, y_metric = self._determine_plot_axes(merged_df, risk_metrics)
        
        if x_metric is None or y_metric is None:
            # Fall back to simple plot
            return self._create_simple_vor_plot(vor_df)
        
        # Create scatter plot
        for pos in sorted(merged_df['Position'].unique()):
            pos_data = merged_df[merged_df['Position'] == pos]
            if len(pos_data) > 0:
                # Filter out NaN values for the metrics we're plotting
                valid_data = pos_data.dropna(subset=[x_metric, y_metric])
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data[x_metric],
                            y=valid_data[y_metric],
                            mode='markers',
                            name=f"{pos}",
                            marker=dict(
                                color=self.POSITION_COLORS.get(str(pos), '#555555'),
                                size=8,
                                opacity=0.8
                            ),
                            text=valid_data.get('PLAYER', valid_data.index),
                            hovertemplate='<b>%{text}</b><br>' +
                                        f'{x_metric}: %{{x:.3f}}<br>' +
                                        f'{y_metric}: %{{y:.3f}}<br>' +
                                        f'Position: {pos}<extra></extra>'
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title=f"Risk-Reward Scatter: {x_metric} vs {y_metric}",
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            height=600
        )
        
        return fig
    
    def _merge_vor_and_risk_data(self, vor_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to merge VOR and risk data."""
        # Look for common keys to merge on
        vor_cols = vor_df.columns.tolist()
        risk_cols = risk_df.columns.tolist()
        
        # Try to find player identifier columns
        player_cols = ['PLAYER', 'Name', 'player_name', 'player_id', 'PlayerID']
        vor_player_col = None
        risk_player_col = None
        
        for col in player_cols:
            if col in vor_cols:
                vor_player_col = col
            if col in risk_cols:
                risk_player_col = col
        
        if vor_player_col and risk_player_col:
            try:
                # Merge on player names
                merged = pd.merge(vor_df, risk_df, left_on=vor_player_col, right_on=risk_player_col, how='left')
                
                # Verify that risk columns are present
                if len(merged) > 0:
                    print(f"Merged {len(merged)} records with risk data")
                    print(f"Risk columns in merged data: {[col for col in merged.columns if 'uncertainty' in col or 'consistency' in col or 'trend' in col or 'mc_' in col]}")
                    return merged
            except Exception as e:
                print(f"Merge failed: {e}")
        
        # If merge fails, try to add risk columns manually
        print("Merge failed, attempting manual column addition")
        merged_df = vor_df.copy()
        
        # Add risk columns as NaN
        for col in risk_df.columns:
            if col not in merged_df.columns:
                merged_df[col] = None
        
        # Try to match players by name (case-insensitive)
        for idx, vor_row in merged_df.iterrows():
            vor_player = str(vor_row.get(vor_player_col, '')).lower() if vor_player_col else ''
            
            # Find matching risk data
            for _, risk_row in risk_df.iterrows():
                risk_player = str(risk_row.get(risk_player_col, '')).lower() if risk_player_col else ''
                
                if vor_player and risk_player and vor_player == risk_player:
                    # Copy risk data to merged DataFrame
                    for col in risk_df.columns:
                        if col not in ['player_name', 'position', 'team']:  # Skip identifier columns
                            merged_df.at[idx, col] = risk_row[col]
                    break
        
        print(f"Manual column addition completed. Risk columns: {[col for col in merged_df.columns if 'uncertainty' in col or 'consistency' in col or 'trend' in col or 'mc_' in col]}")
        return merged_df
    
    def _determine_plot_axes(self, df: pd.DataFrame, risk_metrics: list[str]) -> tuple[Optional[str], Optional[str]]:
        """Determine which metrics to use for x and y axes."""
        # Priority order for risk metrics
        risk_priority = ['overall_uncertainty', 'consistency_score', 'trend_uncertainty', 'monte_carlo_std', 'cv_from_vor']
        
        # Find the best available risk metric
        best_risk_metric = None
        for metric in risk_priority:
            if metric in risk_metrics:
                best_risk_metric = metric
                break
        
        if best_risk_metric is None:
            return None, None
        
        # Y-axis should be VOR (reward)
        y_metric = 'VOR'
        
        # X-axis should be the risk metric
        x_metric = best_risk_metric
        
        return x_metric, y_metric
