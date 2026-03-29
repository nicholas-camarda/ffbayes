#!/usr/bin/env python3
"""
Draft Strategy Success Rate Plot.

This plot summarizes the share of successful outcomes by draft strategy when
real results are available. It stays intentionally lightweight so the package
can import cleanly even if the caller only needs the other validation plots.
"""

from typing import Dict, Iterable, Optional

import pandas as pd
import plotly.graph_objects as go

from .base_plots import DraftStrategyPlot


class DraftStrategySuccessRate(DraftStrategyPlot):
    """Bar chart of success rate by strategy."""

    def __init__(self, output_dir: str = None):
        super().__init__("Draft Strategy Success Rate", output_dir)

    def get_required_columns(self) -> list[str]:
        """This plot accepts flexible real-world inputs, so no hard requirements."""
        return []

    def create_plot(self, data: Dict[str, pd.DataFrame] | pd.DataFrame) -> go.Figure:
        """Create a success-rate chart from real strategy results."""
        df = self._coerce_dataframe(data)

        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Draft Strategy Success Rate - No Data Available",
                annotations=[dict(text="No strategy results available", showarrow=False)],
            )
            return fig

        strategy_col = self._find_column(df.columns, ['strategy', 'draft_strategy', 'approach'])
        if strategy_col is None:
            strategy_col = '_strategy'
            df[strategy_col] = 'Strategy'

        success_rate_col = self._find_column(
            df.columns,
            ['success_rate', 'success_rate_pct', 'win_rate', 'hit_rate'],
        )
        success_col = self._find_column(df.columns, ['success', 'won', 'hit', 'is_success'])

        if success_rate_col is not None:
            summary = (
                df[[strategy_col, success_rate_col]]
                .dropna()
                .groupby(strategy_col, as_index=False)[success_rate_col]
                .mean()
                .sort_values(success_rate_col, ascending=False)
            )
            y_values = summary[success_rate_col]
            y_label = success_rate_col.replace('_', ' ').title()
        elif success_col is not None:
            working = df[[strategy_col, success_col]].dropna().copy()
            working[success_col] = working[success_col].astype(float)
            summary = (
                working.groupby(strategy_col, as_index=False)[success_col]
                .mean()
                .sort_values(success_col, ascending=False)
            )
            y_values = summary[success_col]
            y_label = "Success Rate"
        else:
            fig = go.Figure()
            fig.update_layout(
                title="Draft Strategy Success Rate - Missing Success Columns",
                annotations=[dict(
                    text=(
                        "Expected one of: success_rate, win_rate, hit_rate, "
                        "success, won, hit, is_success"
                    ),
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                )],
            )
            return fig

        fig = go.Figure(
            data=[
                go.Bar(
                    x=summary[strategy_col].astype(str),
                    y=y_values,
                    marker_color='#1f77b4',
                )
            ]
        )
        fig.update_layout(
            title='Draft Strategy Success Rate',
            xaxis_title='Strategy',
            yaxis_title=y_label,
        )
        return fig

    def _coerce_dataframe(
        self, data: Dict[str, pd.DataFrame] | pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Extract a DataFrame from the supported input shapes."""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            for key in (
                'success_rates',
                'strategy_results',
                'draft_results',
                'results',
            ):
                value = data.get(key)
                if isinstance(value, pd.DataFrame):
                    return value

            for value in data.values():
                if isinstance(value, pd.DataFrame):
                    return value

        return None

    @staticmethod
    def _find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
        """Find the first matching column name from a list of candidates."""
        lower_map = {col.lower(): col for col in columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        return None
