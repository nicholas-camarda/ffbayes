#!/usr/bin/env python3
"""
Base plot classes for fantasy football visualization with common styling and error handling.
Provides abstract base classes for different plot types with consistent design and validation.
"""

import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePlot(ABC):
    """Abstract base class for all fantasy football plots."""
    
    # Common color palette for positions
    POSITION_COLORS = {
        'QB': '#1f77b4',  # Blue
        'RB': '#ff7f0e',  # Orange  
        'WR': '#2ca02c',  # Green
        'TE': '#d62728',  # Red
        'K': '#9467bd',   # Purple
        'DST': '#8c564b'  # Brown
    }
    
    # Common layout settings
    DEFAULT_LAYOUT = {
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'showlegend': True,
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80}
    }
    
    def __init__(self, title: str, output_dir: str = None):
        """
        Initialize base plot with title and output directory.
        
        Args:
            title: Plot title
            output_dir: Directory to save plots (defaults to the runtime
                plots tree under `~/ProjectsRuntime/ffbayes/plots/<year>/pre_draft/visualizations/`)
        """
        self.title = title
        self.output_dir = output_dir or self._get_default_output_dir()
        self.figure = None
        self.data_validation_errors = []
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_default_output_dir(self) -> str:
        """Get default output directory for plots."""
        from ffbayes.utils.path_constants import get_pre_draft_plots_dir

        current_year = datetime.now().year
        return str(get_pre_draft_plots_dir(current_year) / "visualizations")
    
    @abstractmethod
    def create_plot(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> go.Figure:
        """
        Create the plot with given data.
        
        Args:
            data: Input data for the plot
            
        Returns:
            Plotly figure object
        """
        pass
    
    def validate_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Validate input data before plotting.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        self.data_validation_errors = []
        
        if data is None:
            self.data_validation_errors.append("Data is None")
            return False
        
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)
        elif isinstance(data, dict):
            return self._validate_dict(data)
        else:
            self.data_validation_errors.append(f"Unsupported data type: {type(data)}")
            return False
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate pandas DataFrame."""
        if df.empty:
            self.data_validation_errors.append("DataFrame is empty")
            return False
        
        # Check for excessive NaN values
        nan_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if nan_percentage > 0.5:
            self.data_validation_errors.append(f"DataFrame has {nan_percentage:.1%} NaN values")
        
        return True
    
    def _validate_dict(self, data: Dict[str, Any]) -> bool:
        """Validate dictionary data."""
        if not data:
            self.data_validation_errors.append("Dictionary is empty")
            return False
        
        return True
    
    def apply_common_styling(self, fig: go.Figure) -> go.Figure:
        """Apply common styling to figure."""
        fig.update_layout(
            title={
                'text': self.title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial, sans-serif'}
            },
            **self.DEFAULT_LAYOUT
        )
        
        return fig
    
    def add_statistical_annotations(self, fig: go.Figure, stats: Dict[str, float]) -> go.Figure:
        """Add statistical annotations to the plot."""
        annotation_text = []
        
        if 'r_squared' in stats:
            annotation_text.append(f"R² = {stats['r_squared']:.3f}")
        if 'mae' in stats:
            annotation_text.append(f"MAE = {stats['mae']:.2f}")
        if 'auc' in stats:
            annotation_text.append(f"AUC = {stats['auc']:.3f}")
        if 'p_value' in stats:
            significance = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "ns"
            annotation_text.append(f"p = {stats['p_value']:.3f} {significance}")
        
        if annotation_text:
            fig.add_annotation(
                text="<br>".join(annotation_text),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, formats: List[str] = None) -> bool:
        """
        Save plot to file.
        
        Args:
            fig: Plotly figure to save
            filename: Base filename (without extension)
            formats: List of formats to save (default: ['png'])
            
        Returns:
            True if save successful, False otherwise
        """
        if formats is None:
            formats = ['png']
        
        saved_any = False
        for format_type in formats:
            try:
                filepath = os.path.join(self.output_dir, f"{filename}.{format_type}")

                if format_type == 'png':
                    # Require kaleido explicitly for reliability
                    try:
                        import kaleido  # noqa: F401
                    except Exception as e:
                        raise RuntimeError(f"Kaleido not available for PNG export: {e}")
                    fig.write_image(filepath, width=1200, height=800, scale=2, engine='kaleido')
                elif format_type == 'html':
                    fig.write_html(filepath)
                elif format_type == 'pdf':
                    fig.write_image(filepath, format='pdf', engine='kaleido')
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue

                logger.info(f"Saved plot: {filepath}")
                saved_any = True
            except Exception as e:
                logger.error(f"Error saving {format_type} for {filename}: {str(e)}")

        return saved_any
    
    def generate_plot(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                     filename: str, formats: List[str] = None) -> Optional[go.Figure]:
        """
        Complete workflow: validate data, create plot, apply styling, and save.
        
        Args:
            data: Input data for the plot
            filename: Base filename for saving
            formats: List of formats to save
            
        Returns:
            Plotly figure if successful, None otherwise
        """
        # Validate data
        if not self.validate_data(data):
            logger.error(f"Data validation failed for {filename}: {self.data_validation_errors}")
            return None
        
        try:
            # Create plot
            fig = self.create_plot(data)
            
            # Apply styling
            fig = self.apply_common_styling(fig)
            
            # Save plot
            if self.save_plot(fig, filename, formats):
                self.figure = fig
                return fig
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating plot {filename}: {str(e)}")
            return None


class ValidationPlot(BasePlot):
    """Base class for validation plots (ROC curves, calibration, convergence)."""
    
    def __init__(self, title: str, output_dir: str = None):
        super().__init__(title, output_dir)
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Enhanced validation for statistical plots."""
        if not super()._validate_dataframe(df):
            return False
        
        # Check minimum data requirements
        if len(df) < 10:
            self.data_validation_errors.append("Insufficient data points for statistical validation (minimum 10)")
            return False
        
        return True
    
    def add_reference_line(self, fig: go.Figure, x_range: tuple, y_range: tuple, 
                          line_type: str = "diagonal") -> go.Figure:
        """Add reference lines for validation plots."""
        if line_type == "diagonal":
            # Perfect prediction line (y = x)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_range,
                mode='lines',
                line=dict(dash='dash', color='gray', width=2),
                name='Perfect Prediction',
                showlegend=True
            ))
        elif line_type == "horizontal":
            # Horizontal reference at y=0.5 for ROC
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                         annotation_text="Random Prediction")
        
        return fig


class DraftStrategyPlot(BasePlot):
    """Base class for draft strategy plots (heatmaps, scatter plots, success rates)."""
    
    def __init__(self, title: str, output_dir: str = None):
        super().__init__(title, output_dir)
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Enhanced validation for draft strategy plots."""
        if not super()._validate_dataframe(df):
            return False
        
        # Check for required columns
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.data_validation_errors.append(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required columns for this plot type."""
        pass
    
    def add_actionable_insight(self, fig: go.Figure, insight: str) -> go.Figure:
        """Add actionable insight annotation to draft strategy plots."""
        fig.add_annotation(
            text=f"<b>Actionable Insight:</b><br>{insight}",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color="darkblue"),
            bgcolor="rgba(173, 216, 230, 0.8)",
            bordercolor="rgba(0, 0, 139, 0.3)",
            borderwidth=2,
            width=400
        )
        
        return fig
