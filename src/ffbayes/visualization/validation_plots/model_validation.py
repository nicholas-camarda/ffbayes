#!/usr/bin/env python3
"""
Model validation framework for fantasy football predictions.
Provides utilities for predicted vs actual analysis across all model types.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import utilities
from ffbayes.utils.validation_metrics import (
    calculate_draft_round_metrics,
    calculate_model_accuracy_metrics,
    calculate_position_specific_metrics,
)

logger = logging.getLogger(__name__)


class ModelValidationFramework:
    """Framework for validating model predictions against actual outcomes."""
    
    def __init__(self):
        """Initialize the model validation framework."""
        self.validation_results = {}
        self.supported_models = ['vor', 'bayesian', 'hybrid']
    
    def load_historical_data(self, data_sources: Dict[str, Union[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for model validation.
        
        Args:
            data_sources: Dictionary with model names as keys and file paths or DataFrames as values
            
        Returns:
            Dictionary with loaded DataFrames for each model
        """
        loaded_data = {}
        
        try:
            for model_name, source in data_sources.items():
                if isinstance(source, pd.DataFrame):
                    loaded_data[model_name] = source.copy()
                elif isinstance(source, (str, Path)):
                    # Load from file path
                    if str(source).endswith('.csv'):
                        loaded_data[model_name] = pd.read_csv(source)
                    elif str(source).endswith(('.xlsx', '.xls')):
                        loaded_data[model_name] = pd.read_excel(source)
                    else:
                        logger.warning(f"Unsupported file format for {model_name}: {source}")
                        continue
                else:
                    logger.warning(f"Unsupported data source type for {model_name}: {type(source)}")
                    continue
                
                logger.info(f"Loaded {len(loaded_data[model_name])} records for {model_name} model")
            
            return loaded_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return {}
    
    def validate_data_requirements(self, df: pd.DataFrame, 
                                 required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required columns for analysis.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        is_valid = len(missing_columns) == 0
        
        if not is_valid:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        return is_valid, missing_columns
    
    def standardize_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data format for consistent analysis.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Standardized DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Standardize column names (lowercase, underscores)
            df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required data types
            numeric_columns = ['predicted_points', 'actual_points', 'predicted_uncertainty', 'adp']
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Standardize position names
            if 'position' in df_clean.columns:
                df_clean['position'] = df_clean['position'].str.upper()
            
            # Add draft round if missing but ADP available
            if 'draft_round' not in df_clean.columns and 'adp' in df_clean.columns:
                # Approximate draft round from ADP (assuming 12-team league)
                df_clean['draft_round'] = np.ceil(df_clean['adp'] / 12).fillna(0).astype(int)
            
            # Remove rows with critical missing data
            critical_columns = ['predicted_points', 'actual_points']
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=critical_columns)
            after_count = len(df_clean)
            
            if before_count != after_count:
                logger.info(f"Removed {before_count - after_count} rows with missing critical data")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error standardizing data format: {str(e)}")
            return df
    
    def compare_model_performance(self, data_dict: Dict[str, pd.DataFrame],
                                predicted_col: str = 'predicted_points',
                                actual_col: str = 'actual_points') -> Dict[str, Dict[str, float]]:
        """
        Compare performance across multiple models.
        
        Args:
            data_dict: Dictionary with model names as keys and DataFrames as values
            predicted_col: Column name for predicted values
            actual_col: Column name for actual values
            
        Returns:
            Dictionary with model comparison results
        """
        try:
            comparison_results = {}
            
            for model_name, df in data_dict.items():
                # Standardize data format
                df_std = self.standardize_data_format(df)
                
                # Validate required columns
                required_cols = [predicted_col, actual_col]
                is_valid, missing_cols = self.validate_data_requirements(df_std, required_cols)
                
                if not is_valid:
                    logger.warning(f"Skipping {model_name} due to missing columns: {missing_cols}")
                    continue
                
                # Calculate overall model performance
                overall_metrics = calculate_model_accuracy_metrics(
                    df_std[predicted_col].values,
                    df_std[actual_col].values
                )
                
                # Calculate position-specific metrics
                position_metrics = calculate_position_specific_metrics(
                    df_std, predicted_col, actual_col
                )
                
                # Calculate draft round metrics
                round_metrics = calculate_draft_round_metrics(
                    df_std, predicted_col, actual_col
                )
                
                comparison_results[model_name] = {
                    'overall': overall_metrics,
                    'by_position': position_metrics,
                    'by_round': round_metrics,
                    'data_points': len(df_std)
                }
                
                logger.info(f"Completed validation for {model_name}: R² = {overall_metrics['r_squared']:.3f}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {str(e)}")
            return {}
    
    def generate_validation_summary(self, comparison_results: Dict[str, Dict]) -> Dict[str, Union[str, Dict]]:
        """
        Generate summary of validation results with actionable insights.
        
        Args:
            comparison_results: Results from compare_model_performance
            
        Returns:
            Summary dictionary with insights and recommendations
        """
        try:
            summary = {
                'model_rankings': {},
                'position_insights': {},
                'round_insights': {},
                'actionable_insights': []
            }
            
            # Rank models by overall R²
            model_r2 = {model: results['overall']['r_squared'] 
                        for model, results in comparison_results.items()}
            summary['model_rankings'] = dict(sorted(model_r2.items(), key=lambda x: x[1], reverse=True))
            
            # Best model overall
            best_model = max(model_r2, key=model_r2.get)
            best_r2 = model_r2[best_model]
            
            summary['actionable_insights'].append(
                f"Best overall model: {best_model.upper()} with R² = {best_r2:.3f}"
            )
            
            # Position-specific insights
            positions = ['QB', 'RB', 'WR', 'TE']
            for position in positions:
                pos_performance = {}
                for model, results in comparison_results.items():
                    if position in results['by_position']:
                        pos_performance[model] = results['by_position'][position]['r_squared']
                
                if pos_performance:
                    best_pos_model = max(pos_performance, key=pos_performance.get)
                    best_pos_r2 = pos_performance[best_pos_model]
                    summary['position_insights'][position] = {
                        'best_model': best_pos_model,
                        'r_squared': best_pos_r2
                    }
                    
                    if best_pos_r2 != best_r2:
                        summary['actionable_insights'].append(
                            f"For {position}: {best_pos_model.upper()} model performs best (R² = {best_pos_r2:.3f})"
                        )
            
            # Draft round insights
            early_rounds = [1, 2, 3]
            late_rounds = [10, 11, 12, 13, 14, 15, 16]
            
            for round_group, rounds, label in [(early_rounds, early_rounds, "early"), 
                                             (late_rounds, late_rounds, "late")]:
                round_performance = {}
                for model, results in comparison_results.items():
                    round_r2_values = []
                    for round_num in rounds:
                        if round_num in results['by_round']:
                            round_r2_values.append(results['by_round'][round_num]['r_squared'])
                    
                    if round_r2_values:
                        round_performance[model] = np.mean(round_r2_values)
                
                if round_performance:
                    best_round_model = max(round_performance, key=round_performance.get)
                    best_round_r2 = round_performance[best_round_model]
                    summary['round_insights'][label] = {
                        'best_model': best_round_model,
                        'r_squared': best_round_r2
                    }
                    
                    summary['actionable_insights'].append(
                        f"For {label} rounds: {best_round_model.upper()} model performs best (R² = {best_round_r2:.3f})"
                    )
            
            # Model confidence insights
            mae_values = {model: results['overall']['mae'] 
                         for model, results in comparison_results.items()}
            most_accurate = min(mae_values, key=mae_values.get)
            lowest_mae = mae_values[most_accurate]
            
            summary['actionable_insights'].append(
                f"Most accurate predictions: {most_accurate.upper()} with MAE = {lowest_mae:.2f} points"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating validation summary: {str(e)}")
            return {
                'model_rankings': {},
                'position_insights': {},
                'round_insights': {},
                'actionable_insights': [f"Error generating summary: {str(e)}"]
            }
    
    def export_validation_results(self, comparison_results: Dict[str, Dict],
                                summary: Dict[str, Union[str, Dict]],
                                output_path: Optional[str] = None) -> bool:
        """
        Export validation results to file for further analysis.
        
        Args:
            comparison_results: Results from model comparison
            summary: Summary from generate_validation_summary
            output_path: Optional output file path
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if output_path is None:
                # Use the existing pipeline's path structure
                output_path = "validation_results.json"
            
            export_data = {
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'model_comparison': comparison_results,
                'summary': summary
            }
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
            
            export_data = convert_numpy(export_data)
            
            # Save to JSON file
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Validation results exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting validation results: {str(e)}")
            return False
    
    def run_complete_validation(self, data_sources: Dict[str, Union[str, pd.DataFrame]],
                              export_results: bool = True) -> Dict[str, Union[Dict, str]]:
        """
        Run complete model validation workflow.
        
        Args:
            data_sources: Dictionary with model data sources
            export_results: Whether to export results to file
            
        Returns:
            Complete validation results including summary and insights
        """
        try:
            logger.info("Starting complete model validation workflow")
            
            # Load historical data
            data_dict = self.load_historical_data(data_sources)
            
            if not data_dict:
                raise ValueError("No valid data loaded for validation")
            
            # Compare model performance
            comparison_results = self.compare_model_performance(data_dict)
            
            if not comparison_results:
                raise ValueError("No comparison results generated")
            
            # Generate summary and insights
            summary = self.generate_validation_summary(comparison_results)
            
            # Combine all results
            complete_results = {
                'comparison_results': comparison_results,
                'summary': summary,
                'models_validated': list(comparison_results.keys()),
                'validation_status': 'completed'
            }
            
            # Export results if requested
            if export_results:
                export_success = self.export_validation_results(comparison_results, summary)
                complete_results['exported'] = export_success
            
            logger.info("Model validation workflow completed successfully")
            return complete_results
            
        except Exception as e:
            logger.error(f"Error in complete validation workflow: {str(e)}")
            return {
                'comparison_results': {},
                'summary': {'actionable_insights': [f"Validation failed: {str(e)}"]},
                'models_validated': [],
                'validation_status': 'failed',
                'error': str(e)
            }
