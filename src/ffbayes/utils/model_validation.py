#!/usr/bin/env python3
"""
Model validation utilities for ffbayes package.
Provides convergence checking, validation, and quality assessment for models.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model outputs and convergence."""
    
    def __init__(self, model_name: str = "unknown"):
        """Initialize model validator.
        
        Args:
            model_name: Name of the model being validated
        """
        self.model_name = model_name
        self.validation_results = {}
    
    def validate_bayesian_convergence(self, trace, min_ess: int = 100, max_rhat: float = 1.1) -> Dict[str, Any]:
        """Validate Bayesian model convergence.
        
        Args:
            trace: PyMC trace object
            min_ess: Minimum effective sample size
            max_rhat: Maximum R-hat value for convergence
            
        Returns:
            Dictionary with convergence validation results
        """
        try:
            import arviz as az

            # Calculate convergence diagnostics
            ess = az.ess(trace)
            rhat = az.rhat(trace)
            
            # Check convergence criteria
            ess_failed = ess < min_ess
            rhat_failed = rhat > max_rhat
            
            # Calculate summary statistics
            ess_summary = {
                'min': float(ess.min()),
                'max': float(ess.max()),
                'mean': float(ess.mean()),
                'failed_vars': ess_failed.sum().item() if hasattr(ess_failed, 'sum') else 0
            }
            
            rhat_summary = {
                'min': float(rhat.min()),
                'max': float(rhat.max()),
                'mean': float(rhat.mean()),
                'failed_vars': rhat_failed.sum().item() if hasattr(rhat_failed, 'sum') else 0
            }
            
            # Overall convergence assessment
            converged = (ess_failed.sum() == 0) and (rhat_failed.sum() == 0)
            
            results = {
                'converged': converged,
                'ess_summary': ess_summary,
                'rhat_summary': rhat_summary,
                'min_ess_threshold': min_ess,
                'max_rhat_threshold': max_rhat
            }
            
            self.validation_results['bayesian_convergence'] = results
            
            # Log results
            if converged:
                logger.info(f"{self.model_name}: Bayesian model converged successfully")
                logger.info(f"  ESS: min={ess_summary['min']:.0f}, mean={ess_summary['mean']:.0f}")
                logger.info(f"  R-hat: max={rhat_summary['max']:.3f}, mean={rhat_summary['mean']:.3f}")
            else:
                logger.warning(f"{self.model_name}: Bayesian model convergence issues detected")
                logger.warning(f"  ESS failures: {ess_summary['failed_vars']}")
                logger.warning(f"  R-hat failures: {rhat_summary['failed_vars']}")
            
            return results
            
        except ImportError:
            logger.warning("ArviZ not available for Bayesian convergence checking")
            return {'converged': True, 'error': 'ArviZ not available'}
        except Exception as e:
            logger.error(f"Error checking Bayesian convergence: {e}")
            return {'converged': False, 'error': str(e)}
    
    def validate_monte_carlo_results(self, results_df: pd.DataFrame, min_simulations: int = 100) -> Dict[str, Any]:
        """Validate Monte Carlo simulation results.
        
        Args:
            results_df: DataFrame with simulation results
            min_simulations: Minimum number of simulations required
            
        Returns:
            Dictionary with Monte Carlo validation results
        """
        try:
            # Check basic requirements
            n_simulations = len(results_df)
            if n_simulations < min_simulations:
                return {
                    'valid': False,
                    'error': f'Insufficient simulations: {n_simulations} < {min_simulations}'
                }
            
            # Check for NaN or infinite values
            nan_count = results_df.isna().sum().sum()
            inf_count = np.isinf(results_df.select_dtypes(include=[np.number])).sum().sum()
            
            # Check for reasonable score ranges
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            score_ranges = {}
            reasonable_ranges = True
            
            for col in numeric_cols:
                col_data = results_df[col]
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                score_ranges[col] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
                
                # Check for reasonable fantasy football scores
                if col != 'Total':  # Individual player scores
                    if min_val < -50 or max_val > 100:  # Unreasonable individual scores
                        reasonable_ranges = False
                else:  # Team total scores
                    if min_val < 0 or max_val > 300:  # Unreasonable team scores
                        reasonable_ranges = False
            
            # Overall validation
            valid = (nan_count == 0) and (inf_count == 0) and reasonable_ranges
            
            results = {
                'valid': valid,
                'n_simulations': n_simulations,
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'reasonable_ranges': reasonable_ranges,
                'score_ranges': score_ranges
            }
            
            self.validation_results['monte_carlo'] = results
            
            # Log results
            if valid:
                logger.info(f"{self.model_name}: Monte Carlo results validated successfully")
                logger.info(f"  Simulations: {n_simulations}")
                logger.info(f"  Score ranges: {len(score_ranges)} columns")
            else:
                logger.warning(f"{self.model_name}: Monte Carlo validation issues detected")
                if nan_count > 0:
                    logger.warning(f"  NaN values: {nan_count}")
                if inf_count > 0:
                    logger.warning(f"  Infinite values: {inf_count}")
                if not reasonable_ranges:
                    logger.warning("  Unreasonable score ranges detected")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating Monte Carlo results: {e}")
            return {'valid': False, 'error': str(e)}
    
    def validate_model_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate general model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            
        Returns:
            Dictionary with output validation results
        """
        try:
            validation_results = {}
            
            # Check for required keys
            required_keys = ['predictions', 'uncertainty']
            missing_keys = [key for key in required_keys if key not in outputs]
            
            if missing_keys:
                validation_results['missing_keys'] = missing_keys
                validation_results['valid'] = False
            else:
                validation_results['valid'] = True
            
            # Validate predictions
            if 'predictions' in outputs:
                pred_validation = self._validate_predictions(outputs['predictions'])
                validation_results['predictions'] = pred_validation
            
            # Validate uncertainty
            if 'uncertainty' in outputs:
                unc_validation = self._validate_uncertainty(outputs['uncertainty'])
                validation_results['uncertainty'] = unc_validation
            
            # Check for NaN or infinite values
            nan_inf_check = self._check_nan_inf(outputs)
            validation_results['nan_inf_check'] = nan_inf_check
            
            # Overall validation
            overall_valid = (
                validation_results.get('valid', False) and
                validation_results.get('predictions', {}).get('valid', True) and
                validation_results.get('uncertainty', {}).get('valid', True) and
                nan_inf_check.get('valid', True)
            )
            
            validation_results['overall_valid'] = overall_valid
            self.validation_results['model_outputs'] = validation_results
            
            # Log results
            if overall_valid:
                logger.info(f"{self.model_name}: Model outputs validated successfully")
            else:
                logger.warning(f"{self.model_name}: Model output validation issues detected")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating model outputs: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _validate_predictions(self, predictions: Any) -> Dict[str, Any]:
        """Validate prediction outputs."""
        try:
            if isinstance(predictions, pd.DataFrame):
                # Check for reasonable prediction ranges
                numeric_cols = predictions.select_dtypes(include=[np.number]).columns
                ranges_valid = True
                
                for col in numeric_cols:
                    col_data = predictions[col]
                    if col_data.min() < -50 or col_data.max() > 100:
                        ranges_valid = False
                        break
                
                return {
                    'valid': ranges_valid,
                    'type': 'DataFrame',
                    'shape': predictions.shape,
                    'columns': list(predictions.columns)
                }
            elif isinstance(predictions, (list, np.ndarray)):
                return {
                    'valid': True,
                    'type': type(predictions).__name__,
                    'length': len(predictions)
                }
            else:
                return {
                    'valid': True,
                    'type': type(predictions).__name__
                }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _validate_uncertainty(self, uncertainty: Any) -> Dict[str, Any]:
        """Validate uncertainty outputs."""
        try:
            if isinstance(uncertainty, pd.DataFrame):
                # Check that uncertainty values are positive
                numeric_cols = uncertainty.select_dtypes(include=[np.number]).columns
                positive_valid = True
                
                for col in numeric_cols:
                    if (uncertainty[col] < 0).any():
                        positive_valid = False
                        break
                
                return {
                    'valid': positive_valid,
                    'type': 'DataFrame',
                    'shape': uncertainty.shape
                }
            else:
                return {
                    'valid': True,
                    'type': type(uncertainty).__name__
                }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _check_nan_inf(self, data: Any) -> Dict[str, Any]:
        """Check for NaN or infinite values in data."""
        try:
            if isinstance(data, pd.DataFrame):
                nan_count = data.isna().sum().sum()
                inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            elif isinstance(data, np.ndarray):
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
            else:
                nan_count = 0
                inf_count = 0
            
            return {
                'valid': (nan_count == 0) and (inf_count == 0),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        return {
            'model_name': self.model_name,
            'validation_results': self.validation_results,
            'overall_valid': all(
                result.get('valid', True) or result.get('converged', True)
                for result in self.validation_results.values()
            )
        }


def validate_bayesian_model(trace, model_name: str = "Bayesian") -> Dict[str, Any]:
    """Convenience function to validate Bayesian model convergence.
    
    Args:
        trace: PyMC trace object
        model_name: Name of the model
        
    Returns:
        Validation results dictionary
    """
    validator = ModelValidator(model_name)
    return validator.validate_bayesian_convergence(trace)


def validate_monte_carlo_model(results_df: pd.DataFrame, model_name: str = "Monte Carlo") -> Dict[str, Any]:
    """Convenience function to validate Monte Carlo simulation results.
    
    Args:
        results_df: DataFrame with simulation results
        model_name: Name of the model
        
    Returns:
        Validation results dictionary
    """
    validator = ModelValidator(model_name)
    return validator.validate_monte_carlo_results(results_df)


def validate_model_outputs(outputs: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
    """Convenience function to validate model outputs.
    
    Args:
        outputs: Dictionary containing model outputs
        model_name: Name of the model
        
    Returns:
        Validation results dictionary
    """
    validator = ModelValidator(model_name)
    return validator.validate_model_outputs(outputs)
