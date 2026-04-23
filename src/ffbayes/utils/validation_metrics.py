#!/usr/bin/env python3
"""
Statistical validation utilities for fantasy football plot improvements.
Provides ROC curves, calibration metrics, significance testing, and model validation functions.
"""

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def calculate_roc_metrics(predicted_uncertainty: np.ndarray, 
                         actual_volatility: np.ndarray) -> Dict[str, float]:
    """
    Calculate ROC curve metrics for uncertainty prediction validation.
    """
    try:
        if len(predicted_uncertainty) != len(actual_volatility):
            raise ValueError("Arrays must have same length")
        if len(np.unique(actual_volatility)) < 2:
            logger.warning("Actual volatility has only one class")
            return {'auc': 0.5, 'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'thresholds': np.array([1, 0])}
        fpr, tpr, thresholds = roc_curve(actual_volatility, predicted_uncertainty)
        auc = roc_auc_score(actual_volatility, predicted_uncertainty)
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc}
    except Exception as e:
        logger.error(f"Error calculating ROC metrics: {str(e)}")
        return {'auc': 0.5, 'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'thresholds': np.array([1, 0])}


def calculate_calibration_metrics(predicted_probs: np.ndarray, 
                                actual_outcomes: np.ndarray,
                                n_bins: int = 10) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate calibration metrics including Brier score and calibration curve."""
    try:
        brier_score = brier_score_loss(actual_outcomes, predicted_probs)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_probs, bin_outcomes, bin_counts = [], [], []
        for i in range(n_bins):
            bin_mask = (predicted_probs >= bin_lowers[i]) & (predicted_probs < bin_uppers[i])
            if i == n_bins - 1:
                bin_mask = bin_mask | (predicted_probs == bin_uppers[i])
            if np.sum(bin_mask) > 0:
                bin_prob = np.mean(predicted_probs[bin_mask])
                bin_outcome = np.mean(actual_outcomes[bin_mask])
                bin_count = np.sum(bin_mask)
            else:
                bin_prob = (bin_lowers[i] + bin_uppers[i]) / 2
                bin_outcome = np.nan
                bin_count = 0
            bin_probs.append(bin_prob)
            bin_outcomes.append(bin_outcome)
            bin_counts.append(bin_count)
        return {
            'brier_score': brier_score,
            'bin_boundaries': bin_boundaries,
            'bin_probs': np.array(bin_probs),
            'bin_outcomes': np.array(bin_outcomes),
            'bin_counts': np.array(bin_counts)
        }
    except Exception as e:
        logger.error(f"Error calculating calibration metrics: {str(e)}")
        return {
            'brier_score': 0.5,
            'bin_boundaries': np.linspace(0, 1, n_bins + 1),
            'bin_probs': np.array([np.nan] * n_bins),
            'bin_outcomes': np.array([np.nan] * n_bins),
            'bin_counts': np.array([0] * n_bins)
        }


def calculate_model_accuracy_metrics(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Calculate model accuracy metrics including R², MAE, and correlation."""
    try:
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        mask = ~(np.isnan(predicted) | np.isnan(actual))
        predicted = predicted[mask]
        actual = actual[mask]
        if len(predicted) == 0:
            return {'r_squared': 0.0, 'mae': np.nan, 'correlation': 0.0}
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        mae = float(np.mean(np.abs(actual - predicted)))
        if (
            len(predicted) > 1
            and np.unique(predicted).size > 1
            and np.unique(actual).size > 1
        ):
            corr = float(np.corrcoef(predicted, actual)[0, 1])
        else:
            corr = float('nan')
        return {'r_squared': float(r_squared), 'mae': mae, 'correlation': corr}
    except Exception as e:
        logger.error(f"Error calculating model accuracy metrics: {str(e)}")
        return {'r_squared': 0.0, 'mae': np.nan, 'correlation': 0.0}


def calculate_significance_test(group1: np.ndarray, group2: np.ndarray, test_type: str = 'ttest') -> Dict[str, float]:
    """Calculate significance between two groups (t-test or Mann–Whitney U)."""
    try:
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        if len(group1) < 2 or len(group2) < 2:
            return {'p_value': 1.0, 'effect_size': 0.0, 'test_type': test_type, 'n1': len(group1), 'n2': len(group2)}
        if test_type == 'ttest':
            stat, p = stats.ttest_ind(group1, group2, equal_var=False)
            es = float((np.mean(group1) - np.mean(group2)) / np.std(np.concatenate([group1, group2])))
            return {'p_value': float(p), 'effect_size': es, 'test_type': test_type, 'n1': len(group1), 'n2': len(group2)}
        else:
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            return {'p_value': float(p), 'effect_size': 0.0, 'test_type': test_type, 'n1': len(group1), 'n2': len(group2)}
    except Exception as e:
        logger.error(f"Error calculating significance test: {str(e)}")
        return {'p_value': 1.0, 'effect_size': 0.0, 'test_type': test_type, 'n1': 0, 'n2': 0}


def calculate_position_specific_metrics(df: pd.DataFrame, predicted_col: str, actual_col: str, position_col: str = 'position') -> Dict[str, Dict[str, float]]:
    """Calculate validation metrics grouped by position."""
    try:
        position_metrics: Dict[str, Dict[str, float]] = {}
        for position in df[position_col].unique():
            if pd.isna(position):
                continue
            pos_data = df[df[position_col] == position]
            if len(pos_data) < 3:
                logger.warning(f"Insufficient data for position {position}: {len(pos_data)} points")
                continue
            predicted = pos_data[predicted_col].values
            actual = pos_data[actual_col].values
            metrics = calculate_model_accuracy_metrics(predicted, actual)
            position_metrics[position] = metrics
        return position_metrics
    except Exception as e:
        logger.error(f"Error calculating position-specific metrics: {str(e)}")
        return {}


def calculate_draft_round_metrics(df: pd.DataFrame, predicted_col: str, actual_col: str, draft_round_col: str = 'draft_round') -> Dict[int, Dict[str, float]]:
    """Calculate validation metrics grouped by draft round."""
    try:
        round_metrics: Dict[int, Dict[str, float]] = {}
        for draft_round in sorted(df[draft_round_col].unique()):
            if pd.isna(draft_round):
                continue
            round_data = df[df[draft_round_col] == draft_round]
            if len(round_data) < 3:
                logger.warning(f"Insufficient data for round {draft_round}: {len(round_data)} points")
                continue
            predicted = round_data[predicted_col].values
            actual = round_data[actual_col].values
            metrics = calculate_model_accuracy_metrics(predicted, actual)
            round_metrics[int(draft_round)] = metrics
        return round_metrics
    except Exception as e:
        logger.error(f"Error calculating draft round metrics: {str(e)}")
        return {}


def validate_convergence_diagnostics(trace_data: Dict[str, np.ndarray], r_hat_threshold: float = 1.01) -> Dict[str, Union[bool, float, Dict]]:
    """Validate MCMC convergence using trace data and R-hat statistics."""
    try:
        convergence_results = {'converged': True, 'parameters': {}, 'worst_r_hat': 1.0, 'n_divergent': 0}
        for param_name, trace in trace_data.items():
            if trace.ndim == 1:
                convergence_results['parameters'][param_name] = {'r_hat': 1.0, 'eff_sample_size': len(trace), 'converged': True}
            else:
                n_chains, n_samples = trace.shape
                chain_means = np.mean(trace, axis=1)
                overall_mean = np.mean(chain_means)
                between_var = n_samples * np.var(chain_means, ddof=1)
                within_var = np.mean(np.var(trace, axis=1, ddof=1))
                var_hat = (within_var + (between_var / n_samples))
                r_hat = np.sqrt(var_hat / within_var) if within_var > 0 else 1.0
                ess = n_chains * n_samples / (1 + 2 * np.sum(np.abs(np.corrcoef(trace.reshape(n_chains, -1))))) if within_var > 0 else n_chains * n_samples
                convergence_results['parameters'][param_name] = {
                    'r_hat': float(r_hat),
                    'eff_sample_size': float(ess) if np.isfinite(ess).all() else float(n_chains * n_samples),
                    'converged': r_hat < r_hat_threshold
                }
                if r_hat > convergence_results['worst_r_hat']:
                    convergence_results['worst_r_hat'] = float(r_hat)
                if r_hat >= r_hat_threshold:
                    convergence_results['converged'] = False
        return convergence_results
    except Exception as e:
        logger.error(f"Error validating convergence: {str(e)}")
        return {'converged': False, 'parameters': {}, 'worst_r_hat': 1.0, 'n_divergent': 0}
