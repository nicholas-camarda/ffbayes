#!/usr/bin/env python3
"""Evaluation-only sampled hierarchical player model utilities.

This module implements a bounded PyMC-based evaluation lane against the same
season-total draft contract used by the empirical-Bayes production estimator.
It is intentionally not wired into the supported CLI, dashboard, or draft
artifact path unless explicit promotion criteria are met.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ffbayes.analysis.bayesian_player_model import (
    _player_prior_features,
    aggregate_season_player_table,
    build_training_examples,
)
from ffbayes.analysis.bayesian_vor_comparison import (
    _coverage_rate,
    build_player_forecast_validation_summary,
)

logger = logging.getLogger(__name__)

SAMPLED_PARAMETERIZATIONS = ('centered', 'non_centered')
SAMPLED_PRIOR_SCALE_FAMILIES = ('conservative', 'medium', 'weaker')
SAMPLED_TARGET_ACCEPT_VALUES = (0.9, 0.95)
SAMPLED_DRAW_BUDGETS = ((1000, 1000), (1500, 1500))
SAMPLED_STRUCTURAL_VARIANTS = (
    'base',
    'team_season',
    'team_season_rookie_priors',
)

SAMPLED_PROMOTION_CRITERIA = {
    'max_rhat': 1.05,
    'min_ess_bulk': 100.0,
    'max_divergences': 0,
    'max_runtime_seconds': 1800.0,
    'min_mae_improvement': 0.5,
    'min_rmse_improvement': 0.5,
}

_RATE_FEATURE_COLUMNS = {
    'base': [
        'prior_rate_mean',
        'player_weighted_rate',
        'latest_rate',
        'position_mean',
        'last_vor_value',
        'last_adp_rank',
        'role_volatility',
        'team_change_indicator',
        'age',
    ],
    'team_season': [
        'prior_rate_mean',
        'player_weighted_rate',
        'latest_rate',
        'position_mean',
        'last_vor_value',
        'last_adp_rank',
        'role_volatility',
        'team_change_indicator',
        'age',
    ],
    'team_season_rookie_priors': [
        'prior_rate_mean',
        'player_weighted_rate',
        'latest_rate',
        'position_mean',
        'last_vor_value',
        'last_adp_rank',
        'role_volatility',
        'team_change_indicator',
        'age',
        'rookie_draft_round',
        'rookie_draft_pick',
        'rookie_combine_score',
        'depth_chart_rank',
    ],
}

_GAMES_FEATURE_COLUMNS = {
    'base': [
        'prior_games_mean',
        'games_played_mean',
        'games_missed_mean',
        'injury_rate',
        'expected_games',
        'team_change_indicator',
        'age',
        'depth_chart_rank',
    ],
    'team_season': [
        'prior_games_mean',
        'games_played_mean',
        'games_missed_mean',
        'injury_rate',
        'expected_games',
        'team_change_indicator',
        'age',
        'depth_chart_rank',
    ],
    'team_season_rookie_priors': [
        'prior_games_mean',
        'games_played_mean',
        'games_missed_mean',
        'injury_rate',
        'expected_games',
        'team_change_indicator',
        'age',
        'depth_chart_rank',
        'rookie_draft_round',
        'rookie_draft_pick',
        'rookie_combine_score',
    ],
}


@dataclass(frozen=True)
class SampledBayesConfig:
    """One bounded sampled-Bayes training and inference configuration."""

    parameterization: str
    prior_scale_family: str
    target_accept: float
    chains: int
    tune: int
    draws: int
    structural_variant: str

    @property
    def config_id(self) -> str:
        return (
            f'{self.structural_variant}__{self.parameterization}__'
            f'{self.prior_scale_family}__ta-{self.target_accept:.2f}__'
            f'tune-{self.tune}__draws-{self.draws}__chains-{self.chains}'
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['config_id'] = self.config_id
        return payload


def build_sampled_bayes_search_space() -> list[SampledBayesConfig]:
    """Return the bounded sampled-Bayes configuration search space."""
    configs: list[SampledBayesConfig] = []
    for structural_variant in SAMPLED_STRUCTURAL_VARIANTS:
        for parameterization in SAMPLED_PARAMETERIZATIONS:
            for prior_scale_family in SAMPLED_PRIOR_SCALE_FAMILIES:
                for target_accept in SAMPLED_TARGET_ACCEPT_VALUES:
                    for tune, draws in SAMPLED_DRAW_BUDGETS:
                        configs.append(
                            SampledBayesConfig(
                                parameterization=parameterization,
                                prior_scale_family=prior_scale_family,
                                target_accept=target_accept,
                                chains=4,
                                tune=tune,
                                draws=draws,
                                structural_variant=structural_variant,
                            )
                        )
    return configs


def _prior_scale_multiplier(prior_scale_family: str) -> float:
    mapping = {
        'conservative': 0.35,
        'medium': 0.75,
        'weaker': 1.25,
    }
    try:
        return mapping[prior_scale_family]
    except KeyError as exc:
        raise ValueError(f'Unsupported prior scale family: {prior_scale_family!r}') from exc


def _max_abs(values: dict[str, float]) -> float:
    return max(abs(float(value)) for value in values.values())


def _build_target_feature_frame(
    train_history: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    *,
    holdout_year: int,
    replacement_quantile: float,
    min_history_seasons: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in holdout_frame.itertuples(index=False):
        features = _player_prior_features(
            train_history=train_history,
            player_name=str(row.Name),
            position=str(row.Position),
            target_season=int(holdout_year),
            replacement_quantile=replacement_quantile,
            target_profile=row._asdict() if hasattr(row, '_asdict') else None,
        )
        if int(features['season_count']) < min_history_seasons:
            continue
        features['actual_points'] = float(getattr(row, 'fantasy_points'))
        features['fantasy_points_rate'] = float(
            getattr(row, 'fantasy_points_rate', getattr(row, 'fantasy_points'))
        )
        features['games_played'] = float(getattr(row, 'games_played', 0.0))
        rows.append(features)

    if not rows:
        raise ValueError(f'No overlapping players to evaluate in {holdout_year}')
    return pd.DataFrame(rows)


def _prepare_standardized_features(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
    train = train_frame.copy()
    target = target_frame.copy()
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}

    for column in feature_columns:
        train_values = pd.to_numeric(train.get(column), errors='coerce')
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        train_values = train_values.fillna(median)
        target_values = pd.to_numeric(target.get(column), errors='coerce').fillna(median)
        mean = float(train_values.mean()) if len(train_values) else 0.0
        scale = float(train_values.std(ddof=0)) if len(train_values) > 1 else 1.0
        medians[column] = median
        means[column] = mean
        scales[column] = max(scale, 1.0)
        train[column] = train_values
        target[column] = target_values

    train_x = np.column_stack(
        [
            (
                train[column].to_numpy(dtype=float) - means[column]
            )
            / scales[column]
            for column in feature_columns
        ]
    )
    target_x = np.column_stack(
        [
            (
                target[column].to_numpy(dtype=float) - means[column]
            )
            / scales[column]
            for column in feature_columns
        ]
    )
    return train_x, target_x, medians, means, scales


def _category_indices(
    train_values: pd.Series, target_values: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_labels = train_values.fillna('UNKNOWN').astype(str).str.strip().replace('', 'UNKNOWN')
    target_labels = target_values.fillna('UNKNOWN').astype(str).str.strip().replace('', 'UNKNOWN')
    categories = sorted(set(train_labels) | set(target_labels))
    mapping = {label: index for index, label in enumerate(categories)}
    train_index = train_labels.map(mapping).to_numpy(dtype=int)
    target_index = target_labels.map(mapping).to_numpy(dtype=int)
    return train_index, target_index, categories


def _diagnostic_scalar(dataset_like: Any, reducer) -> float:
    values: list[float] = []
    if hasattr(dataset_like, 'data_vars'):
        iterator = dataset_like.data_vars.values()
    else:
        iterator = [dataset_like]
    for data_array in iterator:
        array = np.asarray(data_array, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            values.append(float(reducer(finite)))
    return float(reducer(values)) if values else float('nan')


def _extract_convergence_diagnostics(idata: Any) -> dict[str, Any]:
    import arviz as az

    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method='bulk')
    divergences = int(np.asarray(idata.sample_stats['diverging']).sum())
    diagnostics = {
        'max_rhat': _diagnostic_scalar(rhat, np.max),
        'min_ess_bulk': _diagnostic_scalar(ess_bulk, np.min),
        'divergences': divergences,
        'chain_count': int(idata.posterior.sizes.get('chain', 0)),
        'draw_count': int(idata.posterior.sizes.get('draw', 0)),
    }
    diagnostics['converged'] = bool(
        np.isfinite(diagnostics['max_rhat'])
        and np.isfinite(diagnostics['min_ess_bulk'])
        and diagnostics['max_rhat'] <= SAMPLED_PROMOTION_CRITERIA['max_rhat']
        and diagnostics['min_ess_bulk'] >= SAMPLED_PROMOTION_CRITERIA['min_ess_bulk']
        and diagnostics['divergences'] <= SAMPLED_PROMOTION_CRITERIA['max_divergences']
    )
    return diagnostics


def _fit_sampled_component_model(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    config: SampledBayesConfig,
    random_seed: int,
) -> dict[str, Any]:
    import pymc as pm
    import pytensor.tensor as pt

    if train_frame.empty:
        raise ValueError('No training rows available for sampled component model')

    train_x, target_x, medians, means, scales = _prepare_standardized_features(
        train_frame,
        target_frame,
        feature_columns=feature_columns,
    )
    train_position_index, target_position_index, position_levels = _category_indices(
        train_frame['position'],
        target_frame['position'],
    )
    include_team_effects = config.structural_variant != 'base'
    if include_team_effects:
        train_team_index, target_team_index, team_levels = _category_indices(
            train_frame.get('current_team', pd.Series(index=train_frame.index, dtype=str)),
            target_frame.get(
                'current_team', pd.Series(index=target_frame.index, dtype=str)
            ),
        )
    else:
        train_team_index = np.zeros(len(train_frame), dtype=int)
        target_team_index = np.zeros(len(target_frame), dtype=int)
        team_levels = ['UNUSED']

    y_raw = pd.to_numeric(train_frame[target_column], errors='coerce').fillna(0.0)
    y_mean = float(y_raw.mean()) if len(y_raw) else 0.0
    y_scale = max(float(y_raw.std(ddof=0)), 1.0)
    y_standardized = ((y_raw - y_mean) / y_scale).to_numpy(dtype=float)
    prior_scale = _prior_scale_multiplier(config.prior_scale_family)

    start = time.perf_counter()
    with pm.Model() as model:
        x_data = pm.Data('x_data', train_x)
        position_idx = pm.Data('position_idx', train_position_index)
        intercept = pm.Normal('intercept', mu=0.0, sigma=prior_scale)
        beta = pm.Normal('beta', mu=0.0, sigma=prior_scale, shape=train_x.shape[1])

        position_sigma = pm.HalfNormal('position_sigma', sigma=prior_scale)
        if config.parameterization == 'non_centered':
            position_offset = pm.Normal(
                'position_offset', mu=0.0, sigma=1.0, shape=len(position_levels)
            )
            position_effect = pm.Deterministic(
                'position_effect', position_offset * position_sigma
            )
        else:
            position_effect = pm.Normal(
                'position_effect',
                mu=0.0,
                sigma=position_sigma,
                shape=len(position_levels),
            )

        mu = intercept + pt.dot(x_data, beta) + position_effect[position_idx]

        if include_team_effects:
            team_idx = pm.Data('team_idx', train_team_index)
            team_sigma = pm.HalfNormal('team_sigma', sigma=prior_scale)
            if config.parameterization == 'non_centered':
                team_offset = pm.Normal(
                    'team_offset', mu=0.0, sigma=1.0, shape=len(team_levels)
                )
                team_effect = pm.Deterministic('team_effect', team_offset * team_sigma)
            else:
                team_effect = pm.Normal(
                    'team_effect', mu=0.0, sigma=team_sigma, shape=len(team_levels)
                )
            mu = mu + team_effect[team_idx]

        sigma = pm.HalfNormal('sigma', sigma=max(prior_scale, 0.5))
        pm.Normal('observed', mu=mu, sigma=sigma, observed=y_standardized)

        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=1,
            target_accept=config.target_accept,
            progressbar=False,
            random_seed=random_seed,
            compute_convergence_checks=False,
        )

    runtime_seconds = float(time.perf_counter() - start)
    diagnostics = _extract_convergence_diagnostics(idata)
    diagnostics['runtime_seconds'] = runtime_seconds
    diagnostics['feature_columns'] = feature_columns
    diagnostics['position_levels'] = position_levels
    diagnostics['team_levels'] = team_levels if include_team_effects else []
    diagnostics['train_rows'] = int(len(train_frame))
    diagnostics['target_rows'] = int(len(target_frame))

    posterior = idata.posterior.stack(sample=('chain', 'draw'))
    intercept = np.asarray(posterior['intercept'], dtype=float)
    beta = np.asarray(
        posterior['beta'].transpose('sample', 'beta_dim_0'),
        dtype=float,
    )
    position_effect = np.asarray(
        posterior['position_effect'].transpose('sample', 'position_effect_dim_0'),
        dtype=float,
    )
    sigma = np.asarray(posterior['sigma'], dtype=float)[:, None]

    mu_standardized = (
        intercept[:, None]
        + beta @ target_x.T
        + position_effect[:, target_position_index]
    )
    if include_team_effects:
        team_effect = posterior['team_effect'].transpose(
            'sample', 'team_effect_dim_0'
        )
        team_effect = np.asarray(team_effect, dtype=float)
        mu_standardized = mu_standardized + team_effect[:, target_team_index]

    rng = np.random.default_rng(random_seed)
    predictive_draws = mu_standardized + rng.normal(
        loc=0.0, scale=1.0, size=mu_standardized.shape
    ) * sigma
    predictive_draws = predictive_draws * y_scale + y_mean

    return {
        'config': config.to_dict(),
        'feature_medians': medians,
        'feature_means': means,
        'feature_scales': scales,
        'diagnostics': diagnostics,
        'predictive_draws': predictive_draws.T,
    }


def _combine_component_predictions(
    frame: pd.DataFrame,
    *,
    rate_result: dict[str, Any],
    games_result: dict[str, Any],
) -> pd.DataFrame:
    rate_draws = np.clip(rate_result['predictive_draws'], 0.0, None)
    games_draws = np.clip(
        games_result['predictive_draws'],
        0.0,
        frame['expected_games'].to_numpy(dtype=float)[:, None],
    )
    total_draws = rate_draws * games_draws
    frame = frame.copy()
    frame['posterior_rate_mean'] = rate_draws.mean(axis=1)
    frame['posterior_rate_std'] = np.maximum(rate_draws.std(axis=1, ddof=0), 1.0)
    frame['posterior_games_mean'] = games_draws.mean(axis=1)
    frame['posterior_games_std'] = np.maximum(games_draws.std(axis=1, ddof=0), 1.0)
    frame['posterior_mean'] = total_draws.mean(axis=1)
    frame['posterior_std'] = np.maximum(total_draws.std(axis=1, ddof=0), 1.0)
    frame['posterior_floor'] = np.quantile(total_draws, 0.10, axis=1)
    frame['posterior_ceiling'] = np.quantile(total_draws, 0.90, axis=1)
    return frame


def _evaluate_prediction_table(frame: pd.DataFrame) -> dict[str, Any]:
    truth = pd.to_numeric(frame['actual_points'], errors='coerce')
    pred = pd.to_numeric(frame['posterior_mean'], errors='coerce')
    std = pd.to_numeric(frame['posterior_std'], errors='coerce')
    valid = truth.notna() & pred.notna() & std.notna()
    if valid.sum() == 0:
        raise ValueError('No valid sampled predictions available for evaluation')
    truth_valid = truth[valid]
    pred_valid = pred[valid]
    std_valid = std[valid]
    errors = truth_valid.to_numpy() - pred_valid.to_numpy()
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    score_rank = pd.Series(pred_valid).rank(method='average')
    truth_rank = pd.Series(truth_valid).rank(method='average')
    spearman = float(score_rank.corr(truth_rank, method='pearson'))
    calibration = {
        'interval_50_coverage': _coverage_rate(truth_valid, pred_valid, std_valid, 0.6745),
        'interval_80_coverage': _coverage_rate(truth_valid, pred_valid, std_valid, 1.2816),
        'mean_predictive_std': float(std_valid.mean()),
    }
    return {
        'player_count': int(valid.sum()),
        'mae': mae,
        'rmse': rmse,
        'spearman_rank_correlation': float(np.nan_to_num(spearman, nan=0.0)),
        'calibration': calibration,
    }


def evaluate_sampled_hierarchical_config(
    history: pd.DataFrame,
    *,
    config: SampledBayesConfig,
    holdout_years: Iterable[int] | None = None,
    replacement_quantile: float = 0.2,
    min_history_seasons: int = 0,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate one sampled hierarchical configuration on rolling holdouts."""
    season_table = aggregate_season_player_table(history)
    available_years = sorted(
        int(season) for season in season_table['Season'].dropna().unique()
    )
    resolved_holdout_years = (
        [int(year) for year in holdout_years]
        if holdout_years is not None
        else (available_years[3:] if len(available_years) > 3 else available_years[1:])
    )
    if not resolved_holdout_years:
        raise ValueError('No holdout seasons available for sampled evaluation')

    holdout_tables: list[pd.DataFrame] = []
    component_diagnostics: list[dict[str, Any]] = []
    start = time.perf_counter()
    for offset, holdout_year in enumerate(resolved_holdout_years):
        train = season_table[season_table['Season'] < holdout_year].copy()
        test = season_table[season_table['Season'] == holdout_year].copy()
        training_examples = build_training_examples(
            train,
            replacement_quantile=replacement_quantile,
            min_history_seasons=1,
        )
        target_frame = _build_target_feature_frame(
            train,
            test,
            holdout_year=holdout_year,
            replacement_quantile=replacement_quantile,
            min_history_seasons=min_history_seasons,
        )
        rate_result = _fit_sampled_component_model(
            training_examples,
            target_frame,
            target_column='target_rate',
            feature_columns=_RATE_FEATURE_COLUMNS[config.structural_variant],
            config=config,
            random_seed=random_seed + offset * 2,
        )
        games_result = _fit_sampled_component_model(
            training_examples,
            target_frame,
            target_column='target_games',
            feature_columns=_GAMES_FEATURE_COLUMNS[config.structural_variant],
            config=config,
            random_seed=random_seed + offset * 2 + 1,
        )
        if not rate_result['diagnostics']['converged'] or not games_result['diagnostics']['converged']:
            return {
                'schema_version': 'sampled_player_forecast_evaluation_v1',
                'config': config.to_dict(),
                'status': 'rejected',
                'reason': 'sampled_component_convergence_failed',
                'holdout_years': resolved_holdout_years,
                'component_diagnostics': {
                    'rate': rate_result['diagnostics'],
                    'availability': games_result['diagnostics'],
                },
                'runtime_seconds': float(time.perf_counter() - start),
            }

        combined = _combine_component_predictions(
            target_frame,
            rate_result=rate_result,
            games_result=games_result,
        )
        combined['holdout_year'] = int(holdout_year)
        holdout_tables.append(combined)
        component_diagnostics.append(
            {
                'holdout_year': int(holdout_year),
                'rate': rate_result['diagnostics'],
                'availability': games_result['diagnostics'],
            }
        )

    combined = pd.concat(holdout_tables, ignore_index=True)
    metrics = _evaluate_prediction_table(combined)
    metrics['holdout_years'] = resolved_holdout_years
    runtime_seconds = float(time.perf_counter() - start)
    return {
        'schema_version': 'sampled_player_forecast_evaluation_v1',
        'config': config.to_dict(),
        'status': 'completed',
        'holdout_years': resolved_holdout_years,
        'metrics': metrics,
        'component_diagnostics': component_diagnostics,
        'runtime_seconds': runtime_seconds,
    }


def select_sampled_bayes_winner(
    evaluation_results: list[dict[str, Any]],
    *,
    empirical_baseline: dict[str, Any],
    search_space: Iterable[SampledBayesConfig],
) -> dict[str, Any]:
    """Select the preferred sampled configuration and promotion decision."""
    allowed_config_ids = {config.config_id for config in search_space}
    evaluated = [
        result
        for result in evaluation_results
        if result.get('config', {}).get('config_id') in allowed_config_ids
    ]
    completed = [result for result in evaluated if result.get('status') == 'completed']
    rejected = [result for result in evaluated if result.get('status') != 'completed']

    if not completed:
        return {
            'production_estimator': 'hierarchical_empirical_bayes',
            'promotion_decision': 'keep_empirical_bayes',
            'promoted': False,
            'preferred_config': None,
            'preferred_result': None,
            'evaluated_config_count': len(evaluated),
            'rejected_config_ids': [
                result.get('config', {}).get('config_id') for result in rejected
            ],
        }

    completed = sorted(
        completed,
        key=lambda result: (
            float(result['metrics']['mae']),
            abs(float(result['metrics']['calibration']['interval_80_coverage']) - 0.8),
            float(result['runtime_seconds']),
        ),
    )
    preferred_result = completed[0]
    preferred_metrics = preferred_result['metrics']
    empirical_metrics = empirical_baseline['overall_forecast']
    promoted = bool(
        float(empirical_metrics['mae']) - float(preferred_metrics['mae'])
        >= SAMPLED_PROMOTION_CRITERIA['min_mae_improvement']
        and float(empirical_metrics.get('rmse', np.inf))
        - float(preferred_metrics['rmse'])
        >= SAMPLED_PROMOTION_CRITERIA['min_rmse_improvement']
        and float(preferred_result['runtime_seconds'])
        <= SAMPLED_PROMOTION_CRITERIA['max_runtime_seconds']
    )
    return {
        'production_estimator': (
            'hierarchical_sampled_bayes'
            if promoted
            else 'hierarchical_empirical_bayes'
        ),
        'promotion_decision': (
            'promote_sampled_bayes' if promoted else 'keep_empirical_bayes'
        ),
        'promoted': promoted,
        'preferred_config': preferred_result['config'],
        'preferred_result': preferred_result,
        'evaluated_config_count': len(evaluated),
        'rejected_config_ids': [
            result.get('config', {}).get('config_id') for result in rejected
        ],
    }


def run_sampled_bayes_search(
    history: pd.DataFrame,
    *,
    holdout_years: Iterable[int] | None = None,
    candidate_configs: Iterable[SampledBayesConfig] | None = None,
    max_configs: int | None = None,
    replacement_quantile: float = 0.2,
    min_history_seasons: int = 0,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Run the bounded sampled-Bayes search against the empirical baseline."""
    search_space = list(candidate_configs or build_sampled_bayes_search_space())
    if max_configs is not None:
        search_space = search_space[: max(0, int(max_configs))]

    empirical_baseline = build_player_forecast_validation_summary(
        history,
        holdout_years=holdout_years,
        replacement_quantile=replacement_quantile,
        min_history_seasons=min_history_seasons,
    )
    evaluation_results = [
        evaluate_sampled_hierarchical_config(
            history,
            config=config,
            holdout_years=holdout_years,
            replacement_quantile=replacement_quantile,
            min_history_seasons=min_history_seasons,
            random_seed=random_seed + index * 101,
        )
        for index, config in enumerate(search_space)
    ]
    selection = select_sampled_bayes_winner(
        evaluation_results,
        empirical_baseline=empirical_baseline,
        search_space=search_space,
    )
    return {
        'schema_version': 'sampled_bayes_search_v1',
        'production_estimator': selection['production_estimator'],
        'promotion_decision': selection['promotion_decision'],
        'promoted': selection['promoted'],
        'search_space': [config.to_dict() for config in search_space],
        'empirical_baseline': empirical_baseline,
        'evaluation_results': evaluation_results,
        'selection': selection,
    }
