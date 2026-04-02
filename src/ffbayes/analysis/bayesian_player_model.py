#!/usr/bin/env python3
"""
Shared Bayesian player-model utilities.

This module provides the season-level feature assembly and empirical-Bayes
posterior projections used by both the direct Bayes-vs-VOR research harness and
the draft-decision backtest. It intentionally uses only local historical data
and transparent closed-form updates so the model remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ffbayes.utils.path_constants import get_unified_dataset_csv_path

FEATURE_AGGREGATES = {
    'FantPtPPR': 'fantasy_points_ppr',
    'adp': 'adp',
    'adp_rank': 'adp_rank',
    'vor_value': 'vor_value',
    'consistency_score_latest': 'consistency_score_latest',
    'floor_ceiling_spread_latest': 'floor_ceiling_spread_latest',
    'team_usage_pct_latest': 'team_usage_pct_latest',
    'recent_form_latest': 'recent_form_latest',
    'season_trend_latest': 'season_trend_latest',
    'role_strength_z': 'role_strength_z',
    'RAV': 'rav',
    'tier_cliff_distance': 'tier_cliff_distance',
    'site_disagreement': 'site_disagreement',
    'adp_std': 'adp_std',
    'Age': 'age',
    'age': 'age',
}

MODEL_FEATURE_COLUMNS = [
    'prior_mean',
    'recent_mean',
    'latest_points',
    'player_weighted_mean',
    'player_trend',
    'player_weighted_std',
    'position_mean',
    'position_std',
    'replacement_baseline',
    'games_played_mean',
    'games_missed_mean',
    'injury_rate',
    'last_adp',
    'last_adp_rank',
    'last_vor_value',
    'last_consistency_score',
    'last_floor_ceiling_spread',
    'last_team_usage_pct',
    'last_recent_form',
    'last_season_trend',
    'last_role_strength_z',
    'last_rav',
    'last_tier_cliff_distance',
    'role_volatility',
    'site_disagreement',
    'team_change_rate',
    'years_in_league',
    'age',
    'usage_x_volatility',
    'adp_x_market_gap',
]


@dataclass
class BayesianRegressionState:
    feature_columns: list[str]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    positions: list[str]
    coefficient_mean: np.ndarray
    coefficient_covariance: np.ndarray
    observation_variance: float


def _safe_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value)


def _normalize_position(value: Any) -> str:
    text = _safe_string(value).strip().upper()
    return text if text else 'UNKNOWN'


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _last_non_null(series: pd.Series) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[-1]


def _mode_or_last(series: pd.Series) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    modes = cleaned.mode()
    if not modes.empty:
        return modes.iloc[-1]
    return cleaned.iloc[-1]


def _non_empty_count(series: pd.Series) -> float:
    cleaned = series.astype(str).str.strip()
    valid = ~(cleaned.isin(['', 'nan', 'None']))
    return float(valid.sum())


def _expected_games_for_season(season: pd.Series) -> pd.Series:
    season_numeric = pd.to_numeric(season, errors='coerce').fillna(2021)
    return np.where(season_numeric >= 2021, 17.0, 16.0)


def load_optional_unified_history() -> pd.DataFrame | None:
    """Load the canonical unified dataset when it is available."""
    dataset_path = get_unified_dataset_csv_path()
    if not dataset_path.exists():
        return None
    try:
        return pd.read_csv(dataset_path)
    except Exception:
        return None


def aggregate_season_player_table(
    history: pd.DataFrame, feature_history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate raw per-game history into a per-player-season modeling table."""
    if history is None or history.empty:
        raise ValueError('history is empty')

    base = history.copy()
    required = {'Season', 'Name', 'Position', 'FantPt'}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    base['Season'] = pd.to_numeric(base['Season'], errors='coerce').astype('Int64')
    base['Name'] = base['Name'].map(_safe_string)
    base['Position'] = base['Position'].map(_normalize_position)
    base['FantPt'] = _coerce_numeric(base['FantPt'])
    if 'FantPtPPR' in base.columns:
        base['FantPtPPR'] = _coerce_numeric(base['FantPtPPR'])
    if 'Tm' in base.columns:
        base['team'] = base['Tm'].map(_safe_string)
    elif 'team' in base.columns:
        base['team'] = base['team'].map(_safe_string)

    base = base.dropna(subset=['Season', 'Name', 'Position', 'FantPt']).copy()

    group_keys = ['Season', 'Name', 'Position']
    aggregate_spec: dict[str, tuple[str, Any]] = {
        'fantasy_points': ('FantPt', 'mean'),
        'games_played': ('FantPt', 'count'),
    }
    if 'FantPtPPR' in base.columns:
        aggregate_spec['fantasy_points_ppr'] = ('FantPtPPR', 'mean')
    if 'team' in base.columns:
        aggregate_spec['team'] = ('team', _mode_or_last)
    if 'GameInjuryStatus' in base.columns:
        aggregate_spec['game_injury_games'] = ('GameInjuryStatus', _non_empty_count)
    if 'PracticeInjuryStatus' in base.columns:
        aggregate_spec['practice_injury_games'] = (
            'PracticeInjuryStatus',
            _non_empty_count,
        )

    season_table = base.groupby(group_keys, as_index=False).agg(**aggregate_spec)

    feature_source = feature_history
    if feature_source is None:
        feature_source = load_optional_unified_history()

    if feature_source is not None and not feature_source.empty:
        features = feature_source.copy()
        features_required = {'Season', 'Name', 'Position'}
        if features_required.issubset(features.columns):
            features['Season'] = pd.to_numeric(
                features['Season'], errors='coerce'
            ).astype('Int64')
            features['Name'] = features['Name'].map(_safe_string)
            features['Position'] = features['Position'].map(_normalize_position)

            feature_aggregates: dict[str, tuple[str, Any]] = {}
            if 'Tm' in features.columns:
                features['team'] = features['Tm'].map(_safe_string)
            elif 'team' in features.columns:
                features['team'] = features['team'].map(_safe_string)
            if 'team' in features.columns:
                feature_aggregates['team'] = ('team', _mode_or_last)

            for source_column, output_column in FEATURE_AGGREGATES.items():
                if source_column in features.columns:
                    feature_aggregates[output_column] = (source_column, 'mean')

            if feature_aggregates:
                feature_table = features.groupby(group_keys, as_index=False).agg(
                    **feature_aggregates
                )
                season_table = season_table.merge(
                    feature_table,
                    on=group_keys,
                    how='left',
                    suffixes=('', '_feature'),
                )
                if 'team_feature' in season_table.columns:
                    season_table['team'] = season_table['team'].combine_first(
                        season_table['team_feature']
                    )
                    season_table = season_table.drop(columns=['team_feature'])

    if 'fantasy_points_ppr' not in season_table.columns:
        season_table['fantasy_points_ppr'] = season_table['fantasy_points']

    expected_games = _expected_games_for_season(season_table['Season'])
    season_table['games_missed'] = np.maximum(
        0.0, expected_games - pd.to_numeric(season_table['games_played'], errors='coerce')
    )
    if 'game_injury_games' in season_table.columns:
        season_table['games_missed'] = np.maximum(
            season_table['games_missed'],
            pd.to_numeric(season_table['game_injury_games'], errors='coerce').fillna(0.0),
        )

    season_table = season_table.sort_values(['Name', 'Position', 'Season']).reset_index(
        drop=True
    )
    season_table['years_in_league'] = (
        season_table.groupby(['Name', 'Position']).cumcount() + 1
    )
    if 'age' not in season_table.columns:
        season_table['age'] = np.nan
    season_table['position_group_size'] = season_table.groupby(['Season', 'Position'])[
        'Name'
    ].transform('count')
    return season_table


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(max(variance, 0.0)))


def _recency_weights(seasons: pd.Series, target_season: int, decay: float = 0.72) -> np.ndarray:
    gaps = np.maximum(0.0, float(target_season) - seasons.to_numpy(dtype=float))
    return np.power(decay, gaps)


def _player_prior_features(
    train_history: pd.DataFrame,
    player_name: str,
    position: str,
    target_season: int,
    replacement_quantile: float = 0.2,
) -> dict[str, Any]:
    history = train_history[train_history['Season'] < target_season].copy()
    player_hist = history[
        (history['Name'] == player_name) & (history['Position'] == position)
    ].sort_values('Season')
    position_hist = history[history['Position'] == position].sort_values('Season')
    overall_points = pd.to_numeric(history['fantasy_points'], errors='coerce').dropna()
    position_points = pd.to_numeric(
        position_hist['fantasy_points'], errors='coerce'
    ).dropna()

    overall_mean = float(overall_points.mean()) if not overall_points.empty else 0.0
    overall_std = float(overall_points.std(ddof=0)) if len(overall_points) > 1 else 12.0
    position_mean = (
        float(position_points.mean()) if not position_points.empty else overall_mean
    )
    position_std = (
        float(position_points.std(ddof=0))
        if len(position_points) > 1
        else max(overall_std, 8.0)
    )
    replacement_baseline = (
        float(position_points.quantile(replacement_quantile))
        if not position_points.empty
        else position_mean
    )

    season_count = int(len(player_hist))
    if season_count == 0:
        return {
            'player_name': player_name,
            'position': position,
            'target_season': int(target_season),
            'season_count': 0,
            'prior_mean': position_mean,
            'prior_std': max(position_std, 12.0),
            'recent_mean': position_mean,
            'latest_points': position_mean,
            'player_weighted_mean': position_mean,
            'player_trend': 0.0,
            'player_weighted_std': position_std,
            'position_mean': position_mean,
            'position_std': position_std,
            'replacement_baseline': replacement_baseline,
            'games_played_mean': 0.0,
            'games_missed_mean': 0.0,
            'injury_rate': 0.0,
            'last_adp': np.nan,
            'last_adp_rank': np.nan,
            'last_vor_value': 0.0,
            'last_consistency_score': 0.0,
            'last_floor_ceiling_spread': position_std,
            'last_team_usage_pct': 0.0,
            'last_recent_form': 0.0,
            'last_season_trend': 0.0,
            'last_role_strength_z': 0.0,
            'last_rav': 0.0,
            'last_tier_cliff_distance': 0.0,
            'role_volatility': 1.0,
            'site_disagreement': 0.0,
            'team_change_rate': 0.0,
            'years_in_league': 0.0,
            'age': np.nan,
            'usage_x_volatility': 0.0,
            'adp_x_market_gap': 0.0,
            'historical_vor_proxy_point': position_mean,
            'historical_vor_proxy_score': position_mean - replacement_baseline,
            'market_proxy_score': 0.0,
        }

    weights = _recency_weights(player_hist['Season'], target_season)
    player_points = pd.to_numeric(player_hist['fantasy_points'], errors='coerce').fillna(
        position_mean
    )
    latest = player_hist.iloc[-1]
    latest_points = float(player_points.iloc[-1])
    recent_mean = float(player_points.tail(2).mean())
    player_weighted_mean = _weighted_mean(player_points.to_numpy(dtype=float), weights)
    player_weighted_std = _weighted_std(player_points.to_numpy(dtype=float), weights)

    if season_count >= 2:
        seasons = player_hist['Season'].to_numpy(dtype=float)
        centered = seasons - seasons.mean()
        denom = float(np.dot(centered, centered))
        if denom > 0:
            player_trend = float(np.dot(centered, player_points.to_numpy(dtype=float) - player_points.mean()) / denom)
        else:
            player_trend = 0.0
    else:
        player_trend = 0.0

    shrinkage = season_count / (season_count + 2.5)
    prior_mean = (
        shrinkage * recent_mean
        + (1.0 - shrinkage) * position_mean
        + 0.30 * player_trend
    )
    prior_std = max(
        player_weighted_std * max(0.75, 1.20 - 0.08 * min(season_count, 4)),
        position_std * 0.55,
        6.0,
    )

    games_played = pd.to_numeric(player_hist.get('games_played'), errors='coerce').fillna(0.0)
    games_missed = pd.to_numeric(player_hist.get('games_missed'), errors='coerce').fillna(0.0)
    expected_games = np.maximum(1.0, _expected_games_for_season(player_hist['Season']))
    injury_rate = float((games_missed / expected_games).mean()) if len(player_hist) else 0.0
    role_volatility = float(
        np.clip(
            player_weighted_std / max(abs(player_weighted_mean), 1.0),
            0.0,
            1.5,
        )
    )

    last_adp = float(pd.to_numeric(pd.Series([latest.get('adp')]), errors='coerce').iloc[0]) if pd.notna(latest.get('adp')) else np.nan
    last_adp_rank = float(pd.to_numeric(pd.Series([latest.get('adp_rank')]), errors='coerce').iloc[0]) if pd.notna(latest.get('adp_rank')) else np.nan
    last_vor_value = float(pd.to_numeric(pd.Series([latest.get('vor_value')]), errors='coerce').fillna(0.0).iloc[0])
    last_consistency = float(pd.to_numeric(pd.Series([latest.get('consistency_score_latest')]), errors='coerce').fillna(0.0).iloc[0])
    last_spread = float(pd.to_numeric(pd.Series([latest.get('floor_ceiling_spread_latest')]), errors='coerce').fillna(player_weighted_std).iloc[0])
    last_usage = float(pd.to_numeric(pd.Series([latest.get('team_usage_pct_latest')]), errors='coerce').fillna(0.0).iloc[0])
    last_recent_form = float(pd.to_numeric(pd.Series([latest.get('recent_form_latest')]), errors='coerce').fillna(0.0).iloc[0])
    last_season_trend = float(pd.to_numeric(pd.Series([latest.get('season_trend_latest')]), errors='coerce').fillna(0.0).iloc[0])
    last_role_strength = float(pd.to_numeric(pd.Series([latest.get('role_strength_z')]), errors='coerce').fillna(0.0).iloc[0])
    last_rav = float(pd.to_numeric(pd.Series([latest.get('rav')]), errors='coerce').fillna(0.0).iloc[0])
    last_tier_cliff = float(pd.to_numeric(pd.Series([latest.get('tier_cliff_distance')]), errors='coerce').fillna(0.0).iloc[0])
    site_disagreement = float(pd.to_numeric(pd.Series([latest.get('site_disagreement')]), errors='coerce').fillna(role_volatility).iloc[0])
    age = float(pd.to_numeric(pd.Series([latest.get('age')]), errors='coerce').iloc[0]) if pd.notna(latest.get('age')) else np.nan

    player_teams = player_hist.get('team')
    if player_teams is None or len(player_teams.dropna()) <= 1:
        team_change_rate = 0.0
    else:
        changes = player_teams.fillna(method='ffill').ne(player_teams.fillna(method='ffill').shift()).sum()
        team_change_rate = float(max(0, changes - 1) / max(1, season_count - 1))

    market_proxy_score = -last_adp if not np.isnan(last_adp) else latest_points

    return {
        'player_name': player_name,
        'position': position,
        'target_season': int(target_season),
        'season_count': season_count,
        'prior_mean': float(prior_mean),
        'prior_std': float(prior_std),
        'recent_mean': recent_mean,
        'latest_points': latest_points,
        'player_weighted_mean': float(player_weighted_mean),
        'player_trend': float(player_trend),
        'player_weighted_std': float(max(player_weighted_std, 1.0)),
        'position_mean': float(position_mean),
        'position_std': float(max(position_std, 1.0)),
        'replacement_baseline': float(replacement_baseline),
        'games_played_mean': float(games_played.mean()) if len(games_played) else 0.0,
        'games_missed_mean': float(games_missed.mean()) if len(games_missed) else 0.0,
        'injury_rate': injury_rate,
        'last_adp': last_adp,
        'last_adp_rank': last_adp_rank,
        'last_vor_value': last_vor_value,
        'last_consistency_score': last_consistency,
        'last_floor_ceiling_spread': last_spread,
        'last_team_usage_pct': last_usage,
        'last_recent_form': last_recent_form,
        'last_season_trend': last_season_trend,
        'last_role_strength_z': last_role_strength,
        'last_rav': last_rav,
        'last_tier_cliff_distance': last_tier_cliff,
        'role_volatility': role_volatility,
        'site_disagreement': site_disagreement,
        'team_change_rate': team_change_rate,
        'years_in_league': float(player_hist['years_in_league'].max()),
        'age': age,
        'usage_x_volatility': float(last_usage * role_volatility),
        'adp_x_market_gap': float(
            0.0 if np.isnan(last_adp) else last_adp * (last_vor_value - replacement_baseline)
        ),
        'historical_vor_proxy_point': latest_points,
        'historical_vor_proxy_score': latest_points - replacement_baseline,
        'market_proxy_score': float(market_proxy_score),
    }


def build_training_examples(
    train_history: pd.DataFrame,
    replacement_quantile: float = 0.2,
    min_history_seasons: int = 1,
) -> pd.DataFrame:
    """Build lagged, draft-time-safe training examples from prior seasons."""
    rows: list[dict[str, Any]] = []
    train_sorted = train_history.sort_values(['Season', 'Name', 'Position']).reset_index(drop=True)
    for row in train_sorted.itertuples(index=False):
        features = _player_prior_features(
            train_history=train_sorted,
            player_name=str(row.Name),
            position=str(row.Position),
            target_season=int(row.Season),
            replacement_quantile=replacement_quantile,
        )
        if int(features['season_count']) < min_history_seasons:
            continue
        features['target_points'] = float(row.fantasy_points)
        rows.append(features)
    return pd.DataFrame(rows)


def fit_bayesian_regression(training_examples: pd.DataFrame) -> BayesianRegressionState | None:
    """Fit a transparent empirical-Bayes linear model with recency weighting."""
    if training_examples is None or training_examples.empty:
        return None

    frame = training_examples.copy()
    numeric_medians: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_scales: dict[str, float] = {}

    for column in MODEL_FEATURE_COLUMNS:
        values = pd.to_numeric(frame.get(column), errors='coerce')
        median = float(values.median()) if values.notna().any() else 0.0
        centered = values.fillna(median)
        mean = float(centered.mean()) if len(centered) else 0.0
        scale = float(centered.std(ddof=0)) if len(centered) > 1 else 1.0
        numeric_medians[column] = median
        numeric_means[column] = mean
        numeric_scales[column] = max(scale, 1.0)
        frame[column] = centered

    positions = sorted(frame['position'].dropna().astype(str).unique().tolist())
    X_parts = [np.ones((len(frame), 1), dtype=float)]
    for column in MODEL_FEATURE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        X_parts.append(((values - numeric_means[column]) / numeric_scales[column])[:, None])
    for position in positions:
        X_parts.append((frame['position'].astype(str) == position).astype(float).to_numpy()[:, None])
    X = np.hstack(X_parts)

    y = pd.to_numeric(frame['target_points'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    seasons = pd.to_numeric(frame['target_season'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    max_season = seasons.max() if len(seasons) else 0.0
    weights = np.exp(-0.18 * (max_season - seasons))

    base_variance = max(float(np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)), 1.0)
    prior_precision = np.eye(X.shape[1], dtype=float)
    prior_precision[0, 0] = 1e-6
    alpha = 1.0 / max(base_variance, 1.0)
    prior_precision *= alpha

    xtwx = X.T @ (X * weights[:, None])
    xtwy = X.T @ (weights * y)
    beta = 1.0 / base_variance

    covariance = np.linalg.pinv(prior_precision + beta * xtwx)
    coefficient_mean = beta * covariance @ xtwy
    residuals = y - (X @ coefficient_mean)
    observation_variance = max(
        float(np.average(residuals**2, weights=weights)),
        base_variance * 0.35,
        1.0,
    )
    beta = 1.0 / observation_variance
    covariance = np.linalg.pinv(prior_precision + beta * xtwx)
    coefficient_mean = beta * covariance @ xtwy

    return BayesianRegressionState(
        feature_columns=MODEL_FEATURE_COLUMNS.copy(),
        numeric_medians=numeric_medians,
        numeric_means=numeric_means,
        numeric_scales=numeric_scales,
        positions=positions,
        coefficient_mean=coefficient_mean,
        coefficient_covariance=covariance,
        observation_variance=observation_variance,
    )


def _design_matrix(
    frame: pd.DataFrame, state: BayesianRegressionState
) -> np.ndarray:
    parts = [np.ones((len(frame), 1), dtype=float)]
    for column in state.feature_columns:
        values = pd.to_numeric(frame.get(column), errors='coerce').fillna(
            state.numeric_medians.get(column, 0.0)
        )
        centered = (values.to_numpy(dtype=float) - state.numeric_means[column]) / state.numeric_scales[column]
        parts.append(centered[:, None])
    for position in state.positions:
        parts.append((frame['position'].astype(str) == position).astype(float).to_numpy()[:, None])
    return np.hstack(parts)


def build_posterior_projection_table(
    train_history: pd.DataFrame,
    target_frame: pd.DataFrame,
    holdout_year: int,
    replacement_quantile: float = 0.2,
    min_history_seasons: int = 0,
) -> pd.DataFrame:
    """Project a holdout board using posterior mean + uncertainty."""
    if train_history is None or train_history.empty:
        raise ValueError('train_history is empty')
    if target_frame is None or target_frame.empty:
        raise ValueError('target_frame is empty')

    training_examples = build_training_examples(
        train_history, replacement_quantile=replacement_quantile, min_history_seasons=1
    )
    regression_state = fit_bayesian_regression(training_examples)

    target_rows: list[dict[str, Any]] = []
    for row in target_frame.itertuples(index=False):
        features = _player_prior_features(
            train_history=train_history,
            player_name=str(row.Name),
            position=str(row.Position),
            target_season=int(holdout_year),
            replacement_quantile=replacement_quantile,
        )
        if int(features['season_count']) < min_history_seasons:
            continue
        features['actual_points'] = float(row.fantasy_points)
        target_rows.append(features)

    if not target_rows:
        raise ValueError(f'No overlapping players to evaluate in {holdout_year}')

    frame = pd.DataFrame(target_rows)

    if regression_state is None:
        regression_mean = frame['prior_mean'].to_numpy(dtype=float)
        regression_var = np.square(np.maximum(frame['prior_std'].to_numpy(dtype=float), 6.0))
    else:
        X = _design_matrix(frame, regression_state)
        regression_mean = X @ regression_state.coefficient_mean
        regression_var = regression_state.observation_variance + np.einsum(
            'ij,jk,ik->i',
            X,
            regression_state.coefficient_covariance,
            X,
        )

    prior_var = np.square(np.maximum(frame['prior_std'].to_numpy(dtype=float), 6.0))
    posterior_var = 1.0 / (
        (1.0 / np.maximum(prior_var, 1.0))
        + (1.0 / np.maximum(regression_var, 1.0))
    )
    posterior_mean = posterior_var * (
        frame['prior_mean'].to_numpy(dtype=float) / np.maximum(prior_var, 1.0)
        + regression_mean / np.maximum(regression_var, 1.0)
    )
    posterior_std = np.sqrt(np.maximum(posterior_var, 1.0))

    frame['regression_mean'] = regression_mean
    frame['regression_std'] = np.sqrt(np.maximum(regression_var, 1.0))
    frame['posterior_mean'] = posterior_mean
    frame['posterior_std'] = posterior_std
    frame['posterior_floor'] = posterior_mean - 1.2816 * posterior_std
    frame['posterior_ceiling'] = posterior_mean + 1.2816 * posterior_std
    frame['posterior_prob_beats_replacement'] = norm.cdf(
        (posterior_mean - frame['replacement_baseline'].to_numpy(dtype=float))
        / np.maximum(posterior_std, 1.0)
    )
    frame['uncertainty_score'] = np.clip(
        posterior_std / np.maximum(np.abs(posterior_mean), 1.0), 0.0, 1.0
    )
    frame['adp'] = frame['last_adp']
    frame['adp_rank'] = frame['last_adp_rank']
    frame['adp_std'] = np.maximum(2.0, frame['posterior_std'] * 0.18)
    frame['vor_value'] = frame['posterior_mean'] - frame['replacement_baseline']
    frame['proj_points_mean'] = frame['posterior_mean']
    frame['proj_points_floor'] = frame['posterior_floor']
    frame['proj_points_ceiling'] = frame['posterior_ceiling']
    frame['std_projection'] = frame['posterior_std']
    frame['games_missed'] = frame['games_missed_mean']
    frame['team_change'] = frame['team_change_rate']
    frame['role_volatility'] = frame['role_volatility']
    frame['site_disagreement'] = frame['site_disagreement']
    frame['source_name'] = f'bayesian_posterior_holdout_{holdout_year}'
    frame['source_updated_at'] = pd.Timestamp(holdout_year, 1, 1)
    return frame
