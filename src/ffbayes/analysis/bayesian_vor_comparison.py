#!/usr/bin/env python3
"""
Bayesian vs VOR comparison.

Backtests a posterior player model against frozen simple baselines using
historical season data. The research goal is explicit:

* forecast target: held-out season fantasy points
* decision target: which players surface as the strongest draft candidates

The Bayesian model combines player-level partial pooling with a population
regression trained on lagged, draft-time-safe features assembled from local
historical data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ffbayes.analysis.bayesian_player_model import (
    aggregate_season_player_table,
    build_posterior_projection_table,
)
from ffbayes.utils.path_constants import SEASON_DATASETS_DIR, get_results_dir
from ffbayes.utils.validation_metrics import calculate_model_accuracy_metrics


def load_season_history(data_directory: Path | str | None = None) -> pd.DataFrame:
    """Load historical season CSVs into one frame."""
    season_dir = Path(data_directory) if data_directory is not None else SEASON_DATASETS_DIR
    files = sorted(season_dir.glob('*season.csv'))
    if not files:
        raise FileNotFoundError(f'No season CSVs found in {season_dir}')
    return pd.concat((pd.read_csv(file_path) for file_path in files), ignore_index=True)


def build_season_player_table(history: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-game data into a season-level modeling table."""
    return aggregate_season_player_table(history)


def _position_metrics(
    frame: pd.DataFrame, score_column: str, point_column: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, sub in frame.groupby('Position'):
        truth = pd.to_numeric(sub['fantasy_points'], errors='coerce')
        pred = pd.to_numeric(sub[point_column], errors='coerce')
        score = pd.to_numeric(sub[score_column], errors='coerce')
        valid = truth.notna() & pred.notna() & score.notna()
        if valid.sum() < 2:
            continue
        truth_valid = truth[valid]
        pred_valid = pred[valid]
        score_valid = score[valid]
        metrics = calculate_model_accuracy_metrics(
            pred_valid.to_numpy(), truth_valid.to_numpy()
        )
        rank_corr = float(
            np.nan_to_num(spearmanr(score_valid, truth_valid).correlation, nan=0.0)
        )
        rmse = float(
            np.sqrt(np.mean(np.square(truth_valid.to_numpy() - pred_valid.to_numpy())))
        )
        rows.append(
            {
                'position': str(position),
                'player_count': int(valid.sum()),
                'mae': float(metrics['mae']),
                'rmse': rmse,
                'spearman_rank_correlation': rank_corr,
            }
        )
    return rows


def _coverage_rate(
    truth: pd.Series, mean: pd.Series, std: pd.Series, z_value: float
) -> float:
    lower = mean - z_value * std
    upper = mean + z_value * std
    return float(((truth >= lower) & (truth <= upper)).mean())


def _evaluate_method(
    frame: pd.DataFrame,
    *,
    score_column: str,
    point_column: str,
    label: str,
    top_k: int,
    std_column: str | None = None,
) -> dict[str, Any]:
    truth = pd.to_numeric(frame['fantasy_points'], errors='coerce')
    point_pred = pd.to_numeric(frame[point_column], errors='coerce')
    score_pred = pd.to_numeric(frame[score_column], errors='coerce')
    valid = truth.notna() & point_pred.notna() & score_pred.notna()
    if valid.sum() == 0:
        raise ValueError(f'No valid rows available to score method {label}')

    truth_valid = truth[valid]
    point_valid = point_pred[valid]
    score_valid = score_pred[valid]
    metrics = calculate_model_accuracy_metrics(
        point_valid.to_numpy(), truth_valid.to_numpy()
    )
    rank_corr = float(np.nan_to_num(spearmanr(score_valid, truth_valid).correlation, nan=0.0))

    ranked = frame.loc[valid].copy()
    ranked['_score'] = score_valid.to_numpy()
    top_actual = float(ranked.nlargest(top_k, '_score')['fantasy_points'].mean())

    payload = {
        **metrics,
        'spearman_rank_correlation': rank_corr,
        'top_k_mean_actual': top_actual,
        'by_position': _position_metrics(
            frame.loc[valid].copy(),
            score_column=score_column,
            point_column=point_column,
        ),
    }

    if std_column is not None and std_column in frame.columns:
        std_pred = pd.to_numeric(frame.loc[valid, std_column], errors='coerce').fillna(0.0)
        payload['calibration'] = {
            'interval_50_coverage': _coverage_rate(
                truth_valid, point_valid, std_pred, 0.6745
            ),
            'interval_80_coverage': _coverage_rate(
                truth_valid, point_valid, std_pred, 1.2816
            ),
            'mean_predictive_std': float(std_pred.mean()),
        }
    return payload


def _top_disagreements(frame: pd.DataFrame, top_n: int = 15) -> list[dict[str, Any]]:
    disagreements = frame.copy()
    disagreements['bayesian_rank'] = disagreements['posterior_mean'].rank(
        method='first', ascending=False
    )
    disagreements['vor_rank'] = disagreements['historical_vor_proxy_score'].rank(
        method='first', ascending=False
    )
    disagreements['market_rank'] = disagreements['market_proxy_score'].rank(
        method='first', ascending=False
    )
    disagreements['abs_rank_gap_vs_vor'] = (
        disagreements['bayesian_rank'] - disagreements['vor_rank']
    ).abs()
    disagreements['abs_rank_gap_vs_market'] = (
        disagreements['bayesian_rank'] - disagreements['market_rank']
    ).abs()
    return (
        disagreements.sort_values(
            ['abs_rank_gap_vs_vor', 'posterior_mean'], ascending=[False, False]
        )
        .head(top_n)[
            [
                'player_name',
                'Position',
                'fantasy_points',
                'posterior_mean',
                'posterior_std',
                'historical_vor_proxy_point',
                'historical_vor_proxy_score',
                'market_proxy_score',
                'bayesian_rank',
                'vor_rank',
                'market_rank',
                'abs_rank_gap_vs_vor',
                'abs_rank_gap_vs_market',
                'posterior_prob_beats_replacement',
            ]
        ]
        .rename(columns={'Position': 'position'})
        .to_dict(orient='records')
    )


def evaluate_holdout_season(
    season_table: pd.DataFrame,
    holdout_year: int,
    shrinkage_strength: float = 2.0,
    trend_weight: float = 0.15,
    replacement_quantile: float = 0.2,
    top_k: int = 24,
    min_history_seasons: int = 0,
) -> dict[str, Any]:
    """Evaluate Bayesian vs VOR on one holdout season."""
    del shrinkage_strength, trend_weight

    train = season_table[season_table['Season'] < holdout_year].copy()
    test = season_table[season_table['Season'] == holdout_year].copy()
    if train.empty:
        raise ValueError(f'No training data available before {holdout_year}')
    if test.empty:
        raise ValueError(f'No holdout data available for {holdout_year}')

    prediction_table = build_posterior_projection_table(
        train_history=train,
        target_frame=test,
        holdout_year=holdout_year,
        replacement_quantile=replacement_quantile,
        min_history_seasons=min_history_seasons,
    )
    prediction_table = prediction_table.rename(
        columns={'position': 'Position', 'actual_points': 'fantasy_points'}
    )

    bayesian_metrics = _evaluate_method(
        prediction_table,
        score_column='posterior_mean',
        point_column='posterior_mean',
        std_column='posterior_std',
        label='bayesian',
        top_k=top_k,
    )
    vor_metrics = _evaluate_method(
        prediction_table,
        score_column='historical_vor_proxy_score',
        point_column='historical_vor_proxy_point',
        label='vor',
        top_k=top_k,
    )
    market_metrics = _evaluate_method(
        prediction_table,
        score_column='market_proxy_score',
        point_column='historical_vor_proxy_point',
        label='market_proxy',
        top_k=top_k,
    )

    return {
        'holdout_year': holdout_year,
        'num_players_evaluated': int(len(prediction_table)),
        'artifact_schema_version': 2,
        'comparison_targets': {
            'forecast_target': 'held_out_season_fantasy_points',
            'decision_target': f'top_{top_k}_draft_candidates_mean_actual_points',
        },
        'ablation': {
            'step': 'hierarchical_empirical_bayes',
            'label': 'posterior_mean_with_lagged_features',
        },
        'bayesian': bayesian_metrics,
        'vor': vor_metrics,
        'market': market_metrics,
        'winner': {
            'mae': 'bayesian'
            if bayesian_metrics['mae'] <= vor_metrics['mae']
            else 'vor',
            'rank_correlation': 'bayesian'
            if bayesian_metrics['spearman_rank_correlation']
            >= vor_metrics['spearman_rank_correlation']
            else 'vor',
            'top_k_mean_actual': 'bayesian'
            if bayesian_metrics['top_k_mean_actual'] >= vor_metrics['top_k_mean_actual']
            else 'vor',
        },
        'improvement': {
            'mae_delta': float(vor_metrics['mae'] - bayesian_metrics['mae']),
            'spearman_delta': float(
                bayesian_metrics['spearman_rank_correlation']
                - vor_metrics['spearman_rank_correlation']
            ),
            'top_k_mean_actual_delta': float(
                bayesian_metrics['top_k_mean_actual'] - vor_metrics['top_k_mean_actual']
            ),
            'market_mae_delta': float(
                market_metrics['mae'] - bayesian_metrics['mae']
            ),
            'market_spearman_delta': float(
                bayesian_metrics['spearman_rank_correlation']
                - market_metrics['spearman_rank_correlation']
            ),
        },
        'top_disagreements': _top_disagreements(prediction_table),
    }


def run_backtest(
    data_directory: Path | str | None = None,
    output_dir: Path | str | None = None,
    holdout_years: Iterable[int] | None = None,
    shrinkage_strength: float = 2.0,
    trend_weight: float = 0.15,
    replacement_quantile: float = 0.2,
    top_k: int = 24,
    min_history_seasons: int = 0,
) -> dict[str, Any]:
    """Run the Bayes-vs-VOR backtest across all eligible holdout seasons."""
    history = load_season_history(data_directory)
    season_table = build_season_player_table(history)

    seasons = sorted(int(season) for season in season_table['Season'].dropna().unique())
    if holdout_years is None:
        holdout_years = seasons[3:] if len(seasons) > 3 else seasons[1:]
    holdout_years = list(int(year) for year in holdout_years)
    if not holdout_years:
        raise ValueError('No holdout seasons available for backtest')

    by_season = [
        evaluate_holdout_season(
            season_table=season_table,
            holdout_year=holdout_year,
            shrinkage_strength=shrinkage_strength,
            trend_weight=trend_weight,
            replacement_quantile=replacement_quantile,
            top_k=top_k,
            min_history_seasons=min_history_seasons,
        )
        for holdout_year in holdout_years
    ]

    bayesian_mae = float(np.mean([item['bayesian']['mae'] for item in by_season]))
    vor_mae = float(np.mean([item['vor']['mae'] for item in by_season]))
    market_mae = float(np.mean([item['market']['mae'] for item in by_season]))
    bayesian_rho = float(
        np.nanmean([item['bayesian']['spearman_rank_correlation'] for item in by_season])
    )
    vor_rho = float(
        np.nanmean([item['vor']['spearman_rank_correlation'] for item in by_season])
    )
    market_rho = float(
        np.nanmean([item['market']['spearman_rank_correlation'] for item in by_season])
    )
    bayesian_top_k = float(np.mean([item['bayesian']['top_k_mean_actual'] for item in by_season]))
    vor_top_k = float(np.mean([item['vor']['top_k_mean_actual'] for item in by_season]))
    market_top_k = float(np.mean([item['market']['top_k_mean_actual'] for item in by_season]))

    season_win_counts = {
        metric: sum(item['winner'][metric] == 'bayesian' for item in by_season)
        for metric in ['mae', 'rank_correlation', 'top_k_mean_actual']
    }

    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'bayesian_vs_vor_backtest',
        'artifact_schema_version': 2,
        'baselines': ['historical_vor_proxy', 'market_proxy'],
        'comparison_targets': {
            'forecast_target': 'held_out_season_fantasy_points',
            'decision_target': f'top_{top_k}_draft_candidates_mean_actual_points',
        },
        'parameters': {
            'shrinkage_strength': shrinkage_strength,
            'trend_weight': trend_weight,
            'replacement_quantile': replacement_quantile,
            'top_k': top_k,
            'min_history_seasons': min_history_seasons,
        },
        'seasons_evaluated': holdout_years,
        'overall': {
            'bayesian': {
                'mae': bayesian_mae,
                'spearman_rank_correlation': bayesian_rho,
                'top_k_mean_actual': bayesian_top_k,
            },
            'vor': {
                'mae': vor_mae,
                'spearman_rank_correlation': vor_rho,
                'top_k_mean_actual': vor_top_k,
            },
            'market': {
                'mae': market_mae,
                'spearman_rank_correlation': market_rho,
                'top_k_mean_actual': market_top_k,
            },
            'winner': {
                'mae': 'bayesian' if bayesian_mae <= vor_mae else 'vor',
                'rank_correlation': 'bayesian' if bayesian_rho >= vor_rho else 'vor',
                'top_k_mean_actual': 'bayesian'
                if bayesian_top_k >= vor_top_k
                else 'vor',
            },
            'improvement': {
                'mae_delta': float(vor_mae - bayesian_mae),
                'spearman_delta': float(bayesian_rho - vor_rho),
                'top_k_mean_actual_delta': float(bayesian_top_k - vor_top_k),
                'market_mae_delta': float(market_mae - bayesian_mae),
                'market_spearman_delta': float(bayesian_rho - market_rho),
            },
            'season_win_counts': season_win_counts,
        },
        'by_season': by_season,
    }

    if output_dir is None:
        current_year = datetime.now().year
        output_dir = get_results_dir(current_year) / 'model_evaluation'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(holdout_years) == 1:
        run_label = str(holdout_years[0])
    else:
        run_label = f'{min(holdout_years)}_to_{max(holdout_years)}'

    output_path = output_dir / f'bayesian_vs_vor_backtest_{run_label}.json'
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    summary['output_path'] = str(output_path)
    summary['plot_path'] = str(_write_summary_plot(summary, output_dir, run_label))
    return summary


def _write_summary_plot(summary: dict[str, Any], output_dir: Path, run_label: str) -> Path:
    """Write a compact scorecard plot for quick review."""
    plot_dir = output_dir.parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f'bayesian_vs_vor_backtest_{run_label}.png'

    labels = ['MAE', 'Spearman', f'Top {summary["parameters"]["top_k"]} Mean']
    bayesian_values = [
        summary['overall']['bayesian']['mae'],
        summary['overall']['bayesian']['spearman_rank_correlation'],
        summary['overall']['bayesian']['top_k_mean_actual'],
    ]
    vor_values = [
        summary['overall']['vor']['mae'],
        summary['overall']['vor']['spearman_rank_correlation'],
        summary['overall']['vor']['top_k_mean_actual'],
    ]
    market_values = [
        summary['overall']['market']['mae'],
        summary['overall']['market']['spearman_rank_correlation'],
        summary['overall']['market']['top_k_mean_actual'],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Bayesian vs VOR vs Market Backtest')
    for ax, label, bayes_value, vor_value, market_value in zip(
        axes, labels, bayesian_values, vor_values, market_values
    ):
        ax.bar(
            ['Bayesian', 'VOR', 'Market'],
            [bayes_value, vor_value, market_value],
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        )
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.2)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(plot_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def main() -> int:
    """CLI entrypoint for the Bayesian vs VOR comparison."""
    print('Running Bayesian vs VOR backtest...')
    print('Scope: rolling holdout seasons with lagged, posterior player projections')
    summary = run_backtest(top_k=12, min_history_seasons=0)

    print(f"Evaluated seasons: {summary['seasons_evaluated']}")
    print(
        'Bayesian rank correlation:',
        f"{summary['overall']['bayesian']['spearman_rank_correlation']:.3f}",
    )
    print(
        'VOR rank correlation:',
        f"{summary['overall']['vor']['spearman_rank_correlation']:.3f}",
    )
    print(
        'Market rank correlation:',
        f"{summary['overall']['market']['spearman_rank_correlation']:.3f}",
    )
    print(f"Scorecard written to: {summary['output_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
