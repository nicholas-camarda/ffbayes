#!/usr/bin/env python3
"""
Bayesian vs VOR comparison

Backtests a simple Bayesian shrinkage forecaster against a simple VOR-style
ranking baseline using historical season data. The goal is to answer the
project question directly: does the Bayesian approach order players better
than a naive VOR ranking when we evaluate against held-out fantasy points?
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ffbayes.utils.path_constants import SEASON_DATASETS_DIR, get_results_dir
from ffbayes.utils.validation_metrics import calculate_model_accuracy_metrics


def load_season_history(data_directory: Path | str | None = None) -> pd.DataFrame:
    """Load per-game season history and combine into a single frame."""
    season_dir = Path(data_directory) if data_directory is not None else SEASON_DATASETS_DIR
    files = sorted(season_dir.glob('*season.csv'))
    if not files:
        raise FileNotFoundError(f'No season CSVs found in {season_dir}')
    return pd.concat((pd.read_csv(file_path) for file_path in files), ignore_index=True)


def build_season_player_table(history: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-game data into a per-player-season table."""
    required_columns = {'Season', 'Name', 'Position', 'FantPt'}
    missing = required_columns.difference(history.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    return (
        history.groupby(['Season', 'Name', 'Position'], as_index=False)['FantPt']
        .mean()
        .rename(columns={'FantPt': 'fantasy_points'})
    )


def _position_replacement_level(train_season: pd.DataFrame, position: str, quantile: float) -> float:
    """Approximate a replacement-level baseline for a position."""
    pos_values = train_season.loc[train_season['Position'] == position, 'fantasy_points'].dropna()
    if pos_values.empty:
        overall = train_season['fantasy_points'].dropna()
        return float(overall.mean()) if not overall.empty else 0.0
    return float(pos_values.quantile(quantile))


def _predict_bayesian_score(
    train_history: pd.DataFrame,
    player_name: str,
    position: str,
    shrinkage_strength: float,
    trend_weight: float,
) -> float:
    """Empirical Bayes-style shrinkage forecast for one player."""
    player_hist = train_history[train_history['Name'] == player_name].sort_values('Season')
    position_mean = float(
        train_history.loc[train_history['Position'] == position, 'fantasy_points'].mean()
        if not train_history.empty
        else 0.0
    )

    if player_hist.empty:
        return position_mean

    player_mean = float(player_hist['fantasy_points'].mean())
    n_seasons = len(player_hist)
    bayes_score = (n_seasons / (n_seasons + shrinkage_strength)) * player_mean
    bayes_score += (shrinkage_strength / (n_seasons + shrinkage_strength)) * position_mean

    if n_seasons >= 2:
        trend = float(player_hist.iloc[-1]['fantasy_points'] - player_hist.iloc[0]['fantasy_points'])
        bayes_score += trend_weight * trend / max(n_seasons - 1, 1)

    return bayes_score


def _predict_vor_scores(
    train_history: pd.DataFrame,
    player_name: str,
    position: str,
    replacement_quantile: float,
) -> tuple[float, float]:
    """Return a VOR-style point forecast and rank score."""
    player_hist = train_history[train_history['Name'] == player_name].sort_values('Season')
    position_replacement = _position_replacement_level(train_history, position, replacement_quantile)

    if player_hist.empty:
        return position_replacement, 0.0

    last_season_mean = float(player_hist.iloc[-1]['fantasy_points'])
    rank_score = last_season_mean - position_replacement
    return last_season_mean, rank_score


def evaluate_holdout_season(
    season_table: pd.DataFrame,
    holdout_year: int,
    shrinkage_strength: float = 2.0,
    trend_weight: float = 0.15,
    replacement_quantile: float = 0.2,
    top_k: int = 24,
    min_history_seasons: int = 0,
) -> dict:
    """Evaluate Bayesian vs VOR on one holdout season."""
    train = season_table[season_table['Season'] < holdout_year].copy()
    test = season_table[season_table['Season'] == holdout_year].copy()

    if train.empty:
        raise ValueError(f'No training data available before {holdout_year}')
    if test.empty:
        raise ValueError(f'No holdout data available for {holdout_year}')

    test = test.copy()
    test['player_key'] = test.apply(
        lambda row: f"{row['Name']}||{row['Position']}",
        axis=1,
    )

    if min_history_seasons > 0:
        train_history_counts = train.groupby(['Name', 'Position'])['Season'].nunique()
        eligible_keys = {
            f"{name}||{position}"
            for (name, position), count in train_history_counts.items()
            if count >= min_history_seasons
        }
        test = test[test['player_key'].isin(eligible_keys)].copy()

    truth = test.set_index('player_key')['fantasy_points']

    bayesian_point = {}
    vor_point = {}
    bayesian_rank = {}
    vor_rank = {}

    for _, row in test.iterrows():
        name = str(row['Name'])
        position = str(row['Position'])
        player_key = str(row['player_key'])

        bayes_score = _predict_bayesian_score(
            train_history=train,
            player_name=name,
            position=position,
            shrinkage_strength=shrinkage_strength,
            trend_weight=trend_weight,
        )
        vor_point_score, vor_rank_score = _predict_vor_scores(
            train_history=train,
            player_name=name,
            position=position,
            replacement_quantile=replacement_quantile,
        )

        bayesian_point[player_key] = bayes_score
        vor_point[player_key] = vor_point_score
        bayesian_rank[player_key] = bayes_score
        vor_rank[player_key] = vor_rank_score

    common = truth.index.intersection(pd.Index(bayesian_point.keys()))
    if common.empty:
        raise ValueError(f'No overlapping players to evaluate in {holdout_year}')

    truth = truth.loc[common]
    bayesian_point_series = pd.Series(bayesian_point).loc[common]
    vor_point_series = pd.Series(vor_point).loc[common]
    bayesian_rank_series = pd.Series(bayesian_rank).loc[common]
    vor_rank_series = pd.Series(vor_rank).loc[common]

    bayesian_point_metrics = calculate_model_accuracy_metrics(
        bayesian_point_series.to_numpy(),
        truth.to_numpy(),
    )
    vor_point_metrics = calculate_model_accuracy_metrics(
        vor_point_series.to_numpy(),
        truth.to_numpy(),
    )

    bayesian_spearman = float(spearmanr(bayesian_rank_series, truth).correlation)
    vor_spearman = float(spearmanr(vor_rank_series, truth).correlation)
    bayesian_spearman = float(np.nan_to_num(bayesian_spearman, nan=0.0))
    vor_spearman = float(np.nan_to_num(vor_spearman, nan=0.0))

    ranked_truth = test.set_index('player_key').loc[common].copy()
    ranked_truth['bayesian_score'] = bayesian_rank_series
    ranked_truth['vor_score'] = vor_rank_series
    top_bayesian = ranked_truth.nlargest(top_k, 'bayesian_score')['fantasy_points'].mean()
    top_vor = ranked_truth.nlargest(top_k, 'vor_score')['fantasy_points'].mean()

    return {
        'holdout_year': holdout_year,
        'num_players_evaluated': int(len(common)),
        'bayesian': {
            **bayesian_point_metrics,
            'spearman_rank_correlation': bayesian_spearman,
            'top_k_mean_actual': float(top_bayesian),
        },
        'vor': {
            **vor_point_metrics,
            'spearman_rank_correlation': vor_spearman,
            'top_k_mean_actual': float(top_vor),
        },
        'winner': {
            'mae': 'bayesian' if bayesian_point_metrics['mae'] <= vor_point_metrics['mae'] else 'vor',
            'rank_correlation': 'bayesian' if bayesian_spearman >= vor_spearman else 'vor',
            'top_k_mean_actual': 'bayesian' if top_bayesian >= top_vor else 'vor',
        },
        'improvement': {
            'mae_delta': float(vor_point_metrics['mae'] - bayesian_point_metrics['mae']),
            'spearman_delta': float(bayesian_spearman - vor_spearman),
            'top_k_mean_actual_delta': float(top_bayesian - top_vor),
        },
    }


def _season_backtest_years(season_table: pd.DataFrame, min_train_seasons: int = 3) -> list[int]:
    seasons = sorted(int(season) for season in season_table['Season'].unique())
    return seasons[min_train_seasons:]


def run_backtest(
    data_directory: Path | str | None = None,
    output_dir: Path | str | None = None,
    holdout_years: Iterable[int] | None = None,
    shrinkage_strength: float = 2.0,
    trend_weight: float = 0.15,
    replacement_quantile: float = 0.2,
    top_k: int = 24,
    min_history_seasons: int = 0,
) -> dict:
    """Run the Bayesian vs VOR backtest across all eligible holdout seasons."""
    history = load_season_history(data_directory)
    season_table = build_season_player_table(history)

    if holdout_years is None:
        holdout_years = _season_backtest_years(season_table)

    holdout_years = list(holdout_years)
    if not holdout_years:
        raise ValueError('No holdout seasons available for backtest')

    by_season = []
    for holdout_year in holdout_years:
        by_season.append(
            evaluate_holdout_season(
                season_table=season_table,
                holdout_year=holdout_year,
                shrinkage_strength=shrinkage_strength,
                trend_weight=trend_weight,
                replacement_quantile=replacement_quantile,
                top_k=top_k,
                min_history_seasons=min_history_seasons,
            )
        )

    bayesian_mae = float(np.mean([item['bayesian']['mae'] for item in by_season]))
    vor_mae = float(np.mean([item['vor']['mae'] for item in by_season]))
    bayesian_rho = float(np.nanmean([item['bayesian']['spearman_rank_correlation'] for item in by_season]))
    vor_rho = float(np.nanmean([item['vor']['spearman_rank_correlation'] for item in by_season]))
    bayesian_top_k = float(np.mean([item['bayesian']['top_k_mean_actual'] for item in by_season]))
    vor_top_k = float(np.mean([item['vor']['top_k_mean_actual'] for item in by_season]))
    season_win_counts = {
        metric: sum(item['winner'][metric] == 'bayesian' for item in by_season)
        for metric in ['mae', 'rank_correlation', 'top_k_mean_actual']
    }

    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'bayesian_vs_vor_backtest',
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
            'winner': {
                'mae': 'bayesian' if bayesian_mae <= vor_mae else 'vor',
                'rank_correlation': 'bayesian' if bayesian_rho >= vor_rho else 'vor',
                'top_k_mean_actual': 'bayesian' if bayesian_top_k >= vor_top_k else 'vor',
            },
            'improvement': {
                'mae_delta': float(vor_mae - bayesian_mae),
                'spearman_delta': float(bayesian_rho - vor_rho),
                'top_k_mean_actual_delta': float(bayesian_top_k - vor_top_k),
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
        run_label = f"{min(holdout_years)}_to_{max(holdout_years)}"

    output_path = output_dir / f'bayesian_vs_vor_backtest_{run_label}.json'
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    summary['output_path'] = str(output_path)

    summary['plot_path'] = str(_write_summary_plot(summary, output_dir, run_label))
    return summary


def _write_summary_plot(summary: dict, output_dir: Path, run_label: str) -> Path:
    """Write a simple scorecard plot for quick inspection."""
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

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle('Bayesian vs VOR Backtest')

    for ax, label, bayes_value, vor_value in zip(axes, labels, bayesian_values, vor_values):
        ax.bar(['Bayesian', 'VOR'], [bayes_value, vor_value], color=['#1f77b4', '#ff7f0e'])
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.2)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(plot_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def main() -> int:
    """CLI entrypoint for the Bayesian vs VOR comparison."""
    print('Running Bayesian vs VOR backtest...')
    print('Scope: players with at least 2 prior seasons, top 12 draft targets')
    summary = run_backtest(top_k=12, min_history_seasons=2)

    print(f"Evaluated seasons: {summary['seasons_evaluated']}")
    print(
        'Bayesian rank correlation:',
        f"{summary['overall']['bayesian']['spearman_rank_correlation']:.3f}",
    )
    print(
        'VOR rank correlation:',
        f"{summary['overall']['vor']['spearman_rank_correlation']:.3f}",
    )
    print(f"Scorecard written to: {summary['output_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
