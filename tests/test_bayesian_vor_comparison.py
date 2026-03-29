import json
import os
import tempfile
from pathlib import Path

import matplotlib
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='matplotlib-'))
matplotlib.use('Agg')

from ffbayes.analysis.bayesian_vor_comparison import (  # noqa: E402
    build_season_player_table,
    evaluate_holdout_season,
    run_backtest,
)


def _build_synthetic_history() -> pd.DataFrame:
    rows = []
    seasons = {
        'Alpha QB': [50, 45, 40, 25],
        'Beta QB': [0, 5, 15, 50],
        'Gamma QB': [20, 20, 20, 20],
        'Delta QB': [8, 8, 8, 8],
    }
    for name, values in seasons.items():
        for season, fant_pt in zip([2021, 2022, 2023, 2024], values, strict=True):
            rows.append(
                {
                    'Season': season,
                    'Name': name,
                    'Position': 'QB',
                    'FantPt': fant_pt,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_holdout_season_prefers_bayesian():
    history = _build_synthetic_history()
    season_table = build_season_player_table(history)

    result = evaluate_holdout_season(
        season_table=season_table,
        holdout_year=2024,
        shrinkage_strength=0.5,
        trend_weight=4.0,
        replacement_quantile=0.5,
        top_k=1,
    )

    assert result['winner']['rank_correlation'] == 'bayesian'
    assert result['winner']['top_k_mean_actual'] == 'bayesian'
    assert result['bayesian']['spearman_rank_correlation'] > result['vor']['spearman_rank_correlation']
    assert result['improvement']['spearman_delta'] > 0
    assert result['improvement']['top_k_mean_actual_delta'] > 0


def test_run_backtest_writes_scorecard_and_plot(monkeypatch, tmp_path):
    history = _build_synthetic_history()
    output_dir = tmp_path / 'model_evaluation'

    monkeypatch.setattr(
        'ffbayes.analysis.bayesian_vor_comparison.load_season_history',
        lambda data_directory=None: history,
    )

    summary = run_backtest(
        output_dir=output_dir,
        holdout_years=[2024],
        shrinkage_strength=0.5,
        trend_weight=4.0,
        replacement_quantile=0.5,
        top_k=1,
    )

    output_path = Path(summary['output_path'])
    plot_path = Path(summary['plot_path'])

    assert output_path.exists()
    assert plot_path.exists()
    assert summary['overall']['winner']['rank_correlation'] == 'bayesian'

    with output_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    assert payload['model_type'] == 'bayesian_vs_vor_backtest'
    assert payload['overall']['winner']['rank_correlation'] == 'bayesian'
    assert payload['overall']['season_win_counts']['rank_correlation'] == 1
