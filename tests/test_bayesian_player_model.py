from __future__ import annotations

import pandas as pd

from ffbayes.analysis.bayesian_player_model import (
    MODEL_FEATURE_COLUMNS,
    _player_prior_features,
    fit_bayesian_regression,
)


def test_player_prior_features_forward_fills_team_changes():
    history = pd.DataFrame(
        [
            {
                'Season': 2022,
                'Name': 'Alpha RB',
                'Position': 'RB',
                'fantasy_points': 12.0,
                'team': 'A',
                'games_played': 17.0,
                'games_missed': 0.0,
                'years_in_league': 1.0,
            },
            {
                'Season': 2023,
                'Name': 'Alpha RB',
                'Position': 'RB',
                'fantasy_points': 13.0,
                'team': None,
                'games_played': 16.0,
                'games_missed': 1.0,
                'years_in_league': 2.0,
            },
            {
                'Season': 2024,
                'Name': 'Alpha RB',
                'Position': 'RB',
                'fantasy_points': 14.0,
                'team': 'B',
                'games_played': 17.0,
                'games_missed': 0.0,
                'years_in_league': 3.0,
            },
            {
                'Season': 2024,
                'Name': 'Beta RB',
                'Position': 'RB',
                'fantasy_points': 8.0,
                'team': 'C',
                'games_played': 17.0,
                'games_missed': 0.0,
                'years_in_league': 1.0,
            },
        ]
    )

    features = _player_prior_features(
        train_history=history,
        player_name='Alpha RB',
        position='RB',
        target_season=2025,
    )

    assert features['team_change_rate'] == 0.5


def test_fit_bayesian_regression_records_model_diagnostics():
    rows = []
    for idx, season in enumerate([2022, 2023, 2024, 2025], start=1):
        row = {column: 0.0 for column in MODEL_FEATURE_COLUMNS}
        row.update(
            {
                'prior_mean': 10.0 + idx,
                'recent_mean': 9.5 + idx,
                'latest_points': 9.0 + idx,
                'player_weighted_mean': 9.2 + idx,
                'position_mean': 8.5,
                'position_std': 3.0,
                'replacement_baseline': 7.0,
                'games_played_mean': 16.0,
                'games_missed_mean': 1.0,
                'position': 'RB',
                'target_season': season,
                'target_points': 11.0 + idx,
            }
        )
        rows.append(row)

    state = fit_bayesian_regression(pd.DataFrame(rows))

    assert state is not None
    assert state.model_diagnostics['training_rows'] == 4
    assert state.model_diagnostics['feature_count'] == len(MODEL_FEATURE_COLUMNS)
    assert state.model_diagnostics['position_count'] == 1
    assert state.model_diagnostics['weighted_rmse'] >= 0.0
