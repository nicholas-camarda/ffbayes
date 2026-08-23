from __future__ import annotations

import pandas as pd

from ffbayes.analysis.bayesian_player_model import (
    MODEL_FEATURE_COLUMNS,
    _player_prior_features,
    aggregate_season_player_table,
    build_posterior_projection_table,
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


def test_aggregate_season_player_table_preserves_points_and_is_order_invariant():
    history = pd.DataFrame(
        [
            {
                'Season': 2025,
                'Name': 'Alpha RB',
                'Position': 'RB',
                'FantPt': 10.0,
                'FantPtPPR': 12.0,
                'Tm': 'NYG',
            },
            {
                'Season': 2025,
                'Name': 'Alpha RB',
                'Position': 'RB',
                'FantPt': 20.0,
                'FantPtPPR': 22.0,
                'Tm': 'NYG',
            },
            {
                'Season': 2025,
                'Name': 'Beta WR',
                'Position': 'WR',
                'FantPt': 15.0,
                'FantPtPPR': 18.0,
                'Tm': 'DAL',
            },
        ]
    )

    aggregated = aggregate_season_player_table(
        history, feature_history=pd.DataFrame()
    )
    reordered = aggregate_season_player_table(
        history.iloc[[2, 0, 1]].reset_index(drop=True), feature_history=pd.DataFrame()
    )

    alpha = aggregated.set_index(['Season', 'Name', 'Position']).loc[
        (2025, 'Alpha RB', 'RB')
    ]
    assert alpha['fantasy_points'] == 30.0
    assert alpha['fantasy_points_rate'] == 15.0
    assert alpha['fantasy_points_ppr'] == 34.0
    assert alpha['games_played'] == 2
    assert alpha['games_missed'] == 15.0
    pd.testing.assert_frame_equal(
        aggregated.sort_index(axis=1), reordered.sort_index(axis=1), check_dtype=False
    )


def test_posterior_projection_does_not_use_holdout_actual_points():
    train_history = pd.DataFrame(
        [
            {
                'Season': season,
                'Name': name,
                'Position': position,
                'FantPt': points,
            }
            for season, alpha, beta in [
                (2022, 140.0, 100.0),
                (2023, 150.0, 110.0),
                (2024, 160.0, 120.0),
            ]
            for name, position, points in [
                ('Alpha RB', 'RB', alpha),
                ('Beta RB', 'RB', beta),
            ]
        ]
    )
    train_table = aggregate_season_player_table(
        train_history, feature_history=pd.DataFrame()
    )
    target = pd.DataFrame(
        {
            'Name': ['Alpha RB', 'Beta RB'],
            'Position': ['RB', 'RB'],
            'fantasy_points': [170.0, 130.0],
        }
    )
    altered_target = target.assign(fantasy_points=[17000.0, 13000.0])

    first = build_posterior_projection_table(
        train_table, target, holdout_year=2025, min_history_seasons=0
    )
    second = build_posterior_projection_table(
        train_table, altered_target, holdout_year=2025, min_history_seasons=0
    )

    projection_columns = [
        'player_name',
        'posterior_mean',
        'posterior_std',
        'posterior_floor',
        'posterior_ceiling',
        'posterior_prob_beats_replacement',
    ]
    pd.testing.assert_frame_equal(
        first[projection_columns], second[projection_columns], check_dtype=False
    )
    assert first['actual_points'].tolist() == [170.0, 130.0]
    assert second['actual_points'].tolist() == [17000.0, 13000.0]


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
