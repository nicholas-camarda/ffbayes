from __future__ import annotations

import warnings

import pandas as pd

from ffbayes.analysis.advanced_stats_calculator import (
    calculate_advanced_stats_from_existing_data,
)


def _sample_advanced_stats_input() -> pd.DataFrame:
    rows: list[dict] = []
    players = [
        ('Alpha QB', 'QB', 'NYG'),
        ('Beta RB', 'RB', 'DAL'),
        ('Gamma WR', 'WR', 'PHI'),
        ('Delta TE', 'TE', 'BUF'),
    ]
    for season in [2023, 2024]:
        for name, position, team in players:
            for game in range(1, 9):
                rows.append(
                    {
                        'Name': name,
                        'Position': position,
                        'Season': season,
                        'G#': game,
                        'FantPt': 10.0 + game + (season - 2023),
                        'Tm': team,
                        'Opp': 'MIA' if game % 2 else 'NE',
                        'Date': f'{season}-09-{game:02d}',
                        '7_game_avg': 12.5 + game / 10.0,
                        'is_home': game % 2 == 1,
                    }
                )
    return pd.DataFrame(rows)


def test_advanced_stats_calculator_avoids_groupby_apply_deprecation_warning():
    base_data = _sample_advanced_stats_input()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        result = calculate_advanced_stats_from_existing_data(base_data)

    messages = [str(item.message) for item in recorded]
    assert not any('DataFrameGroupBy.apply operated on the grouping columns' in msg for msg in messages)
    for column in [
        'boom_bust_ratio',
        'floor_ceiling_spread',
        'early_late_season_diff',
        'wr_big_play_dependency',
        'te_usage_reliability',
        'recent_form',
        'season_trend',
        'consistency_over_time',
    ]:
        assert column in result.columns
