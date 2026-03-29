from __future__ import annotations

import pandas as pd
import pytest

import ffbayes.data_pipeline.collect_data as collect_data


def _make_merged_frame(*, team_column: str, season: int = 2025) -> pd.DataFrame:
    rows = [
        {
            'player_id': 'p1',
            'player_display_name': 'Alpha Player',
            'season': season,
            'position': 'RB',
            'week': 1,
            'fantasy_points': 12.5,
            'fantasy_points_ppr': 15.0,
            'gameday': '2025-09-01',
            team_column: 'NYG',
            'home_team': 'NYG',
            'away_team': 'DAL',
        },
        {
            'player_id': 'p2',
            'player_display_name': 'Beta Player',
            'season': season,
            'position': 'WR',
            'week': 1,
            'fantasy_points': 8.0,
            'fantasy_points_ppr': 9.5,
            'gameday': '2025-09-01',
            team_column: 'DAL',
            'home_team': 'NYG',
            'away_team': 'DAL',
        },
    ]
    return pd.DataFrame(rows)


def _make_minimal_required_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'player_id': 'p1',
                'player_display_name': 'Alpha Player',
                'season': 2025,
                'position': 'RB',
                'week': 1,
                'fantasy_points': 12.5,
                'fantasy_points_ppr': 15.0,
                'gameday': '2025-09-01',
                'home_team': 'NYG',
                'away_team': 'DAL',
            }
        ]
    )


def test_process_dataset_supports_recent_team_and_home_away_logic():
    merged_df = _make_merged_frame(team_column='recent_team')

    processed_df = collect_data.process_dataset(merged_df, 2025)

    assert list(processed_df.columns) == collect_data.PROCESS_DATASET_OUTPUT_COLUMNS
    assert len(processed_df) == 2
    assert list(processed_df['Tm']) == ['NYG', 'DAL']
    assert list(processed_df['Opp']) == ['DAL', 'NYG']
    assert list(processed_df['is_home']) == [1, 0]


def test_process_dataset_supports_player_team_fallback():
    merged_df = _make_merged_frame(team_column='player_team')
    merged_df = merged_df.drop(columns=['recent_team'], errors='ignore')

    processed_df = collect_data.process_dataset(merged_df, 2025)

    assert list(processed_df.columns) == collect_data.PROCESS_DATASET_OUTPUT_COLUMNS
    assert len(processed_df) == 2
    assert list(processed_df['Tm']) == ['NYG', 'DAL']


def test_process_dataset_requires_one_team_column():
    merged_df = _make_minimal_required_frame()

    with pytest.raises(ValueError, match='recent_team or player_team'):
        collect_data.process_dataset(merged_df, 2025)


def test_collect_data_by_year_skips_empty_processed_frames(tmp_path, monkeypatch):
    season_dir = tmp_path / 'season_datasets'
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(
        collect_data, 'check_data_availability', lambda year: (True, 1)
    )
    monkeypatch.setattr(
        collect_data,
        'create_dataset',
        lambda year: _make_merged_frame(team_column='recent_team', season=year),
    )
    monkeypatch.setattr(
        collect_data,
        'process_dataset',
        lambda final_df, year: pd.DataFrame(columns=collect_data.PROCESS_DATASET_OUTPUT_COLUMNS),
    )

    result = collect_data.collect_data_by_year(2024)

    assert result is None
    assert not (season_dir / '2024season.csv').exists()
