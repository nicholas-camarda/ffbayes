from __future__ import annotations

from pathlib import Path

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


def test_process_dataset_suppresses_row_progress_in_summary_mode(
    monkeypatch, capsys
):
    monkeypatch.setenv('FFBAYES_PROCESS_DATASET_PROGRESS', 'summary')
    monkeypatch.setattr(
        collect_data,
        'add_player_rankings',
        lambda df, year: df,
    )

    merged_df = _make_merged_frame(team_column='recent_team')
    processed_df = collect_data.process_dataset(merged_df, 2025)
    output = capsys.readouterr().out

    assert len(processed_df) == 2
    assert 'Processed 0/2 rows' not in output
    assert 'Processing summary for 2025' in output


def test_create_dataset_does_not_attach_vor_rankings(monkeypatch):
    players = pd.DataFrame(
        [
            {
                'player_id': 'p1',
                'player_display_name': 'Alpha Player',
                'position': 'RB',
                'recent_team': 'NYG',
                'season': 2025,
                'week': 1,
                'season_type': 'REG',
                'fantasy_points': 12.5,
                'fantasy_points_ppr': 15.0,
                'game_injury_report_status': 'ACTIVE',
                'practice_injury_report_status': 'FULL',
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                'game_id': '2025_01_NYG_DAL',
                'week': 1,
                'season': 2025,
                'gameday': '2025-09-01',
                'home_team': 'NYG',
                'away_team': 'DAL',
                'away_score': 10,
                'home_score': 17,
            }
        ]
    )

    monkeypatch.setattr(
        collect_data.nflverse_backend,
        'load_weekly_player_stats',
        lambda years: players,
    )
    monkeypatch.setattr(
        collect_data.nflverse_backend,
        'load_schedules',
        lambda years: schedules,
    )
    monkeypatch.setattr(
        collect_data.nflverse_backend,
        'load_weekly_defense_stats',
        lambda years: pd.DataFrame(),
    )
    monkeypatch.setattr(
        collect_data,
        'add_player_rankings',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('add_player_rankings should not be called')
        ),
    )

    merged_df = collect_data.create_dataset(2025)

    assert 'rank' not in merged_df.columns
    assert 'Name' in merged_df.columns


def test_collect_nfl_data_reuses_cached_season_files(
    tmp_path, monkeypatch, capsys
):
    active_raw = tmp_path / 'active' / 'data' / 'raw' / 'season_datasets'
    legacy_raw = tmp_path / 'legacy' / 'data' / 'raw' / 'season_datasets'
    active_processed = tmp_path / 'active' / 'data' / 'processed'
    active_raw.mkdir(parents=True, exist_ok=True)
    legacy_raw.mkdir(parents=True, exist_ok=True)
    active_processed.mkdir(parents=True, exist_ok=True)

    cached_file = legacy_raw / '2025season.csv'
    pd.DataFrame(
        [
            {
                'G#': 1,
                'Date': '2025-09-01',
                'Tm': 'NYG',
                'Away': 'DAL',
                'Opp': 'DAL',
                'FantPt': 12.5,
                'FantPtPPR': 15.0,
                'Name': 'Alpha Player',
                'PlayerID': 'p1',
                'Position': 'RB',
                'Season': 2025,
                'GameInjuryStatus': None,
                'PracticeInjuryStatus': None,
                'is_home': 1,
            }
        ]
    ).to_csv(cached_file, index=False)

    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', active_raw)
    monkeypatch.setattr(
        collect_data, 'LEGACY_SEASON_DATASETS_DIR', legacy_raw
    )
    monkeypatch.setattr(collect_data, 'RAW_DATA_DIR', tmp_path / 'active' / 'data' / 'raw')
    monkeypatch.setattr(
        collect_data, 'COMBINED_DATASETS_DIR', active_processed / 'combined_datasets'
    )

    class DummyMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def start_timer(self):
            return None

        class _Context:
            def __init__(self, total, title):
                self.total = total

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def monitor(self, total, title):
            return self._Context(total, title)

    monkeypatch.setattr(collect_data, 'ProgressMonitor', DummyMonitor)
    monkeypatch.setattr(
        collect_data,
        'check_data_availability',
        lambda year: (_ for _ in ()).throw(
            AssertionError('network availability check should not run')
        ),
    )
    monkeypatch.setattr(
        collect_data,
        'collect_data_by_year',
        lambda year: (_ for _ in ()).throw(
            AssertionError('year collection should not run')
        ),
    )
    monkeypatch.setattr(
        collect_data,
        'combine_datasets',
        lambda directory_path, output_directory_path, years_to_process: pd.DataFrame(
            {'year': years_to_process}
        ),
    )

    successful_years = collect_data.collect_nfl_data(
        years=[2025], allow_stale_latest=True
    )
    output = capsys.readouterr().out

    assert successful_years == [2025]
    assert (active_raw / '2025season.csv').exists()
    assert 'Seeded cached season file for 2025' in output
    assert 'cached season file already exists' in output
