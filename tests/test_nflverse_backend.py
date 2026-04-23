from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from ffbayes.data_pipeline import nflverse_backend as backend


class _FakePolarsFrame:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self):
        return self._frame.copy()


def test_load_weekly_player_stats_normalizes_and_merges_injury_fields(
    monkeypatch,
):
    player_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'player_id': 'p1',
                    'player_name': 'Alpha Player',
                    'pos': 'RB',
                    'team': 'NYG',
                    'season': 2025,
                    'week': 1,
                    'game_type': 'REG',
                    'fantasy_points': 12.5,
                    'fantasy_points_ppr': 15.0,
                }
            ]
        )
    )
    injury_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'player_id': 'p1',
                    'season': 2025,
                    'week': 1,
                    'report_status': 'Questionable',
                    'practice_status': 'Limited',
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(
        load_player_stats=lambda seasons, summary_level: player_frame,
        load_injuries=lambda seasons: injury_frame,
    )
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    result = backend.load_weekly_player_stats([2025])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        'player_id',
        'player_display_name',
        'position',
        'recent_team',
        'season',
        'week',
        'season_type',
        'fantasy_points',
        'fantasy_points_ppr',
        'game_injury_report_status',
        'practice_injury_report_status',
    ]
    assert result.loc[0, 'player_display_name'] == 'Alpha Player'
    assert result.loc[0, 'season_type'] == 'REG'
    assert result.loc[0, 'game_injury_report_status'] == 'Questionable'
    assert result.loc[0, 'practice_injury_report_status'] == 'Limited'


def test_load_schedules_normalizes_columns(monkeypatch):
    schedule_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'game_id': '2025_01_NYG_DAL',
                    'week': 1,
                    'season': 2025,
                    'game_date': '2025-09-01',
                    'home_team': 'NYG',
                    'away_team': 'DAL',
                    'home_score': 24,
                    'away_score': 17,
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(
        load_schedules=lambda seasons: schedule_frame,
    )
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    result = backend.load_schedules([2025])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        'game_id',
        'week',
        'season',
        'gameday',
        'home_team',
        'away_team',
        'home_score',
        'away_score',
    ]
    assert result.loc[0, 'gameday'] == '2025-09-01'


def test_load_weekly_defense_stats_normalizes_required_columns(monkeypatch):
    defense_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'team': 'NYG',
                    'week': 1,
                    'season': 2025,
                    'def_sacks': 3,
                    'def_ints': 1,
                    'def_tackles_combined': 48,
                    'def_missed_tackles': 5,
                    'def_pressures': 12,
                    'def_times_hitqb': 7,
                    'def_times_hurried': 4,
                    'def_times_blitzed': 6,
                    'def_yards_allowed': 310,
                    'def_receiving_td_allowed': 1,
                    'def_completions_allowed': 22,
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(
        load_pfr_advstats=lambda seasons, stat_type, summary_level: defense_frame,
    )
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    result = backend.load_weekly_defense_stats([2025])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(backend.DEFENSE_REQUIRED_COLUMNS)
    assert result.loc[0, 'def_sacks'] == 3


def test_load_weekly_player_stats_raises_clear_schema_error(monkeypatch):
    missing_col_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'player_id': 'p1',
                    'player_name': 'Alpha Player',
                    'pos': 'RB',
                    'team': 'NYG',
                    'season': 2025,
                    'week': 1,
                    'game_type': 'REG',
                    'fantasy_points': 12.5,
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(
        load_player_stats=lambda seasons, summary_level: missing_col_frame,
    )
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    with pytest.raises(backend.NFLVerseBackendError, match='player stats'):
        backend.load_weekly_player_stats([2025])


def test_load_draft_picks_normalizes_columns(monkeypatch):
    draft_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'draft_year': 2025,
                    'gsis_id': 'p1',
                    'player_name': 'Alpha Rookie',
                    'pos': 'WR',
                    'team': 'NYG',
                    'round': 1,
                    'pick': 12,
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(load_draft_picks=lambda seasons: draft_frame)
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    result = backend.load_draft_picks([2025])

    assert list(result.columns) == list(backend.DRAFT_PICK_COLUMNS)
    assert result.loc[0, 'draft_round'] == 1
    assert result.loc[0, 'draft_pick'] == 12


def test_load_depth_charts_normalizes_rank_column(monkeypatch):
    depth_frame = _FakePolarsFrame(
        pd.DataFrame(
            [
                {
                    'season': 2025,
                    'gsis_id': 'p1',
                    'player_name': 'Alpha Rookie',
                    'pos': 'WR',
                    'team': 'NYG',
                    'depth_team_order': 2,
                }
            ]
        )
    )
    fake_backend = SimpleNamespace(load_depth_charts=lambda seasons: depth_frame)
    monkeypatch.setattr(backend, '_get_backend_module', lambda: fake_backend)

    result = backend.load_depth_charts([2025])

    assert list(result.columns) == list(backend.DEPTH_CHART_COLUMNS)
    assert result.loc[0, 'depth_chart_rank'] == 2
