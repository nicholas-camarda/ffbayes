"""Backend adapter for nflverse data loaders.

This module is the only place in the runtime code that knows about the
nflverse backend package. It converts Polars frames to pandas DataFrames,
normalizes column names, and raises backend-specific errors when the shape
does not match the collector contract.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd


class NFLVerseBackendError(RuntimeError):
    """Raised when the nflverse backend fails or returns an unexpected shape."""


PLAYER_STATS_REQUIRED_COLUMNS = (
    'player_id',
    'player_display_name',
    'position',
    'recent_team',
    'season',
    'week',
    'season_type',
    'fantasy_points',
    'fantasy_points_ppr',
)

PLAYER_STATS_OPTIONAL_COLUMNS = (
    'game_injury_report_status',
    'practice_injury_report_status',
)

SCHEDULE_REQUIRED_COLUMNS = (
    'game_id',
    'week',
    'season',
    'gameday',
    'home_team',
    'away_team',
    'home_score',
    'away_score',
)

DEFENSE_REQUIRED_COLUMNS = (
    'team',
    'week',
    'season',
    'def_sacks',
    'def_ints',
    'def_tackles_combined',
    'def_missed_tackles',
    'def_pressures',
    'def_times_hitqb',
    'def_times_hurried',
    'def_times_blitzed',
    'def_yards_allowed',
    'def_receiving_td_allowed',
    'def_completions_allowed',
)

ROSTER_REQUIRED_COLUMNS = (
    'player_id',
    'player_display_name',
    'recent_team',
    'season',
    'week',
)


@lru_cache(maxsize=1)
def _get_backend_module():
    """Import the nflreadpy backend lazily."""
    try:
        import nflreadpy as backend
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise NFLVerseBackendError(
            'nflreadpy is required for NFL collection. Install nflreadpy to '
            'collect data with the nflverse backend.'
        ) from exc

    return backend


def _to_pandas(frame: Any) -> pd.DataFrame:
    """Convert a backend frame to pandas without leaking Polars outward."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()

    to_pandas = getattr(frame, 'to_pandas', None)
    if callable(to_pandas):
        return to_pandas()

    return pd.DataFrame(frame)


def _load_frame(loader_name: str, loader, **kwargs) -> pd.DataFrame:
    """Load a backend frame and wrap backend errors consistently."""
    try:
        raw_frame = loader(**kwargs)
    except NFLVerseBackendError:
        raise
    except Exception as exc:  # pragma: no cover - backend specific failures
        raise NFLVerseBackendError(
            f'nflreadpy loader {loader_name} failed: {exc}'
        ) from exc

    try:
        return _to_pandas(raw_frame)
    except Exception as exc:  # pragma: no cover - conversion guard
        raise NFLVerseBackendError(
            f'nflreadpy loader {loader_name} returned an invalid frame: {exc}'
        ) from exc


def _normalize_columns(
    frame: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
    *,
    context: str,
) -> pd.DataFrame:
    """Normalize a frame to the canonical collector schema."""
    renamed = frame.copy()
    rename_map: dict[str, str] = {}
    missing_columns: list[str] = []

    for target_name, candidate_names in aliases.items():
        for candidate_name in candidate_names:
            if candidate_name in renamed.columns:
                if candidate_name != target_name:
                    rename_map[candidate_name] = target_name
                break
        else:
            missing_columns.append(target_name)

    if rename_map:
        renamed = renamed.rename(columns=rename_map)

    if missing_columns:
        raise NFLVerseBackendError(
            f'Missing required nflreadpy {context} columns: {missing_columns}'
        )

    return renamed


def _ensure_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add missing optional columns as explicit nulls."""
    result = frame.copy()
    for column_name in columns:
        if column_name not in result.columns:
            result[column_name] = None
    return result


def _merge_status_columns(
    player_frame: pd.DataFrame,
    seasons: list[int],
    backend,
) -> pd.DataFrame:
    """Try to enrich player stats with weekly injury status fields."""
    status_frame = pd.DataFrame()

    for loader_name in ('load_injuries', 'load_rosters_weekly'):
        loader = getattr(backend, loader_name, None)
        if loader is None:
            continue

        try:
            candidate = _load_frame(loader_name, loader, seasons=seasons)
        except NFLVerseBackendError:
            continue

        if candidate.empty:
            continue

        if loader_name == 'load_injuries':
            aliases = {
                'player_id': ('player_id', 'gsis_id'),
                'season': ('season',),
                'week': ('week',),
                'game_injury_report_status': (
                    'game_injury_report_status',
                    'report_status',
                    'game_status',
                    'injury_status',
                ),
                'practice_injury_report_status': (
                    'practice_injury_report_status',
                    'practice_status',
                    'practice',
                ),
            }
        else:
            aliases = {
                'player_id': ('player_id', 'gsis_id'),
                'player_display_name': ('player_display_name', 'player_name', 'name'),
                'recent_team': ('recent_team', 'team'),
                'season': ('season',),
                'week': ('week',),
                'game_injury_report_status': (
                    'game_injury_report_status',
                    'report_status',
                    'game_status',
                    'injury_status',
                ),
                'practice_injury_report_status': (
                    'practice_injury_report_status',
                    'practice_status',
                    'practice',
                ),
            }

        try:
            status_frame = _normalize_columns(
                candidate, aliases, context=loader_name
            )
            break
        except NFLVerseBackendError:
            continue

    if status_frame.empty:
        return _ensure_columns(player_frame, PLAYER_STATS_OPTIONAL_COLUMNS)

    join_keys = [
        column_name
        for column_name in ('player_id', 'season', 'week')
        if column_name in player_frame.columns and column_name in status_frame.columns
    ]
    status_columns = [
        column_name
        for column_name in PLAYER_STATS_OPTIONAL_COLUMNS
        if column_name in status_frame.columns
    ]

    if not join_keys or not status_columns:
        return _ensure_columns(player_frame, PLAYER_STATS_OPTIONAL_COLUMNS)

    merged = player_frame.copy()
    for status_column in status_columns:
        if status_column in merged.columns:
            continue

        lookup = status_frame[join_keys + [status_column]].drop_duplicates(join_keys)
        merged = merged.merge(lookup, on=join_keys, how='left')

    return _ensure_columns(merged, PLAYER_STATS_OPTIONAL_COLUMNS)


def load_weekly_player_stats(years: list[int]) -> pd.DataFrame:
    """Load weekly player stats as a pandas DataFrame."""
    backend = _get_backend_module()
    frame = _load_frame(
        'load_player_stats',
        backend.load_player_stats,
        seasons=years,
        summary_level='week',
    )
    normalized = _normalize_columns(
        frame,
        {
            'player_id': ('player_id',),
            'player_display_name': (
                'player_display_name',
                'player_name',
                'player',
                'name',
            ),
            'position': ('position', 'pos'),
            'recent_team': ('recent_team', 'team'),
            'season': ('season',),
            'week': ('week',),
            'season_type': ('season_type', 'game_type'),
            'fantasy_points': ('fantasy_points',),
            'fantasy_points_ppr': ('fantasy_points_ppr',),
        },
        context='player stats',
    )
    normalized = _merge_status_columns(normalized, years, backend)
    return _ensure_columns(normalized, PLAYER_STATS_OPTIONAL_COLUMNS)


def load_schedules(years: list[int]) -> pd.DataFrame:
    """Load schedules as a pandas DataFrame."""
    backend = _get_backend_module()
    frame = _load_frame('load_schedules', backend.load_schedules, seasons=years)
    return _normalize_columns(
        frame,
        {
            'game_id': ('game_id',),
            'week': ('week',),
            'season': ('season',),
            'gameday': ('gameday', 'game_date', 'date'),
            'home_team': ('home_team',),
            'away_team': ('away_team',),
            'home_score': ('home_score',),
            'away_score': ('away_score',),
        },
        context='schedule',
    )


def load_injuries(years: list[int]) -> pd.DataFrame:
    """Load injuries as a pandas DataFrame."""
    backend = _get_backend_module()
    frame = _load_frame('load_injuries', backend.load_injuries, seasons=years)
    return _normalize_columns(
        frame,
        {
            'player_id': ('player_id', 'gsis_id'),
            'player_display_name': ('player_display_name', 'player_name', 'name'),
            'season': ('season',),
            'week': ('week',),
            'game_injury_report_status': (
                'game_injury_report_status',
                'report_status',
                'game_status',
                'injury_status',
            ),
            'practice_injury_report_status': (
                'practice_injury_report_status',
                'practice_status',
                'practice',
            ),
        },
        context='injury',
    )


def load_weekly_rosters(years: list[int]) -> pd.DataFrame:
    """Load weekly rosters as a pandas DataFrame."""
    backend = _get_backend_module()
    frame = _load_frame(
        'load_rosters_weekly', backend.load_rosters_weekly, seasons=years
    )
    normalized = _normalize_columns(
        frame,
        {
            'player_id': ('player_id', 'gsis_id'),
            'player_display_name': ('player_display_name', 'player_name', 'name'),
            'recent_team': ('recent_team', 'team'),
            'season': ('season',),
            'week': ('week',),
        },
        context='weekly roster',
    )
    return _ensure_columns(normalized, PLAYER_STATS_OPTIONAL_COLUMNS)


def load_weekly_defense_stats(years: list[int]) -> pd.DataFrame:
    """Load weekly defensive PFR statistics as a pandas DataFrame."""
    backend = _get_backend_module()
    frame = _load_frame(
        'load_pfr_advstats',
        backend.load_pfr_advstats,
        seasons=years,
        stat_type='def',
        summary_level='week',
    )
    return _normalize_columns(
        frame,
        {
            'team': ('team',),
            'week': ('week',),
            'season': ('season',),
            'def_sacks': ('def_sacks',),
            'def_ints': ('def_ints',),
            'def_tackles_combined': ('def_tackles_combined',),
            'def_missed_tackles': ('def_missed_tackles',),
            'def_pressures': ('def_pressures',),
            'def_times_hitqb': ('def_times_hitqb',),
            'def_times_hurried': ('def_times_hurried',),
            'def_times_blitzed': ('def_times_blitzed',),
            'def_yards_allowed': ('def_yards_allowed',),
            'def_receiving_td_allowed': ('def_receiving_td_allowed',),
            'def_completions_allowed': ('def_completions_allowed',),
        },
        context='defense stats',
    )
