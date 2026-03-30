#!/usr/bin/env python3
"""
Draft decision system for FFBayes.

This module replaces the old "rank players and call it Bayesian" flow with a
draft-facing decision stack:

* normalize player data from multiple public/free sources
* build a player-level decision table
* score live snake-draft recommendations conditional on roster state
* simulate strategy backtests on historical seasons
* export a portable workbook and dashboard payload for draft day

The implementation is intentionally conservative and data-first. When inputs
are missing, the code falls back to transparent heuristics instead of failing
open with invented values.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.stats import spearmanr

POSITION_ORDER = ['QB', 'RB', 'WR', 'TE', 'DST', 'K']
DEFAULT_FLEX_WEIGHTS = {'RB': 0.45, 'WR': 0.45, 'TE': 0.10}
DEFAULT_ROSTER_TEMPLATE = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,
    'DST': 1,
    'K': 1,
}


def _safe_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value)


def _coerce_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(float(value))
    except Exception:
        return default


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    mean = values.mean(skipna=True)
    std = values.std(ddof=0, skipna=True)
    if pd.isna(std) or np.isclose(std, 0.0):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


def _clamp(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    return series.clip(lower=low, upper=high)


def _normalize_position(value: Any) -> str:
    pos = _safe_string(value).upper().strip()
    if not pos:
        return 'UNKNOWN'
    if pos.startswith('RB'):
        return 'RB'
    if pos.startswith('WR'):
        return 'WR'
    if pos.startswith('QB'):
        return 'QB'
    if pos.startswith('TE'):
        return 'TE'
    if pos.startswith('DST') or pos in {'D/ST', 'DEF', 'DEFENSE'}:
        return 'DST'
    if pos.startswith('K'):
        return 'K'
    return pos[:3]


def _pick_first_row(series: pd.Series, fallback: Any = None) -> Any:
    for value in series:
        if pd.notna(value) and _safe_string(value):
            return value
    return fallback


@dataclass(frozen=True)
class LeagueSettings:
    """League-level configuration used to convert projections into decisions."""

    league_size: int = 10
    draft_position: int = 10
    scoring_type: str = 'PPR'
    ppr_value: float = 0.5
    risk_tolerance: str = 'medium'
    roster_spots: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_ROSTER_TEMPLATE)
    )
    flex_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FLEX_WEIGHTS)
    )
    bench_slots: int = 6

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> 'LeagueSettings':
        mapping = mapping or {}
        league_settings = mapping.get('league_settings', mapping)
        roster = dict(DEFAULT_ROSTER_TEMPLATE)
        roster.update(
            {
                k.upper(): _coerce_int(v, roster.get(k.upper(), 0))
                for k, v in league_settings.get('roster_spots', {}).items()
            }
        )
        flex_weights = dict(DEFAULT_FLEX_WEIGHTS)
        flex_weights.update(
            {
                k.upper(): float(v)
                for k, v in league_settings.get('flex_weights', {}).items()
            }
        )
        return cls(
            league_size=_coerce_int(league_settings.get('league_size', 10), 10),
            draft_position=_coerce_int(league_settings.get('draft_position', 10), 10),
            scoring_type=_safe_string(league_settings.get('scoring_type', 'PPR'))
            or 'PPR',
            ppr_value=_coerce_float(
                league_settings.get('ppr_value', league_settings.get('ppr', 0.5)), 0.5
            ),
            risk_tolerance=_safe_string(league_settings.get('risk_tolerance', 'medium'))
            or 'medium',
            roster_spots=roster,
            flex_weights=flex_weights,
            bench_slots=_coerce_int(league_settings.get('bench_slots', 6), 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def starters_by_position(self) -> dict[str, int]:
        return {
            pos: self.roster_spots.get(pos, 0) * self.league_size
            for pos in POSITION_ORDER
        }

    def effective_replacement_slots(self) -> dict[str, int]:
        starters = self.starters_by_position()
        flex_slots = self.roster_spots.get('FLEX', 0) * self.league_size
        flex_allocation = {
            'RB': int(round(flex_slots * self.flex_weights.get('RB', 0.0))),
            'WR': int(round(flex_slots * self.flex_weights.get('WR', 0.0))),
            'TE': int(round(flex_slots * self.flex_weights.get('TE', 0.0))),
        }
        replacement = dict(starters)
        replacement['RB'] = replacement.get('RB', 0) + flex_allocation['RB']
        replacement['WR'] = replacement.get('WR', 0) + flex_allocation['WR']
        replacement['TE'] = replacement.get('TE', 0) + flex_allocation['TE']
        return replacement

    def round_count(self) -> int:
        base = sum(
            self.roster_spots.get(pos, 0)
            for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']
        )
        base += self.bench_slots
        return max(1, base)


@dataclass
class DraftContext:
    """State of the live draft board when the recommendation is generated."""

    current_pick_number: int
    drafted_players: set[str] = field(default_factory=set)
    keepers: set[str] = field(default_factory=set)
    your_players: set[str] = field(default_factory=set)
    roster_counts: dict[str, int] = field(default_factory=dict)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    selected_player: str | None = None

    def drafted_set(self) -> set[str]:
        drafted = {
            name.strip().lower()
            for name in self.drafted_players
            if isinstance(name, str) and name.strip()
        }
        drafted.update(
            {
                name.strip().lower()
                for name in self.keepers
                if isinstance(name, str) and name.strip()
            }
        )
        drafted.update(
            {
                name.strip().lower()
                for name in self.your_players
                if isinstance(name, str) and name.strip()
            }
        )
        return drafted


@dataclass(frozen=True)
class LiveDraftState:
    """Client-side draft state tracked by the dashboard."""

    current_pick_number: int
    next_pick_number: int
    taken_players: list[str] = field(default_factory=list)
    your_players: list[str] = field(default_factory=list)
    roster_counts: dict[str, int] = field(default_factory=dict)
    action_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DraftDecisionArtifacts:
    """Container for all draft-facing artifacts produced by the system."""

    league_settings: LeagueSettings
    decision_table: pd.DataFrame
    recommendations: pd.DataFrame
    roster_scenarios: pd.DataFrame
    tier_cliffs: pd.DataFrame
    source_freshness: pd.DataFrame
    backtest: dict[str, Any]
    dashboard_payload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'league_settings': self.league_settings.to_dict(),
            'decision_table': self.decision_table.to_dict(orient='records'),
            'recommendations': self.recommendations.to_dict(orient='records'),
            'roster_scenarios': self.roster_scenarios.to_dict(orient='records'),
            'tier_cliffs': self.tier_cliffs.to_dict(orient='records'),
            'source_freshness': self.source_freshness.to_dict(orient='records'),
            'backtest': self.backtest,
            'dashboard_payload': self.dashboard_payload,
            'metadata': self.metadata,
        }


def normalize_player_frame(player_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a messy player frame into a standard draft table schema."""
    if player_frame is None or player_frame.empty:
        raise ValueError('player_frame is empty')

    df = player_frame.copy()

    rename_map = {}
    canonical_columns = set(df.columns)
    for column in df.columns:
        normalized = column.strip().lower().replace(' ', '_')
        if normalized in {'player', 'player_name', 'name', 'playername'}:
            if 'player_name' in canonical_columns:
                continue
            rename_map[column] = 'player_name'
            canonical_columns.add('player_name')
        elif normalized in {'pos', 'position', 'slot'}:
            if 'position' in canonical_columns:
                continue
            rename_map[column] = 'position'
            canonical_columns.add('position')
        elif normalized in {
            'fpts',
            'fantpt',
            'fantasy_points',
            'projected_points',
            'projected_fpts',
            'proj_points',
            'projection',
        }:
            if 'proj_points_mean' in canonical_columns:
                continue
            rename_map[column] = 'proj_points_mean'
            canonical_columns.add('proj_points_mean')
        elif normalized in {'adp', 'avg', 'average_draft_position', 'market_rank'}:
            if 'adp' in canonical_columns:
                continue
            rename_map[column] = 'adp'
            canonical_columns.add('adp')
        elif normalized in {'mean_projection', 'mean_proj', 'consensus_projection'}:
            if 'proj_points_mean' in canonical_columns:
                continue
            rename_map[column] = 'proj_points_mean'
            canonical_columns.add('proj_points_mean')
        elif normalized in {'std_projection', 'projection_std', 'projection_spread'}:
            if 'std_projection' in canonical_columns:
                continue
            rename_map[column] = 'std_projection'
            canonical_columns.add('std_projection')
        elif normalized in {'uncertainty_score', 'risk_score', 'volatility_score'}:
            if 'uncertainty_score' in canonical_columns:
                continue
            rename_map[column] = 'uncertainty_score'
            canonical_columns.add('uncertainty_score')
        elif normalized in {'vor', 'value_over_replacement'}:
            if 'vor_value' in canonical_columns:
                continue
            rename_map[column] = 'vor_value'
            canonical_columns.add('vor_value')
        elif normalized in {'valuerank', 'vor_rank', 'market_rank_numeric'}:
            if 'market_rank' in canonical_columns:
                continue
            rename_map[column] = 'market_rank'
            canonical_columns.add('market_rank')

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    if 'player_name' not in df.columns:
        raise ValueError('player_frame must include a player name column')
    if 'position' not in df.columns:
        raise ValueError('player_frame must include a position column')

    df['player_name'] = df['player_name'].map(_safe_string)
    df['position'] = df['position'].map(_normalize_position)

    if 'proj_points_mean' not in df.columns:
        source_columns = [
            'mean_projection',
            'FPTS',
            'FantPt',
            'projected_points',
            'projection',
        ]
        for column in source_columns:
            if column in player_frame.columns:
                df['proj_points_mean'] = pd.to_numeric(
                    player_frame[column], errors='coerce'
                )
                break
    df['proj_points_mean'] = pd.to_numeric(df.get('proj_points_mean'), errors='coerce')

    if 'adp' in df.columns:
        df['adp'] = pd.to_numeric(df['adp'], errors='coerce')
    else:
        df['adp'] = np.nan

    if 'std_projection' in df.columns:
        df['std_projection'] = pd.to_numeric(df['std_projection'], errors='coerce')
    else:
        df['std_projection'] = np.nan

    if 'uncertainty_score' in df.columns:
        df['uncertainty_score'] = pd.to_numeric(
            df['uncertainty_score'], errors='coerce'
        )
    else:
        df['uncertainty_score'] = np.nan

    if 'vor_value' in df.columns:
        df['vor_value'] = pd.to_numeric(df['vor_value'], errors='coerce')
    else:
        df['vor_value'] = np.nan

    if 'market_rank' in df.columns:
        df['market_rank'] = pd.to_numeric(df['market_rank'], errors='coerce')
    else:
        df['market_rank'] = np.nan

    if 'season_count' not in df.columns:
        df['season_count'] = np.nan
    if 'games_missed' not in df.columns:
        df['games_missed'] = np.nan
    if 'age' not in df.columns:
        df['age'] = np.nan
    if 'years_in_league' not in df.columns:
        df['years_in_league'] = np.nan
    if 'team_change' not in df.columns:
        df['team_change'] = np.nan
    if 'role_volatility' not in df.columns:
        df['role_volatility'] = np.nan
    if 'role_label' not in df.columns:
        df['role_label'] = ''
    if 'team_pass_rate' not in df.columns:
        df['team_pass_rate'] = np.nan
    if 'neutral_pace' not in df.columns:
        df['neutral_pace'] = np.nan
    if 'vacated_targets' not in df.columns:
        df['vacated_targets'] = np.nan
    if 'teammate_competition' not in df.columns:
        df['teammate_competition'] = np.nan
    if 'site_disagreement' not in df.columns:
        df['site_disagreement'] = np.nan
    if 'adp_std' not in df.columns:
        df['adp_std'] = np.nan
    if 'source_name' not in df.columns:
        df['source_name'] = 'current_market_snapshot'
    if 'source_updated_at' not in df.columns:
        df['source_updated_at'] = pd.NaT

    df = df.drop_duplicates(subset=['player_name', 'position'], keep='last').reset_index(drop=True)
    return df


def _position_baseline(
    frame: pd.DataFrame, position: str, slot_count: int, fallback: float | None = None
) -> float:
    pos_values = pd.to_numeric(
        frame.loc[frame['position'] == position, 'proj_points_mean'], errors='coerce'
    ).dropna()
    if pos_values.empty:
        if fallback is not None and not pd.isna(fallback):
            return float(fallback)
        overall = pd.to_numeric(frame['proj_points_mean'], errors='coerce').dropna()
        return float(overall.mean()) if not overall.empty else 0.0

    slot_count = max(1, min(len(pos_values), int(round(slot_count))))
    sorted_values = pos_values.sort_values(ascending=False).reset_index(drop=True)
    return float(sorted_values.iloc[slot_count - 1])


def _compute_freshness(source_frame: pd.DataFrame) -> pd.DataFrame:
    if source_frame is None or source_frame.empty:
        return pd.DataFrame(
            columns=['source_name', 'source_updated_at', 'freshness_days', 'row_count']
        )

    summary_rows = []
    for source_name, sub in source_frame.groupby('source_name', dropna=False):
        updated = pd.to_datetime(sub['source_updated_at'], errors='coerce')
        freshest = updated.max() if not updated.isna().all() else pd.NaT
        freshness_days = (
            (pd.Timestamp.now(tz=None) - freshest).days
            if pd.notna(freshest)
            else np.nan
        )
        summary_rows.append(
            {
                'source_name': source_name if source_name else 'unknown',
                'source_updated_at': freshest,
                'freshness_days': freshness_days,
                'row_count': int(len(sub)),
                'missing_rate': float(sub.isna().mean().mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def availability_probability(
    adp: float | int | None,
    target_pick: int,
    adp_std: float | int | None = None,
    uncertainty_score: float | int | None = None,
) -> float:
    """Estimate the probability a player survives to a future pick."""
    if adp is None or pd.isna(adp):
        return 0.5

    spread = _coerce_float(adp_std, np.nan)
    if pd.isna(spread) or spread <= 0:
        spread = 2.5

    if uncertainty_score is not None and not pd.isna(uncertainty_score):
        spread += 2.0 * float(uncertainty_score)

    z = (float(adp) - float(target_pick)) / max(1.0, spread)
    prob = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(prob, 0.0, 1.0))


def build_decision_table(
    player_frame: pd.DataFrame,
    league_settings: LeagueSettings | None = None,
    context: DraftContext | None = None,
) -> pd.DataFrame:
    """Build a player-level decision table aligned to the league context."""
    settings = league_settings or LeagueSettings()
    context = context or DraftContext(current_pick_number=settings.draft_position)

    df = normalize_player_frame(player_frame)

    if df['proj_points_mean'].isna().all():
        raise ValueError('player_frame must contain a usable projection column')

    df['proj_points_mean'] = pd.to_numeric(df['proj_points_mean'], errors='coerce')

    # Project floors/ceilings from uncertainty or historical spread.
    uncertainty = df['uncertainty_score'].copy()
    if uncertainty.isna().all():
        uncertainty = (
            pd.to_numeric(df['std_projection'], errors='coerce')
            / df['proj_points_mean'].replace(0, np.nan)
        ).fillna(0.15)
    else:
        uncertainty = uncertainty.fillna(
            (
                pd.to_numeric(df['std_projection'], errors='coerce')
                / df['proj_points_mean'].replace(0, np.nan)
            ).fillna(0.15)
        )

    df['uncertainty_score'] = _clamp(uncertainty, 0.0, 1.0)

    spread_from_std = pd.to_numeric(df['std_projection'], errors='coerce')
    fallback_spread = (
        df['proj_points_mean'].abs() * (0.08 + 0.35 * df['uncertainty_score'])
    ).fillna(0.0)
    df['proj_points_floor'] = df['proj_points_mean'] - spread_from_std.fillna(
        fallback_spread
    )
    df['proj_points_ceiling'] = (
        df['proj_points_mean'] + spread_from_std.fillna(fallback_spread) * 1.25
    )

    # Replacement and starter baselines depend on league structure.
    starter_slots = settings.starters_by_position()
    replacement_slots = settings.effective_replacement_slots()
    df['starter_baseline'] = np.nan
    df['replacement_baseline'] = np.nan
    for position in df['position'].dropna().unique():
        starter_baseline = _position_baseline(
            df,
            position,
            starter_slots.get(position, 0),
            fallback=df['proj_points_mean'].mean(),
        )
        replacement_baseline = _position_baseline(
            df, position, replacement_slots.get(position, 0), fallback=starter_baseline
        )
        pos_mask = df['position'] == position
        df.loc[pos_mask, 'starter_baseline'] = starter_baseline
        df.loc[pos_mask, 'replacement_baseline'] = replacement_baseline

    df['starter_delta'] = df['proj_points_mean'] - df['starter_baseline']
    df['replacement_delta'] = df['proj_points_mean'] - df['replacement_baseline']

    # Market rank and relative market gap.
    if df['adp'].notna().any():
        df['market_rank'] = df['adp'].rank(method='first', ascending=True)
    elif df['market_rank'].notna().any():
        df['market_rank'] = pd.to_numeric(df['market_rank'], errors='coerce')
    else:
        df['market_rank'] = df['proj_points_mean'].rank(method='first', ascending=False)

    df['model_rank'] = df['proj_points_mean'].rank(method='first', ascending=False)
    df['market_gap'] = df['market_rank'] - df['model_rank']

    # Availability is based on the next time we expect to pick again in a snake draft.
    target_pick = next_pick_number(
        context.current_pick_number, settings.draft_position, settings.league_size
    )
    df['availability_at_pick'] = [
        availability_probability(
            adp=row.adp,
            target_pick=target_pick,
            adp_std=row.adp_std,
            uncertainty_score=row.uncertainty_score,
        )
        for row in df.itertuples(index=False)
    ]

    # Risk and upside signals.
    missing_history = pd.to_numeric(df['season_count'], errors='coerce').fillna(0.0)
    games_missed = pd.to_numeric(df['games_missed'], errors='coerce').fillna(0.0)
    age = pd.to_numeric(df['age'], errors='coerce').fillna(
        df['age'].median(skipna=True) if df['age'].notna().any() else 27.0
    )
    years = pd.to_numeric(df['years_in_league'], errors='coerce').fillna(0.0)
    team_change = pd.to_numeric(df['team_change'], errors='coerce').fillna(0.0)
    role_volatility = pd.to_numeric(df['role_volatility'], errors='coerce').fillna(0.0)
    site_disagreement = pd.to_numeric(df['site_disagreement'], errors='coerce').fillna(
        0.0
    )
    adp_std = pd.to_numeric(df['adp_std'], errors='coerce').fillna(0.0)

    history_penalty = _clamp(
        pd.Series(1.0 / np.maximum(1.0, missing_history + 1.0)), 0.0, 1.0
    )
    injury_penalty = _clamp(pd.Series(np.tanh(games_missed / 6.0)), 0.0, 1.0)
    age_penalty = _clamp(pd.Series(np.maximum(0.0, (age - 29.0) / 8.0)), 0.0, 1.0)
    role_penalty = _clamp(pd.Series(np.maximum(0.0, role_volatility)), 0.0, 1.0)
    team_penalty = _clamp(pd.Series(np.maximum(0.0, team_change)), 0.0, 1.0)
    disagreement_penalty = _clamp(
        pd.Series(np.maximum(0.0, np.maximum(site_disagreement, adp_std / 10.0))),
        0.0,
        1.0,
    )

    if 'projected_points_spread' in df.columns:
        spread = pd.to_numeric(df['projected_points_spread'], errors='coerce').fillna(
            0.0
        )
    else:
        spread = (df['proj_points_ceiling'] - df['proj_points_floor']) / 2.0
    upside_gap = (df['proj_points_ceiling'] - df['proj_points_mean']).fillna(0.0)

    df['fragility_score'] = _clamp(
        0.22 * history_penalty
        + 0.25 * injury_penalty
        + 0.12 * age_penalty
        + 0.16 * team_penalty
        + 0.15 * role_penalty
        + 0.10 * disagreement_penalty,
        0.0,
        1.0,
    )

    df['upside_score'] = _clamp(
        _zscore(upside_gap).rank(pct=True)
        + 0.35 * _zscore(df['availability_at_pick']).rank(pct=True)
        + 0.15 * _zscore(df['proj_points_mean']).rank(pct=True),
        0.0,
        1.0,
    )

    # Lean toward players who are both good and likely to survive.
    df['starter_need'] = 0.0
    roster_need = {
        'QB': max(
            0, settings.roster_spots.get('QB', 0) - context.roster_counts.get('QB', 0)
        ),
        'RB': max(
            0, settings.roster_spots.get('RB', 0) - context.roster_counts.get('RB', 0)
        ),
        'WR': max(
            0, settings.roster_spots.get('WR', 0) - context.roster_counts.get('WR', 0)
        ),
        'TE': max(
            0, settings.roster_spots.get('TE', 0) - context.roster_counts.get('TE', 0)
        ),
    }
    roster_need_total = sum(roster_need.values()) or 1
    for position, need in roster_need.items():
        if need > 0:
            df.loc[df['position'] == position, 'starter_need'] = (
                need / roster_need_total
            )

    scarcity_by_position = df.groupby('position')['player_name'].transform('count')
    df['position_scarcity'] = _clamp(
        1.0 / np.maximum(1.0, scarcity_by_position), 0.0, 1.0
    )

    risk_multiplier = {'low': 0.80, 'medium': 1.00, 'high': 1.18}.get(
        settings.risk_tolerance.lower(), 1.00
    )
    df['draft_score'] = (
        0.34 * _zscore(df['starter_delta']).fillna(0.0)
        + 0.20 * _zscore(df['replacement_delta']).fillna(0.0)
        + 0.16 * _zscore(df['proj_points_mean']).fillna(0.0)
        + 0.12 * _zscore(df['availability_at_pick']).fillna(0.0)
        + 0.10 * _zscore(df['upside_score']).fillna(0.0)
        + 0.08 * _zscore(df['starter_need']).fillna(0.0)
        + 0.08 * _zscore(df['position_scarcity']).fillna(0.0)
        - (0.25 * risk_multiplier) * _zscore(df['fragility_score']).fillna(0.0)
        + 0.06 * _zscore(df['market_gap']).fillna(0.0)
    )

    df['draft_score'] = df['draft_score'].fillna(0.0)
    df['draft_rank'] = (
        df['draft_score'].rank(method='first', ascending=False).astype(int)
    )
    df['draft_tier'] = _assign_tiers(df['draft_score'])
    df['market_value_gap'] = df['model_rank'] - df['market_rank']
    df['why_flags'] = df.apply(_build_why_flags, axis=1)

    df = df.sort_values(
        ['draft_score', 'proj_points_mean'], ascending=[False, False]
    ).reset_index(drop=True)
    return df


def _assign_tiers(scores: pd.Series, num_tiers: int = 5) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype='object')
    quantiles = pd.qcut(
        scores.rank(method='first', ascending=False),
        q=min(num_tiers, len(scores)),
        labels=False,
        duplicates='drop',
    )
    tiers = pd.Series(quantiles, index=scores.index).fillna(0).astype(int) + 1
    return tiers.map(lambda tier: f'Tier {tier}')


def _build_why_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if (
        pd.notna(row.get('adp'))
        and pd.notna(row.get('market_rank'))
        and (row['market_rank'] - row['model_rank']) > 8
    ):
        flags.append('market_discount')
    if pd.notna(row.get('availability_at_pick')) and row['availability_at_pick'] < 0.40:
        flags.append('survival_risk')
    if pd.notna(row.get('upside_score')) and row['upside_score'] > 0.70:
        flags.append('upside_play')
    if pd.notna(row.get('fragility_score')) and row['fragility_score'] > 0.60:
        flags.append('fragile')
    if pd.notna(row.get('starter_delta')) and row['starter_delta'] > 0:
        flags.append('starter_gain')
    if (
        row.get('position') in {'RB', 'WR'}
        and pd.notna(row.get('position_scarcity'))
        and row['position_scarcity'] > 0.10
    ):
        flags.append('scarce_position')
    if pd.notna(row.get('season_count')) and row['season_count'] <= 1:
        flags.append('cohort_prior')
    if pd.notna(row.get('site_disagreement')) and row['site_disagreement'] > 0.25:
        flags.append('projection_disagreement')
    return '|'.join(flags)


def next_pick_number(
    current_pick_number: int, draft_position: int, league_size: int
) -> int:
    """Return the next pick at which our team will draft in a snake draft."""
    current_pick_number = max(1, int(current_pick_number))
    draft_position = max(1, int(draft_position))
    league_size = max(1, int(league_size))

    picks = []
    rounds = max(1, math.ceil(current_pick_number / league_size) + 2)
    for round_num in range(1, rounds + 1):
        if round_num % 2 == 1:
            pick = (round_num - 1) * league_size + draft_position
        else:
            pick = round_num * league_size - draft_position + 1
        picks.append(pick)

    for pick in picks:
        if pick > current_pick_number:
            return pick
    return picks[-1]


def build_recommendations(
    decision_table: pd.DataFrame,
    league_settings: LeagueSettings,
    context: DraftContext,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the best current picks and survival-friendly wait options."""
    available = decision_table.copy()
    if context.drafted_players:
        drafted = context.drafted_set()
        available = available[
            ~available['player_name'].str.lower().isin(drafted)
        ].copy()
    if context.keepers:
        keepers = {name.lower() for name in context.keepers}
        available = available[
            ~available['player_name'].str.lower().isin(keepers)
        ].copy()

    if available.empty:
        return pd.DataFrame(
            columns=[
                'player_name',
                'position',
                'draft_score',
                'availability_at_pick',
                'expected_regret',
                'position_run_risk',
                'roster_fit_score',
                'pick_mode',
                'recommendation_lane',
                'lane_rank',
                'rationale',
            ]
        )

    next_turn_pick = next_pick_number(
        context.current_pick_number,
        league_settings.draft_position,
        league_settings.league_size,
    )
    available = available.copy()
    available['availability_to_next_pick'] = [
        availability_probability(
            adp=row.adp,
            target_pick=next_turn_pick,
            adp_std=row.adp_std,
            uncertainty_score=row.uncertainty_score,
        )
        for row in available.itertuples(index=False)
    ]

    position_counts_remaining = available['position'].value_counts().to_dict()
    position_need = {
        pos: max(
            0,
            league_settings.roster_spots.get(pos, 0)
            - context.roster_counts.get(pos, 0),
        )
        for pos in ['QB', 'RB', 'WR', 'TE']
    }
    total_need = sum(position_need.values()) or 1

    def _pos_run_risk(position: str) -> float:
        remaining = position_counts_remaining.get(position, 0)
        need = position_need.get(position, 0)
        demand = max(1, league_settings.league_size * max(1, need))
        return float(np.clip(1.0 - remaining / demand, 0.0, 1.0))

    available['position_run_risk'] = available['position'].map(_pos_run_risk)
    available['roster_fit_score'] = available['position'].map(
        lambda pos: position_need.get(pos, 0) / total_need
    )

    # Regret proxy: how much value we lose if we wait until next turn.
    best_now = available['draft_score'].max()
    available['expected_regret'] = np.maximum(
        0.0, best_now - available['draft_score']
    ) * (1.0 - available['availability_to_next_pick'])
    available['expected_regret'] += 0.25 * available['position_run_risk']

    # Combine current pick utility with wait-survival utility.
    risk_bias = {'low': -0.08, 'medium': 0.0, 'high': 0.08}.get(
        league_settings.risk_tolerance.lower(), 0.0
    )
    available['current_pick_utility'] = (
        available['draft_score']
        + 0.15 * available['roster_fit_score']
        + 0.10 * available['position_run_risk']
        - 0.20 * (1.0 - available['availability_to_next_pick'])
        + risk_bias * available['upside_score']
    )
    available['wait_utility'] = (
        available['draft_score'] * available['availability_to_next_pick']
        + 0.18 * available['upside_score']
        - 0.15 * available['fragility_score']
    )

    now = (
        available.sort_values(
            ['current_pick_utility', 'draft_score'], ascending=[False, False]
        )
        .head(top_n)
        .copy()
    )
    now['pick_mode'] = 'now'
    now['recommendation_lane'] = ['pick_now'] + ['fallback'] * max(0, len(now) - 1)
    now['lane_rank'] = list(range(1, len(now) + 1))
    now['rationale'] = now.apply(_rationale_now, axis=1)

    wait = (
        available.sort_values(
            ['wait_utility', 'availability_to_next_pick'], ascending=[False, False]
        )
        .head(top_n)
        .copy()
    )
    wait = wait[~wait['player_name'].isin(now['player_name'])].copy()
    wait['pick_mode'] = 'wait'
    wait['recommendation_lane'] = 'can_wait'
    wait['lane_rank'] = list(range(1, len(wait) + 1))
    wait['rationale'] = wait.apply(_rationale_wait, axis=1)

    cols = [
        'player_name',
        'position',
        'proj_points_mean',
        'proj_points_floor',
        'proj_points_ceiling',
        'adp',
        'market_rank',
        'availability_to_next_pick',
        'replacement_delta',
        'starter_delta',
        'upside_score',
        'fragility_score',
        'draft_score',
        'draft_tier',
        'why_flags',
        'position_run_risk',
        'roster_fit_score',
        'expected_regret',
        'current_pick_utility',
        'wait_utility',
        'pick_mode',
        'recommendation_lane',
        'lane_rank',
        'rationale',
    ]
    combined = pd.concat([now[cols], wait[cols]], ignore_index=True)
    lane_order = {'pick_now': 0, 'fallback': 1, 'can_wait': 2}
    combined['lane_order'] = combined['recommendation_lane'].map(lane_order).fillna(9)
    combined = combined.sort_values(
        ['lane_order', 'lane_rank', 'draft_score'], ascending=[True, True, False]
    ).drop(columns=['lane_order'])
    combined = combined.reset_index(drop=True)
    return combined


def _rationale_now(row: pd.Series) -> str:
    parts = [f'{row["position"]} value', f'score={row["draft_score"]:.2f}']
    if row.get('why_flags'):
        parts.append(row['why_flags'].replace('|', ', '))
    if pd.notna(row.get('availability_to_next_pick')):
        parts.append(f'survival={row["availability_to_next_pick"]:.0%}')
    return '; '.join(parts)


def _rationale_wait(row: pd.Series) -> str:
    parts = ['wait candidate', f'score={row["draft_score"]:.2f}']
    if pd.notna(row.get('availability_to_next_pick')):
        parts.append(f'survival={row["availability_to_next_pick"]:.0%}')
    if pd.notna(row.get('expected_regret')):
        parts.append(f'regret={row["expected_regret"]:.2f}')
    return '; '.join(parts)


def _recommendation_rows_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.to_dict(orient='records')


def build_live_recommendation_snapshot(
    decision_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    roster_scenarios: pd.DataFrame,
    league_settings: LeagueSettings,
    context: DraftContext,
) -> dict[str, Any]:
    """Build the live contingency view used by the dashboard and workbook."""
    next_turn_pick = next_pick_number(
        context.current_pick_number,
        league_settings.draft_position,
        league_settings.league_size,
    )
    recommendation_rows = recommendations.copy()
    if recommendation_rows.empty:
        recommendation_rows = pd.DataFrame(
            columns=[
                'player_name',
                'position',
                'draft_score',
                'availability_to_next_pick',
                'expected_regret',
                'roster_fit_score',
                'position_run_risk',
                'why_flags',
                'recommendation_lane',
                'lane_rank',
            ]
        )

    pick_now = recommendation_rows[
        recommendation_rows['recommendation_lane'] == 'pick_now'
    ].head(1)
    fallbacks = recommendation_rows[
        recommendation_rows['recommendation_lane'] == 'fallback'
    ].head(5)
    can_wait = recommendation_rows[
        recommendation_rows['recommendation_lane'] == 'can_wait'
    ].sort_values(
        ['availability_to_next_pick', 'draft_score'], ascending=[False, False]
    )

    roster_need = {
        pos: max(
            0, league_settings.roster_spots.get(pos, 0) - context.roster_counts.get(pos, 0)
        )
        for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K']
    }

    best_roster_paths = roster_scenarios.head(3).to_dict(orient='records')
    wait_survival = can_wait.head(5).to_dict(orient='records')

    primary = pick_now.head(1).to_dict(orient='records')
    primary_pick = primary[0] if primary else {}

    return {
        'current_pick_number': context.current_pick_number,
        'next_pick_number': next_turn_pick,
        'roster_need': roster_need,
        'pick_now': primary_pick,
        'fallbacks': _recommendation_rows_to_records(fallbacks),
        'can_wait': _recommendation_rows_to_records(can_wait.head(5)),
        'wait_candidates': wait_survival,
        'best_roster_paths': best_roster_paths,
        'board_pressure': (
            decision_table.groupby('position')['player_name'].count().sort_values(
                ascending=False
            ).to_dict()
            if not decision_table.empty
            else {}
        ),
    }


def build_tier_cliffs(decision_table: pd.DataFrame) -> pd.DataFrame:
    """Compute position-consistent tier cliffs from the decision table."""
    rows = []
    for position, group in decision_table.groupby('position'):
        ordered = group.sort_values('proj_points_mean', ascending=False).reset_index(
            drop=True
        )
        if ordered.empty:
            continue
        diffs = ordered['proj_points_mean'].diff(-1).fillna(0.0)
        threshold = diffs.quantile(0.75) if len(diffs) > 3 else diffs.max()
        threshold = float(np.nan_to_num(threshold, nan=0.0))
        cliff_mask = (
            diffs >= threshold if threshold > 0 else pd.Series([False] * len(ordered))
        )
        for idx, row in ordered.iterrows():
            rows.append(
                {
                    'position': position,
                    'player_name': row['player_name'],
                    'draft_score': row['draft_score'],
                    'proj_points_mean': row['proj_points_mean'],
                    'tier_cliff_distance': float(diffs.iloc[idx]),
                    'is_tier_cliff': bool(cliff_mask.iloc[idx])
                    if len(cliff_mask) > idx
                    else False,
                    'draft_tier': row['draft_tier'],
                }
            )
    return pd.DataFrame(rows)


def build_roster_scenarios(
    decision_table: pd.DataFrame, league_settings: LeagueSettings
) -> pd.DataFrame:
    """Generate simple roster-construction archetypes and their utility tradeoffs."""
    archetypes = [
        ('balanced', {'RB': 1.0, 'WR': 1.0, 'QB': 0.8, 'TE': 0.8}),
        ('hero_rb', {'RB': 1.2, 'WR': 0.9, 'QB': 0.7, 'TE': 0.8}),
        ('zero_rb', {'RB': 0.8, 'WR': 1.2, 'QB': 0.7, 'TE': 0.8}),
        ('elite_qb', {'RB': 0.95, 'WR': 0.95, 'QB': 1.15, 'TE': 0.85}),
        ('tight_end_attack', {'RB': 0.95, 'WR': 0.95, 'QB': 0.8, 'TE': 1.15}),
    ]

    rows = []
    top = decision_table.head(80)
    for name, weights in archetypes:
        score = 0.0
        count = 0
        for pos, weight in weights.items():
            pos_df = top[top['position'] == pos].head(
                max(1, league_settings.roster_spots.get(pos, 0) + 1)
            )
            if pos_df.empty:
                continue
            score += float(pos_df['draft_score'].mean()) * weight
            count += 1
        rows.append(
            {
                'scenario': name,
                'utility_proxy': score / max(1, count),
                'mean_draft_score': float(top['draft_score'].mean()),
                'mean_upside_score': float(top['upside_score'].mean()),
                'mean_fragility_score': float(top['fragility_score'].mean()),
                'recommended_build': ', '.join(
                    [f'{pos}:{weight}' for pos, weight in weights.items()]
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values('utility_proxy', ascending=False)
        .reset_index(drop=True)
    )


def _starter_points_from_roster(
    team_roster: pd.DataFrame, league_settings: LeagueSettings
) -> float:
    if team_roster.empty:
        return 0.0

    def _take_best(
        pos: str, count: int, pool: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        if count <= 0 or pool.empty:
            return pool, 0.0
        chosen = pool.nlargest(count, 'actual_points')
        remaining = pool.drop(chosen.index)
        return remaining, float(chosen['actual_points'].sum())

    pool = team_roster.copy()
    total = 0.0
    for pos in ['QB', 'RB', 'WR', 'TE']:
        need = league_settings.roster_spots.get(pos, 0)
        pos_pool = pool[pool['position'] == pos]
        pool, subtotal = _take_best(pos, need, pos_pool)
        total += subtotal

    flex_slots = league_settings.roster_spots.get('FLEX', 0)
    if flex_slots > 0:
        flex_pool = team_roster[team_roster['position'].isin(['RB', 'WR', 'TE'])]
        flex_pool = flex_pool.loc[
            ~flex_pool.index.isin(team_roster.index.difference(flex_pool.index))
        ]
        if not flex_pool.empty:
            total += float(
                flex_pool.nlargest(flex_slots, 'actual_points')['actual_points'].sum()
            )
    return float(total)


def _team_actual_points(
    team_players: pd.DataFrame, league_settings: LeagueSettings
) -> float:
    if team_players.empty:
        return 0.0
    total = 0.0
    chosen = pd.Series(False, index=team_players.index)
    for pos, count in [
        ('QB', league_settings.roster_spots.get('QB', 0)),
        ('RB', league_settings.roster_spots.get('RB', 0)),
        ('WR', league_settings.roster_spots.get('WR', 0)),
        ('TE', league_settings.roster_spots.get('TE', 0)),
    ]:
        pos_pool = team_players[(team_players['position'] == pos) & (~chosen)]
        if pos_pool.empty or count <= 0:
            continue
        take = pos_pool.nlargest(count, 'actual_points')
        chosen.loc[take.index] = True
        total += float(take['actual_points'].sum())
    flex_slots = league_settings.roster_spots.get('FLEX', 0)
    if flex_slots > 0:
        flex_pool = team_players[
            (team_players['position'].isin(['RB', 'WR', 'TE'])) & (~chosen)
        ]
        if not flex_pool.empty:
            take = flex_pool.nlargest(flex_slots, 'actual_points')
            chosen.loc[take.index] = True
            total += float(take['actual_points'].sum())
    return float(total)


def _build_strategy_ranker(strategy_name: str):
    def _rank(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if strategy_name == 'market':
            frame['strategy_rank'] = frame['market_rank'].rank(
                method='first', ascending=True
            )
            frame['strategy_score'] = -frame['market_rank']
        elif strategy_name in {'vor', 'historical_vor_proxy'}:
            frame['strategy_rank'] = frame['replacement_delta'].rank(
                method='first', ascending=False
            )
            frame['strategy_score'] = frame['replacement_delta']
        elif strategy_name == 'consensus':
            frame['strategy_rank'] = (
                0.65 * frame['proj_points_mean'] + 0.35 * frame['availability_at_pick']
            ).rank(method='first', ascending=False)
            frame['strategy_score'] = (
                0.65 * frame['proj_points_mean'] + 0.35 * frame['availability_at_pick']
            )
        elif strategy_name == 'draft_score':
            frame['strategy_rank'] = frame['draft_score'].rank(
                method='first', ascending=False
            )
            frame['strategy_score'] = frame['draft_score']
        else:
            raise ValueError(f'Unknown strategy: {strategy_name}')
        return frame.sort_values(
            ['strategy_score', 'proj_points_mean'], ascending=[False, False]
        ).reset_index(drop=True)

    return _rank


def build_historical_vor_proxy_table(
    test_frame: pd.DataFrame, train_frame: pd.DataFrame
) -> pd.DataFrame:
    """Build a clearly labeled historical VOR proxy from pre-holdout information."""
    if test_frame is None or test_frame.empty:
        return pd.DataFrame(columns=['Season', 'Name', 'Position', 'historical_vor_proxy'])

    proxy = test_frame.copy()
    replacement_values: dict[str, float] = {}
    for position in proxy['Position'].dropna().unique():
        pos_train = pd.to_numeric(
            train_frame.loc[train_frame['Position'] == position, 'actual_points'],
            errors='coerce',
        ).dropna()
        if pos_train.empty:
            replacement_values[position] = 0.0
        else:
            slot_count = max(1, min(len(pos_train), 12))
            replacement_values[position] = float(
                pos_train.sort_values(ascending=False).reset_index(drop=True).iloc[
                    slot_count - 1
                ]
            )

    proxy['historical_vor_proxy'] = proxy.apply(
        lambda row: float(row['proj_points_mean'])
        - replacement_values.get(row['Position'], 0.0),
        axis=1,
    )
    proxy['historical_vor_proxy_rank'] = proxy.groupby('Position')[
        'historical_vor_proxy'
    ].rank(method='first', ascending=False)
    proxy['historical_vor_proxy_label'] = 'historical_vor_proxy'
    proxy['source_name'] = proxy.get('source_name', 'historical_vor_proxy')
    return proxy


def _draft_team_from_pool(
    available: pd.DataFrame,
    team_roster: list[dict[str, Any]],
    strategy_name: str,
    round_number: int,
    league_settings: LeagueSettings,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if available.empty:
        return available, team_roster

    frame = available.copy()
    ranker = _build_strategy_ranker(strategy_name)
    ranked = ranker(frame)

    # Prefer roster gaps first, then best available.
    roster_counts = (
        pd.Series([row['position'] for row in team_roster]).value_counts().to_dict()
    )
    for pos in ['QB', 'RB', 'WR', 'TE']:
        need = league_settings.roster_spots.get(pos, 0) - roster_counts.get(pos, 0)
        if need > 0:
            positional = ranked[ranked['position'] == pos]
            if not positional.empty:
                choice = positional.iloc[0]
                team_roster.append(choice.to_dict())
                return ranked[
                    ranked['player_name'] != choice['player_name']
                ].reset_index(drop=True), team_roster

    if round_number > league_settings.round_count() - 3:
        # Late rounds: lean into upside if the build is already stable.
        ranked = ranked.sort_values(
            ['upside_score', 'draft_score'], ascending=[False, False]
        )

    choice = ranked.iloc[0]
    team_roster.append(choice.to_dict())
    available = ranked[ranked['player_name'] != choice['player_name']].reset_index(
        drop=True
    )
    return available, team_roster


def run_draft_backtest(
    season_history: pd.DataFrame,
    league_settings: LeagueSettings | None = None,
    holdout_years: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Run a snake-draft backtest across historical seasons."""
    settings = league_settings or LeagueSettings()
    if season_history is None or season_history.empty:
        raise ValueError('season_history is empty')

    df = season_history.copy()
    required = {'Season', 'Name', 'Position', 'FantPt'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'season_history is missing required columns: {sorted(missing)}'
        )

    df['Season'] = pd.to_numeric(df['Season'], errors='coerce').astype('Int64')
    df['Name'] = df['Name'].map(_safe_string)
    df['Position'] = df['Position'].map(_normalize_position)
    df['FantPt'] = pd.to_numeric(df['FantPt'], errors='coerce')
    df = df.dropna(subset=['Season', 'Name', 'Position', 'FantPt']).copy()

    season_table = (
        df.groupby(['Season', 'Name', 'Position'], as_index=False)['FantPt']
        .mean()
        .rename(columns={'FantPt': 'actual_points'})
    )

    seasons = sorted(int(s) for s in season_table['Season'].unique())
    if holdout_years is None:
        holdout_years = seasons[2:] if len(seasons) > 2 else seasons[1:]
    holdout_years = [int(year) for year in holdout_years]
    if not holdout_years:
        raise ValueError('No holdout seasons available')

    by_season = []
    for holdout_year in holdout_years:
        train = season_table[season_table['Season'] < holdout_year].copy()
        test = season_table[season_table['Season'] == holdout_year].copy()
        if train.empty or test.empty:
            continue

        train_means = train.groupby(['Name', 'Position'], as_index=False).agg(
            train_mean=('actual_points', 'mean'),
            train_std=('actual_points', 'std'),
            season_count=('actual_points', 'count'),
        )
        latest = (
            train.sort_values('Season')
            .groupby(['Name', 'Position'], as_index=False)
            .tail(1)
        )
        latest = latest[['Name', 'Position', 'actual_points']].rename(
            columns={'actual_points': 'latest_points'}
        )
        test = test.merge(train_means, on=['Name', 'Position'], how='left')
        test = test.merge(latest, on=['Name', 'Position'], how='left')

        # Build a simple market proxy from the latest prior season plus uncertainty.
        test['proj_points_mean'] = (
            test['train_mean']
            .fillna(test['latest_points'])
            .fillna(train['actual_points'].mean())
        )
        test['std_projection'] = (
            test['train_std'].fillna(test['proj_points_mean'].std(ddof=0)).fillna(0.0)
        )
        test['adp'] = test.groupby('Position')['latest_points'].rank(
            method='first', ascending=False
        )
        test['adp'] = (
            test['adp'] + test['std_projection'].fillna(0.0).rank(method='first') * 0.15
        )
        test['uncertainty_score'] = _clamp(
            (
                test['std_projection'] / test['proj_points_mean'].replace(0, np.nan)
            ).fillna(0.15),
            0.0,
            1.0,
        )
        test['site_disagreement'] = _clamp(
            _zscore(test['std_projection']).rank(pct=True), 0.0, 1.0
        )
        test['adp_std'] = test['std_projection'].fillna(0.0)
        test['role_volatility'] = test['uncertainty_score']
        test['games_missed'] = 0.0
        test['age'] = 27.0
        test['years_in_league'] = test['season_count'].fillna(0.0)
        test['team_change'] = 0.0
        test['role_label'] = 'historical'
        test['source_name'] = f'historical_holdout_{holdout_year}'
        test['source_updated_at'] = pd.Timestamp(holdout_year, 1, 1)

        context = DraftContext(
            current_pick_number=settings.draft_position, drafted_players=set()
        )
        decision_table = build_decision_table(test, settings, context)
        historical_vor_proxy_table = build_historical_vor_proxy_table(test, train)
        if not historical_vor_proxy_table.empty:
            historical_vor_proxy_table = historical_vor_proxy_table.rename(
                columns={'Name': 'player_name', 'Position': 'position'}
            )
            decision_table = decision_table.merge(
                historical_vor_proxy_table[
                    [
                        'player_name',
                        'position',
                        'historical_vor_proxy',
                        'historical_vor_proxy_rank',
                    ]
                ],
                on=['player_name', 'position'],
                how='left',
            )
            decision_table['historical_vor_proxy'] = decision_table[
                'historical_vor_proxy'
            ].fillna(decision_table['replacement_delta'])
        else:
            decision_table['historical_vor_proxy'] = decision_table['replacement_delta']
        by_strategy = {}
        for strategy in ['market', 'historical_vor_proxy', 'consensus', 'draft_score']:
            available = decision_table.copy()
            team_rosters = [[] for _ in range(settings.league_size)]
            picks_per_team = settings.round_count()
            draft_order = []
            for round_number in range(1, picks_per_team + 1):
                if round_number % 2 == 1:
                    order = list(range(settings.league_size))
                else:
                    order = list(reversed(range(settings.league_size)))
                draft_order.extend(order)

            available_pool = available.copy()
            for pick_index, team_idx in enumerate(
                draft_order[: settings.league_size * picks_per_team], start=1
            ):
                round_number = math.ceil(pick_index / settings.league_size)
                if team_idx == settings.draft_position - 1:
                    available_pool, team_rosters[team_idx] = _draft_team_from_pool(
                        available_pool,
                        team_rosters[team_idx],
                        strategy,
                        round_number,
                        settings,
                    )
                else:
                    available_pool, team_rosters[team_idx] = _draft_team_from_pool(
                        available_pool,
                        team_rosters[team_idx],
                        'market',
                        round_number,
                        settings,
                    )
                if available_pool.empty:
                    break

            our_roster = (
                pd.DataFrame(team_rosters[settings.draft_position - 1])
                if team_rosters[settings.draft_position - 1]
                else pd.DataFrame(columns=decision_table.columns)
            )
            if not our_roster.empty and 'actual_points' not in our_roster.columns:
                our_roster['actual_points'] = (
                    test.set_index(['Name', 'Position'])
                    .reindex(
                        pd.MultiIndex.from_frame(our_roster[['Name', 'Position']])
                    )['actual_points']
                    .to_numpy()
                )
            roster_points = _team_actual_points(our_roster, settings)
            lineup_points = _starter_points_from_roster(our_roster, settings)
            market_truth = test.copy()
            market_truth['baseline_rank'] = market_truth['proj_points_mean'].rank(
                method='first', ascending=False
            )
            market_truth['draft_score_rank'] = decision_table['draft_score'].rank(
                method='first', ascending=False
            )

            metric_rank = {}
            for metric_name, score_col in [
                ('market', 'market_rank'),
                ('historical_vor_proxy', 'historical_vor_proxy'),
                ('consensus', 'proj_points_mean'),
                ('draft_score', 'draft_score'),
            ]:
                test_eval = test.copy()
                test_eval['player_key'] = (
                    test_eval['Name'] + '||' + test_eval['Position']
                )
                pred = decision_table.copy()
                pred['player_key'] = pred['player_name'] + '||' + pred['position']
                if score_col not in pred.columns:
                    score_col = 'draft_score'
                pred_scores = pred[['player_key', score_col]].rename(
                    columns={score_col: 'predicted_score'}
                )
                merged = test_eval.merge(pred_scores, on='player_key', how='left')
                if merged['predicted_score'].notna().sum() > 1:
                    if metric_name == 'market':
                        corr_input = -merged['predicted_score']
                    else:
                        corr_input = merged['predicted_score']
                    corr = float(
                        spearmanr(corr_input, merged['actual_points']).correlation
                    )
                else:
                    corr = 0.0
                metric_rank[metric_name] = float(np.nan_to_num(corr, nan=0.0))

            by_strategy[strategy] = {
                'strategy': strategy,
                'holdout_year': holdout_year,
                'our_team_actual_points': float(roster_points),
                'our_team_lineup_points': float(lineup_points),
                'predicted_rank_correlation': metric_rank.get(strategy, 0.0),
                'drafted_players': [
                    row.get('player_name', '')
                    for row in team_rosters[settings.draft_position - 1]
                ],
                'position_counts': pd.Series(
                    [
                        row.get('position', 'UNKNOWN')
                        for row in team_rosters[settings.draft_position - 1]
                    ]
                )
                .value_counts()
                .to_dict(),
                'strategy_label': strategy,
            }

        best_strategy = max(
            by_strategy.values(), key=lambda item: item['our_team_lineup_points']
        )
        by_season.append(
            {
                'holdout_year': holdout_year,
                'by_strategy': by_strategy,
                'winner': best_strategy['strategy'],
                'lineup_points_winner': float(best_strategy['our_team_lineup_points']),
            }
        )

    if not by_season:
        raise ValueError('Unable to run backtest with the provided seasons')

    overall_rows = []
    for strategy in ['market', 'historical_vor_proxy', 'consensus', 'draft_score']:
        season_values = [
            season['by_strategy'][strategy]['our_team_lineup_points']
            for season in by_season
            if strategy in season['by_strategy']
        ]
        if not season_values:
            continue
        overall_rows.append(
            {
                'strategy': strategy,
                'mean_lineup_points': float(np.mean(season_values)),
                'median_lineup_points': float(np.median(season_values)),
                'season_count': len(season_values),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values(
        'mean_lineup_points', ascending=False
    )
    winner = (
        _pick_first_row(overall['strategy']) if not overall.empty else 'draft_score'
    )

    return {
        'model_type': 'draft_decision_backtest',
        'league_settings': settings.to_dict(),
        'holdout_years': holdout_years,
        'by_season': by_season,
        'overall': {'by_strategy': overall.to_dict(orient='records'), 'winner': winner},
    }


def build_dashboard_payload(
    decision_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    tier_cliffs: pd.DataFrame,
    roster_scenarios: pd.DataFrame,
    source_freshness: pd.DataFrame,
    league_settings: LeagueSettings,
    backtest: dict[str, Any] | None = None,
    context: DraftContext | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable payload for the local dashboard."""
    backtest = backtest or {}
    context = context or DraftContext(
        current_pick_number=league_settings.draft_position
    )
    live_state = build_live_recommendation_snapshot(
        decision_table,
        recommendations,
        roster_scenarios,
        league_settings,
        context,
    )
    recommendation_inputs_cols = [
        'player_name',
        'position',
        'proj_points_mean',
        'proj_points_floor',
        'proj_points_ceiling',
        'adp',
        'adp_std',
        'market_rank',
        'availability_at_pick',
        'availability_to_next_pick',
        'starter_delta',
        'replacement_delta',
        'upside_score',
        'fragility_score',
        'draft_score',
        'draft_tier',
        'why_flags',
        'current_pick_utility',
        'wait_utility',
        'position_run_risk',
        'roster_fit_score',
        'expected_regret',
        'recommendation_lane',
        'lane_rank',
        'pick_mode',
        'rationale',
    ]
    recommendation_source = recommendations if not recommendations.empty else decision_table
    recommendation_inputs = recommendation_source[
        [
            column
            for column in recommendation_inputs_cols
            if column in recommendation_source.columns
        ]
    ].copy()
    recommendation_summary = recommendations.head(12).to_dict(orient='records')
    position_summary = (
        decision_table.groupby('position')
        .agg(
            player_count=('player_name', 'count'),
            mean_draft_score=('draft_score', 'mean'),
            mean_availability=('availability_at_pick', 'mean'),
            mean_upside=('upside_score', 'mean'),
            mean_fragility=('fragility_score', 'mean'),
            mean_proj=('proj_points_mean', 'mean'),
        )
        .reset_index()
        .to_dict(orient='records')
    )
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'league_settings': league_settings.to_dict(),
        'current_pick_number': context.current_pick_number,
        'next_pick_number': next_pick_number(
            context.current_pick_number,
            league_settings.draft_position,
            league_settings.league_size,
        ),
        'current_draft_context_defaults': {
            'current_pick_number': context.current_pick_number,
            'next_pick_number': live_state['next_pick_number'],
            'taken_players': sorted(context.drafted_set()),
            'your_players': sorted(context.your_players),
            'roster_counts': context.roster_counts,
        },
        'recommendation_inputs': recommendation_inputs.to_dict(orient='records'),
        'recommendation_summary': recommendation_summary,
        'live_state': live_state,
        'decision_table': decision_table.to_dict(orient='records'),
        'position_summary': position_summary,
        'tier_cliffs': tier_cliffs.to_dict(orient='records'),
        'roster_scenarios': roster_scenarios.to_dict(orient='records'),
        'source_freshness': source_freshness.to_dict(orient='records'),
        'backtest': backtest,
        'supporting_math': {
            'draft_score_mean': float(decision_table['draft_score'].mean())
            if not decision_table.empty
            else 0.0,
            'draft_score_std': float(decision_table['draft_score'].std(ddof=0))
            if not decision_table.empty
            else 0.0,
            'availability_mean': float(decision_table['availability_at_pick'].mean())
            if not decision_table.empty
            else 0.0,
            'top_draft_score': float(decision_table['draft_score'].max())
            if not decision_table.empty
            else 0.0,
        },
    }


def _format_sheet(worksheet: openpyxl.worksheet.worksheet.Worksheet) -> None:
    header_fill = PatternFill(
        start_color='173F5F', end_color='173F5F', fill_type='solid'
    )
    header_font = Font(bold=True, color='FFFFFF', size=11)
    body_font = Font(size=10)
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')
    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = body_font
            cell.alignment = Alignment(
                horizontal='left', vertical='top', wrap_text=True
            )
    for column in worksheet.columns:
        max_len = 0
        for cell in column:
            try:
                max_len = max(max_len, len(str(cell.value)))
            except Exception:
                continue
        worksheet.column_dimensions[get_column_letter(column[0].column)].width = min(
            max_len + 2, 48
        )


def _write_dataframe_sheet(
    workbook: openpyxl.Workbook, title: str, df: pd.DataFrame
) -> None:
    ws = workbook.create_sheet(title)
    if df is None or df.empty:
        ws.append(['No data available'])
        return
    cols = list(df.columns)
    ws.append(cols)
    for _, row in df.iterrows():
        ws.append([row[col] for col in cols])
    _format_sheet(ws)


def export_workbook(
    decision_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    tier_cliffs: pd.DataFrame,
    roster_scenarios: pd.DataFrame,
    source_freshness: pd.DataFrame,
    output_path: Path | str,
    league_settings: LeagueSettings,
    backtest: dict[str, Any] | None = None,
    context: DraftContext | None = None,
) -> Path:
    """Write the portable draft workbook."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    board_cols = [
        'player_name',
        'position',
        'proj_points_mean',
        'proj_points_floor',
        'proj_points_ceiling',
        'adp',
        'market_rank',
        'availability_at_pick',
        'replacement_delta',
        'starter_delta',
        'upside_score',
        'fragility_score',
        'draft_score',
        'draft_rank',
        'draft_tier',
        'why_flags',
    ]
    board = decision_table[
        [col for col in board_cols if col in decision_table.columns]
    ].copy()
    _write_dataframe_sheet(wb, 'Big Board', board)

    by_position = decision_table.sort_values(
        ['position', 'draft_score'], ascending=[True, False]
    ).copy()
    _write_dataframe_sheet(wb, 'By Position', by_position)

    live_summary = pd.DataFrame(
        [
            {
                'metric': 'current_pick_number',
                'value': context.current_pick_number if context else league_settings.draft_position,
            },
            {
                'metric': 'next_pick_number',
                'value': next_pick_number(
                    context.current_pick_number
                    if context
                    else league_settings.draft_position,
                    league_settings.draft_position,
                    league_settings.league_size,
                ),
            },
            {
                'metric': 'draft_position',
                'value': league_settings.draft_position,
            },
            {
                'metric': 'league_size',
                'value': league_settings.league_size,
            },
        ]
    )
    _write_dataframe_sheet(wb, 'Live Context', live_summary)

    pick_now = recommendations[
        recommendations.get('recommendation_lane', pd.Series(dtype=object)) == 'pick_now'
    ].copy()
    if pick_now.empty and not recommendations.empty:
        pick_now = recommendations.head(1).copy()
    _write_dataframe_sheet(wb, 'Pick Now', pick_now)

    fallbacks = recommendations[
        recommendations.get('recommendation_lane', pd.Series(dtype=object)) == 'fallback'
    ].copy()
    _write_dataframe_sheet(wb, 'Fallback Ladder', fallbacks)

    can_wait = recommendations[
        recommendations.get('recommendation_lane', pd.Series(dtype=object)) == 'can_wait'
    ].copy()
    _write_dataframe_sheet(wb, 'Can Wait', can_wait)

    _write_dataframe_sheet(wb, 'My Picks', recommendations.copy())

    _write_dataframe_sheet(wb, 'Tier Cliffs', tier_cliffs)
    availability = decision_table[
        [
            'player_name',
            'position',
            'adp',
            'availability_at_pick',
            'draft_score',
            'why_flags',
        ]
    ].copy()
    _write_dataframe_sheet(wb, 'Availability', availability)

    round_rows = []
    current_round_count = league_settings.round_count()
    for rnd in range(1, current_round_count + 1):
        row = {'round': rnd}
        cut = max(1, league_settings.league_size * rnd)
        top = decision_table.head(cut)
        row['best_player'] = _pick_first_row(top['player_name'])
        row['best_position'] = _pick_first_row(top['position'])
        row['mean_draft_score'] = (
            float(top['draft_score'].mean()) if not top.empty else np.nan
        )
        round_rows.append(row)
    _write_dataframe_sheet(wb, 'Targets By Round', pd.DataFrame(round_rows))

    _write_dataframe_sheet(wb, 'Roster Construction Scenarios', roster_scenarios)
    _write_dataframe_sheet(
        wb,
        'Player Notes',
        decision_table[
            [
                'player_name',
                'position',
                'why_flags',
                'draft_tier',
                'fragility_score',
                'upside_score',
            ]
        ],
    )

    diagnostics = pd.DataFrame(
        [
            {'metric': 'player_count', 'value': int(len(decision_table))},
            {'metric': 'recommendation_count', 'value': int(len(recommendations))},
            {
                'metric': 'draft_score_mean',
                'value': float(decision_table['draft_score'].mean()),
            },
            {
                'metric': 'draft_score_std',
                'value': float(decision_table['draft_score'].std(ddof=0)),
            },
            {
                'metric': 'top_draft_score',
                'value': float(decision_table['draft_score'].max()),
            },
            {
                'metric': 'availability_mean',
                'value': float(decision_table['availability_at_pick'].mean()),
            },
            {
                'metric': 'source_count',
                'value': int(source_freshness['source_name'].nunique())
                if not source_freshness.empty
                else 0,
            },
        ]
    )
    if backtest:
        diagnostics = pd.concat(
            [
                diagnostics,
                pd.DataFrame(
                    [
                        {
                            'metric': 'backtest_model',
                            'value': backtest.get('model_type', ''),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    _write_dataframe_sheet(wb, 'Model Diagnostics', diagnostics)
    _write_dataframe_sheet(wb, 'Source Freshness', source_freshness)

    # Add a compact backtest summary if available.
    if backtest and backtest.get('overall'):
        summary = pd.DataFrame(backtest['overall'].get('by_strategy', []))
        if not summary.empty:
            _write_dataframe_sheet(wb, 'Backtest Summary', summary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def _build_charts(
    decision_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    league_settings: LeagueSettings,
) -> list[Any]:
    import plotly.graph_objects as go

    top = decision_table.head(120).copy()
    fig1 = go.Figure()
    for position in sorted(top['position'].dropna().unique()):
        sub = top[top['position'] == position]
        fig1.add_trace(
            go.Scatter(
                x=sub['adp'],
                y=sub['draft_score'],
                mode='markers',
                name=position,
                text=sub['player_name'],
                marker=dict(size=9, opacity=0.8),
            )
        )
    fig1.update_layout(
        title='ADP vs Draft Score',
        xaxis_title='ADP',
        yaxis_title='Draft Score',
        height=420,
    )

    fig2 = go.Figure()
    if not recommendations.empty:
        fig2.add_trace(
            go.Bar(
                x=recommendations['player_name'].head(10),
                y=recommendations['availability_to_next_pick'].head(10),
                name='Survival to next pick',
            )
        )
    fig2.update_layout(
        title='Availability by Pick', yaxis_title='Probability', height=420
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=top['fragility_score'],
            y=top['upside_score'],
            mode='markers',
            text=top['player_name'],
            marker=dict(
                color=top['draft_score'], colorscale='Viridis', showscale=True, size=10
            ),
        )
    )
    fig3.update_layout(
        title='Upside vs Fragility',
        xaxis_title='Fragility',
        yaxis_title='Upside',
        height=420,
    )

    return [fig1, fig2, fig3]


def export_dashboard_html(
    decision_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_path: Path | str,
    league_settings: LeagueSettings,
    backtest: dict[str, Any] | None = None,
    source_freshness: pd.DataFrame | None = None,
    dashboard_payload: dict[str, Any] | None = None,
) -> Path:
    """Write a local interactive HTML dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backtest = backtest or {}
    source_freshness = (
        source_freshness if source_freshness is not None else pd.DataFrame()
    )
    payload = dashboard_payload or build_dashboard_payload(
        decision_table,
        recommendations,
        build_tier_cliffs(decision_table),
        build_roster_scenarios(decision_table, league_settings),
        source_freshness,
        league_settings,
        backtest=backtest,
        context=DraftContext(current_pick_number=league_settings.draft_position),
    )
    payload_json = json.dumps(payload, default=str)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FFBayes Draft Command Center</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #081120;
      --panel: rgba(15, 23, 42, 0.86);
      --panel-strong: rgba(17, 24, 39, 0.96);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.16);
      --border: rgba(148, 163, 184, 0.18);
      --good: #34d399;
      --warn: #f59e0b;
      --bad: #f87171;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(99, 102, 241, 0.12), transparent 24%),
        linear-gradient(180deg, #050b16 0%, #0b1324 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      padding: 20px;
    }}
    h1, h2, h3, h4, p {{ margin: 0; }}
    .shell {{
      max-width: 1600px;
      margin: 0 auto;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .hero-card, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(10px);
    }}
    .hero-card {{
      padding: 22px;
    }}
    .hero-title {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .hero-title h1 {{
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .muted {{ color: var(--muted); }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .pill {{
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #c9f0ff;
      border: 1px solid rgba(56, 189, 248, 0.24);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .card.strong {{
      background: var(--panel-strong);
    }}
    .card h2 {{
      font-size: 18px;
      margin-bottom: 8px;
    }}
    .card h3 {{
      font-size: 15px;
      margin-bottom: 8px;
    }}
    .section {{
      margin-top: 14px;
    }}
    .subtle {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .recommendation-primary {{
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }}
    .primary-name {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .primary-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(148, 163, 184, 0.12);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .metric .label {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 3px;
    }}
    .metric .value {{
      font-size: 15px;
      font-weight: 600;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.12);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      color: var(--text);
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.10);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #8cd3ff;
      background: rgba(15, 23, 42, 0.8);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .search-row {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    input[type="search"] {{
      flex: 1 1 260px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--text);
      border-radius: 12px;
      padding: 11px 12px;
    }}
    button {{
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      cursor: pointer;
    }}
    button:hover {{
      background: rgba(255, 255, 255, 0.09);
    }}
    .action-bar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .list {{
      display: grid;
      gap: 10px;
    }}
    .list-item {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(148, 163, 184, 0.12);
      border-radius: 14px;
    }}
    .item-title {{
      font-weight: 600;
    }}
    .item-subtitle {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }}
    .lane {{
      display: grid;
      gap: 8px;
    }}
    .lane .lane-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }}
    .tiny {{
      font-size: 12px;
    }}
    .flag {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: rgba(148, 163, 184, 0.14);
      color: #dbeafe;
      margin-right: 6px;
      margin-top: 4px;
      font-size: 11px;
    }}
    .good {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    .responsive-hide {{ display: block; }}
    @media (max-width: 1100px) {{
      .hero, .grid, .two-col, .metric-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-card">
        <div class="hero-title">
          <h1>FFBayes Draft Command Center</h1>
          <span class="pill">Live pre-draft board</span>
        </div>
        <p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | League {league_settings.league_size}-team | Draft position {league_settings.draft_position}</p>
        <div class="meta" id="status-pills"></div>
      </div>
      <div class="hero-card">
        <div class="subtle">Current turn</div>
        <div class="primary-name" id="current-turn-label">Pick now</div>
        <div class="primary-meta" id="turn-meta"></div>
      </div>
    </div>

    <div class="grid">
      <div class="stack">
        <div class="card strong">
          <h2>Primary Recommendation</h2>
          <div class="subtle">The single pick we want on this turn.</div>
          <div class="recommendation-primary" id="primary-card"></div>
        </div>

        <div class="two-col">
          <div class="card">
            <h2>Fallback Ladder</h2>
            <div class="subtle">If the top target is gone, these are the next-best alternatives.</div>
            <div class="list" id="fallback-list"></div>
          </div>
          <div class="card">
            <h2>Can Wait</h2>
            <div class="subtle">Players with the strongest chance to survive to the next pick.</div>
            <div class="list" id="wait-list"></div>
          </div>
        </div>

        <div class="card">
          <h2>Draft Board</h2>
          <div class="subtle">Search, mark off taken players, mark your picks, and undo the last action.</div>
          <div class="search-row">
            <input id="player-search" type="search" placeholder="Search players by name or position" />
            <button type="button" id="undo-button">Undo last action</button>
            <button type="button" id="advance-button">Advance pick</button>
          </div>
          <div class="action-bar" id="board-actions"></div>
          <div class="table-wrap" style="max-height: 460px; margin-top: 12px;">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Pos</th>
                  <th>Score</th>
                  <th>Next pick survival</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="board-table"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="card">
          <h2>Roster & Construction</h2>
          <div class="subtle">Current needs, tier cliffs, and position-run pressure.</div>
          <div class="metric-grid" id="roster-metrics"></div>
          <div class="section">
            <h3>Best roster paths</h3>
            <div class="list" id="roster-paths"></div>
          </div>
          <div class="section">
            <h3>Tier cliffs</h3>
            <div class="table-wrap" style="max-height: 240px;">
              <table>
                <thead><tr><th>Pos</th><th>Player</th><th>Cliff</th></tr></thead>
                <tbody id="tier-cliff-table"></tbody>
              </table>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>Support Panel</h2>
          <div class="subtle">Math, freshness, and explanation stats behind the recommendation.</div>
          <div class="metric-grid" id="support-metrics"></div>
          <div class="section">
            <h3>Source freshness</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Source</th><th>Age</th><th>Rows</th></tr></thead>
                <tbody id="freshness-table"></tbody>
              </table>
            </div>
          </div>
          <div class="section">
            <h3>Backtest snapshot</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Strategy</th><th>Mean lineup points</th></tr></thead>
                <tbody id="backtest-table"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    window.FFBAYES_DASHBOARD = {payload_json};

    (() => {{
      const data = window.FFBAYES_DASHBOARD;
      const state = {{
        takenPlayers: new Set((data.live_state && data.live_state.taken_players) || []),
        yourPlayers: new Set((data.live_state && data.live_state.your_players) || []),
        rosterCounts: Object.assign({{}}, (data.live_state && data.live_state.roster_counts) || {{}}),
        history: Array.isArray(data.live_state && data.live_state.action_history)
          ? [...data.live_state.action_history]
          : [],
        currentPickNumber: (data.live_state && data.live_state.current_pick_number) || data.current_pick_number || 1,
        leagueSize: (data.league_settings && data.league_settings.league_size) || 10,
        draftPosition: (data.league_settings && data.league_settings.draft_position) || 10,
        search: '',
      }};

      const rosterTemplate = Object.assign({{}}, (data.league_settings && data.league_settings.roster_spots) || {{}});
      const flexWeights = Object.assign({{}}, (data.league_settings && data.league_settings.flex_weights) || {{}});
      const decisionTable = Array.isArray(data.decision_table) ? data.decision_table : [];
      const tierCliffs = Array.isArray(data.tier_cliffs) ? data.tier_cliffs : [];
      const rosterScenarios = Array.isArray(data.roster_scenarios) ? data.roster_scenarios : [];
      const freshness = Array.isArray(data.source_freshness) ? data.source_freshness : [];
      const backtest = data.backtest && data.backtest.overall && Array.isArray(data.backtest.overall.by_strategy)
        ? data.backtest.overall.by_strategy
        : [];

      function safeLower(value) {{
        return (value || '').toString().trim().toLowerCase();
      }}

      function nextPickNumber(currentPickNumber, draftPosition, leagueSize) {{
        const current = Math.max(1, Number(currentPickNumber) || 1);
        const draft = Math.max(1, Number(draftPosition) || 1);
        const size = Math.max(1, Number(leagueSize) || 1);
        const rounds = Math.max(1, Math.ceil(current / size) + 2);
        for (let roundNum = 1; roundNum <= rounds; roundNum += 1) {{
          const pick = roundNum % 2 === 1
            ? (roundNum - 1) * size + draft
            : roundNum * size - draft + 1;
          if (pick > current) {{
            return pick;
          }}
        }}
        return current + size;
      }}

      function availabilityProbability(adp, targetPick, adpStd, uncertaintyScore) {{
        const adpValue = Number(adp);
        if (!Number.isFinite(adpValue)) return 0.5;
        let spread = Number(adpStd);
        if (!Number.isFinite(spread) || spread <= 0) spread = 2.5;
        const uncertainty = Number(uncertaintyScore);
        if (Number.isFinite(uncertainty)) spread += 2.0 * uncertainty;
        const z = (adpValue - Number(targetPick)) / Math.max(1, spread);
        const prob = 1 / (1 + Math.exp(-z));
        return Math.max(0, Math.min(1, prob));
      }}

      function rosterNeed() {{
        const need = {{}};
        ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K'].forEach((pos) => {{
          need[pos] = Math.max(0, (rosterTemplate[pos] || 0) - (state.rosterCounts[pos] || 0));
        }});
        return need;
      }}

      function positionNeedCore() {{
        const need = rosterNeed();
        return ['QB', 'RB', 'WR', 'TE'].reduce((acc, pos) => acc + need[pos], 0) || 1;
      }}

      function computeRow(row, bestNow, positionCounts) {{
        const nextTurnPick = nextPickNumber(state.currentPickNumber, state.draftPosition, state.leagueSize);
        const availToNext = availabilityProbability(row.adp, nextTurnPick, row.adp_std, row.uncertainty_score);
        const need = rosterNeed();
        const totalNeed = positionNeedCore();
        const posNeed = need[row.position] || 0;
        const remaining = positionCounts[row.position] || 0;
        const demand = Math.max(1, state.leagueSize * Math.max(1, posNeed));
        const positionRunRisk = Math.max(0, Math.min(1, 1 - remaining / demand));
        const rosterFitScore = posNeed / totalNeed;
        const expectedRegret = Math.max(0, (bestNow - (Number(row.draft_score) || 0))) * (1 - availToNext) + 0.25 * positionRunRisk;
        const currentPickUtility =
          (Number(row.draft_score) || 0)
          + 0.15 * rosterFitScore
          + 0.10 * positionRunRisk
          - 0.20 * (1 - availToNext)
          + 0.04 * (Number(row.upside_score) || 0);
        const waitUtility =
          (Number(row.draft_score) || 0) * availToNext
          + 0.18 * (Number(row.upside_score) || 0)
          - 0.15 * (Number(row.fragility_score) || 0);
        return Object.assign({{}}, row, {{
          availability_to_next_pick: availToNext,
          position_run_risk: positionRunRisk,
          roster_fit_score: rosterFitScore,
          expected_regret: expectedRegret,
          current_pick_utility: currentPickUtility,
          wait_utility: waitUtility,
        }});
      }}

      function availableRows() {{
        const query = safeLower(state.search);
        const rawRows = decisionTable
          .filter((row) => !state.takenPlayers.has(safeLower(row.player_name)) && !state.yourPlayers.has(safeLower(row.player_name)))
          .filter((row) => !query || safeLower(row.player_name).includes(query) || safeLower(row.position).includes(query));
        const positionCounts = rawRows.reduce((counts, row) => {{
          counts[row.position] = (counts[row.position] || 0) + 1;
          return counts;
        }}, {{}});
        const bestNow = rawRows.reduce((best, row) => Math.max(best, Number(row.draft_score) || 0), 0);
        return rawRows.map((row) => computeRow(row, bestNow, positionCounts));
      }}

      function recommendedPanels() {{
        const rows = availableRows();
        const sortedNow = [...rows].sort((a, b) => (b.current_pick_utility - a.current_pick_utility) || (b.draft_score - a.draft_score));
        const sortedWait = [...rows].sort((a, b) => (b.wait_utility - a.wait_utility) || (b.availability_to_next_pick - a.availability_to_next_pick));
        const pickNow = sortedNow.slice(0, 1);
        const fallbacks = sortedNow.slice(1, 6);
        const canWait = sortedWait.filter((row) => !pickNow.some((pick) => safeLower(pick.player_name) === safeLower(row.player_name))).slice(0, 5);
        return {{ pickNow, fallbacks, canWait, rows }};
      }}

      function fmtPct(value) {{
        return `${{Math.round((Number(value) || 0) * 100)}}%`;
      }}

      function fmtNum(value, digits = 2) {{
        const num = Number(value);
        return Number.isFinite(num) ? num.toFixed(digits) : '0.00';
      }}

      function flagsHtml(row) {{
        const flags = safeLower(row.why_flags).split('|').filter(Boolean);
        return flags.map((flag) => `<span class="flag">${{flag}}</span>`).join('');
      }}

      function recommendationCard(row, leadLabel) {{
        if (!row) {{
          return `<div class="muted">No available recommendation.</div>`;
        }}
        return `
          <div class="primary-meta">
            <span class="pill">${{leadLabel}}</span>
            <span class="pill">${{row.position}}</span>
            <span class="pill">score ${{fmtNum(row.draft_score)}}</span>
          </div>
          <div class="primary-name">${{row.player_name}}</div>
          <div class="metric-grid">
            <div class="metric"><span class="label">Draft score</span><span class="value">${{fmtNum(row.draft_score)}}</span></div>
            <div class="metric"><span class="label">Availability to next pick</span><span class="value">${{fmtPct(row.availability_to_next_pick)}}</span></div>
            <div class="metric"><span class="label">Expected regret of waiting</span><span class="value">${{fmtNum(row.expected_regret)}}</span></div>
            <div class="metric"><span class="label">Roster fit</span><span class="value">${{fmtNum(row.roster_fit_score)}}</span></div>
            <div class="metric"><span class="label">Position-run risk</span><span class="value">${{fmtNum(row.position_run_risk)}}</span></div>
            <div class="metric"><span class="label">Why it matters</span><span class="value tiny">${{row.rationale || row.why_flags || 'No rationale'}}</span></div>
          </div>
          <div>${{flagsHtml(row)}}</div>
        `;
      }}

      function renderTurnMeta(panel) {{
        const row = panel.pickNow[0];
        const nextTurnPick = nextPickNumber(state.currentPickNumber, state.draftPosition, state.leagueSize);
        const lead = row ? `${{row.player_name}} • ${{row.position}}` : 'No current target';
        document.getElementById('current-turn-label').textContent = lead;
        document.getElementById('turn-meta').innerHTML = `
          <span class="pill">Current pick ${{state.currentPickNumber}}</span>
          <span class="pill">Next pick ${{nextTurnPick}}</span>
          <span class="pill">Taken ${{state.takenPlayers.size}}</span>
          <span class="pill">Yours ${{state.yourPlayers.size}}</span>
        `;
        document.getElementById('primary-card').innerHTML = recommendationCard(row, 'Pick now');
      }}

      function renderList(containerId, rows, emptyLabel, accentLabel) {{
        const container = document.getElementById(containerId);
        if (!rows.length) {{
          container.innerHTML = `<div class="muted">${{emptyLabel}}</div>`;
          return;
        }}
        container.innerHTML = rows.map((row) => `
          <div class="list-item">
            <div>
              <div class="item-title">${{row.player_name}} <span class="muted">• ${{row.position}}</span></div>
              <div class="item-subtitle">
                score ${{fmtNum(row.draft_score)}} • survival ${{fmtPct(row.availability_to_next_pick)}} • regret ${{fmtNum(row.expected_regret)}}
              </div>
            </div>
            <span class="pill">${{accentLabel}}</span>
          </div>
        `).join('');
      }}

      function renderBoardActions() {{
        const container = document.getElementById('board-actions');
        const topPlayers = availableRows().slice(0, 5);
        container.innerHTML = topPlayers.map((row) => `
          <button type="button" data-player="${{row.player_name}}" data-action="taken">${{row.player_name}} taken</button>
          <button type="button" data-player="${{row.player_name}}" data-action="mine">Mine</button>
        `).join('');
        container.querySelectorAll('button').forEach((button) => {{
          button.addEventListener('click', () => {{
            const player = button.getAttribute('data-player');
            const action = button.getAttribute('data-action');
            applyAction(action, player);
          }});
        }});
      }}

      function renderBoardTable(rows) {{
        const tbody = document.getElementById('board-table');
        if (!rows.length) {{
          tbody.innerHTML = '<tr><td colspan="5" class="muted">No players match the current filter.</td></tr>';
          return;
        }}
        tbody.innerHTML = rows.slice(0, 30).map((row) => `
          <tr>
            <td>
              <div><strong>${{row.player_name}}</strong></div>
              <div class="muted tiny">${{row.position}} • ${{row.pick_mode || ''}} ${{row.recommendation_lane || ''}}</div>
            </td>
            <td>${{row.position}}</td>
            <td>${{fmtNum(row.draft_score)}}</td>
            <td>${{fmtPct(row.availability_to_next_pick)}}</td>
            <td>
              <button type="button" data-player="${{row.player_name}}" data-action="taken">Taken</button>
              <button type="button" data-player="${{row.player_name}}" data-action="mine">Mine</button>
            </td>
          </tr>
        `).join('');
        tbody.querySelectorAll('button').forEach((button) => {{
          button.addEventListener('click', () => {{
            applyAction(button.getAttribute('data-action'), button.getAttribute('data-player'));
          }});
        }});
      }}

      function renderRosterPanel(panel) {{
        const need = rosterNeed();
        const metrics = document.getElementById('roster-metrics');
        metrics.innerHTML = Object.entries(need).map(([key, value]) => `
          <div class="metric">
            <span class="label">${{key}} need</span>
            <span class="value">${{value}}</span>
          </div>
        `).join('');

        document.getElementById('roster-paths').innerHTML = (data.live_state && data.live_state.best_roster_paths || []).map((row) => `
          <div class="list-item">
            <div>
              <div class="item-title">${{row.scenario || 'scenario'}}</div>
              <div class="item-subtitle">${{row.recommended_build || ''}}</div>
            </div>
            <span class="pill">${{fmtNum(row.utility_proxy)}}</span>
          </div>
        `).join('') || '<div class="muted">No roster paths available.</div>';

        document.getElementById('tier-cliff-table').innerHTML = tierCliffs.slice(0, 8).map((row) => `
          <tr>
            <td>${{row.position}}</td>
            <td>${{row.player_name}}</td>
            <td>${{fmtNum(row.tier_cliff_distance)}}</td>
          </tr>
        `).join('') || '<tr><td colspan="3" class="muted">No tier cliff data.</td></tr>';
      }}

      function renderSupport(panel) {{
        const support = document.getElementById('support-metrics');
        const supporting = data.supporting_math || {{}};
        const live = panel.pickNow[0] || {{}};
        const metrics = [
          ['Draft score mean', fmtNum(supporting.draft_score_mean)],
          ['Draft score std', fmtNum(supporting.draft_score_std)],
          ['Availability mean', fmtPct(supporting.availability_mean)],
          ['Top score', fmtNum(supporting.top_draft_score)],
          ['Next pick', String(nextPickNumber(state.currentPickNumber, state.draftPosition, state.leagueSize))],
          ['Current turn', String(state.currentPickNumber)],
        ];
        support.innerHTML = metrics.map(([label, value]) => `
          <div class="metric">
            <span class="label">${{label}}</span>
            <span class="value">${{value}}</span>
          </div>
        `).join('');

        document.getElementById('freshness-table').innerHTML = freshness.map((row) => `
          <tr>
            <td>${{row.source_name}}</td>
            <td>${{row.freshness_days}}</td>
            <td>${{row.row_count}}</td>
          </tr>
        `).join('') || '<tr><td colspan="3" class="muted">No freshness data.</td></tr>';

        document.getElementById('backtest-table').innerHTML = backtest.map((row) => `
          <tr>
            <td>${{row.strategy}}</td>
            <td>${{fmtNum(row.mean_lineup_points)}}</td>
          </tr>
        `).join('') || '<tr><td colspan="2" class="muted">No backtest data.</td></tr>';

        const status = document.getElementById('status-pills');
        status.innerHTML = [
          ['players', decisionTable.length],
          ['taken', state.takenPlayers.size],
          ['yours', state.yourPlayers.size],
          ['current pick', state.currentPickNumber],
          ['next pick', nextPickNumber(state.currentPickNumber, state.draftPosition, state.leagueSize)],
        ].map(([label, value]) => `<span class="pill">${{label}} ${{value}}</span>`).join('');
      }}

      function render() {{
        const panel = recommendedPanels();
        renderTurnMeta(panel);
        renderList('fallback-list', panel.fallbacks, 'No fallback options are available.', 'fallback');
        renderList('wait-list', panel.canWait, 'No wait candidates are available.', 'wait');
        renderBoardActions();
        renderBoardTable(panel.rows);
        renderRosterPanel(panel);
        renderSupport(panel);
      }}

      function applyAction(action, playerName) {{
        const normalized = safeLower(playerName);
        const row = decisionTable.find((entry) => safeLower(entry.player_name) === normalized);
        if (!row) return;
        if (action === 'taken') {{
          if (!state.takenPlayers.has(normalized)) {{
            state.takenPlayers.add(normalized);
            state.history.push({{ type: 'taken', player_name: row.player_name }});
          }}
        }} else if (action === 'mine') {{
          if (!state.yourPlayers.has(normalized)) {{
            state.yourPlayers.add(normalized);
            state.takenPlayers.add(normalized);
            state.rosterCounts[row.position] = (state.rosterCounts[row.position] || 0) + 1;
            state.history.push({{ type: 'mine', player_name: row.player_name, position: row.position }});
          }}
        }}
        render();
      }}

      function undoLast() {{
        const last = state.history.pop();
        if (!last) return;
        const normalized = safeLower(last.player_name);
        if (last.type === 'taken') {{
          state.takenPlayers.delete(normalized);
        }} else if (last.type === 'mine') {{
          state.yourPlayers.delete(normalized);
          state.takenPlayers.delete(normalized);
          if (last.position && state.rosterCounts[last.position]) {{
            state.rosterCounts[last.position] = Math.max(0, (state.rosterCounts[last.position] || 0) - 1);
          }}
        }}
        render();
      }}

      function advancePick() {{
        state.currentPickNumber = Math.max(1, state.currentPickNumber + 1);
        render();
      }}

      document.getElementById('player-search').addEventListener('input', (event) => {{
        state.search = event.target.value;
        render();
      }});
      document.getElementById('undo-button').addEventListener('click', undoLast);
      document.getElementById('advance-button').addEventListener('click', advancePick);

      render();
    }})();
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')
    return output_path


def build_draft_decision_artifacts(
    player_frame: pd.DataFrame,
    league_settings: LeagueSettings | None = None,
    context: DraftContext | None = None,
    season_history: pd.DataFrame | None = None,
) -> DraftDecisionArtifacts:
    """Build the full set of draft decision artifacts in memory."""
    settings = league_settings or LeagueSettings()
    context = context or DraftContext(current_pick_number=settings.draft_position)
    decision_table = build_decision_table(player_frame, settings, context)
    recommendations = build_recommendations(decision_table, settings, context)
    tier_cliffs = build_tier_cliffs(decision_table)
    roster_scenarios = build_roster_scenarios(decision_table, settings)
    source_freshness = _compute_freshness(normalize_player_frame(player_frame))
    backtest = (
        run_draft_backtest(season_history, settings)
        if season_history is not None and not season_history.empty
        else {}
    )
    dashboard_payload = build_dashboard_payload(
        decision_table,
        recommendations,
        tier_cliffs,
        roster_scenarios,
        source_freshness,
        settings,
        backtest=backtest,
        context=context,
    )
    metadata = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'context': {
            'current_pick_number': context.current_pick_number,
            'draft_position': settings.draft_position,
            'league_size': settings.league_size,
        },
        'decision_table_columns': list(decision_table.columns),
    }
    return DraftDecisionArtifacts(
        league_settings=settings,
        decision_table=decision_table,
        recommendations=recommendations,
        roster_scenarios=roster_scenarios,
        tier_cliffs=tier_cliffs,
        source_freshness=source_freshness,
        backtest=backtest,
        dashboard_payload=dashboard_payload,
        metadata=metadata,
    )


def save_draft_decision_artifacts(
    artifacts: DraftDecisionArtifacts,
    output_dir: Path | str,
    year: int | None = None,
    filename_prefix: str = '',
    dashboard_dir: Path | str | None = None,
    diagnostics_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Write workbook, dashboard payload, and HTML dashboard to disk."""
    year = year or datetime.now().year
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = Path(dashboard_dir) if dashboard_dir is not None else output_dir
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = (
        Path(diagnostics_dir) if diagnostics_dir is not None else output_dir
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = output_dir / f'draft_board_{filename_prefix}{year}.xlsx'
    payload_path = dashboard_dir / f'dashboard_payload_{filename_prefix}{year}.json'
    html_path = dashboard_dir / f'draft_board_{filename_prefix}{year}.html'
    compat_path = dashboard_dir / f'draft_board_{filename_prefix}{year}.json'
    backtest_years = artifacts.backtest.get('holdout_years', [])
    backtest_suffix = (
        f'{min(backtest_years)}-{max(backtest_years)}' if backtest_years else str(year)
    )
    backtest_path = (
        output_dir / f'draft_decision_backtest_{filename_prefix}{backtest_suffix}.json'
    )
    model_output_dir = output_dir / 'model_outputs'
    model_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = model_output_dir / f'current_year_model_comparison_{year}.json'

    export_workbook(
        artifacts.decision_table,
        artifacts.recommendations,
        artifacts.tier_cliffs,
        artifacts.roster_scenarios,
        artifacts.source_freshness,
        workbook_path,
        artifacts.league_settings,
        backtest=artifacts.backtest,
    )
    payload_path.write_text(
        json.dumps(artifacts.dashboard_payload, default=str, indent=2), encoding='utf-8'
    )
    export_dashboard_html(
        artifacts.decision_table,
        artifacts.recommendations,
        html_path,
        artifacts.league_settings,
        backtest=artifacts.backtest,
        source_freshness=artifacts.source_freshness,
        dashboard_payload=artifacts.dashboard_payload,
    )
    compat_path.write_text(
        json.dumps(artifacts.dashboard_payload, default=str, indent=2),
        encoding='utf-8',
    )
    if artifacts.backtest:
        backtest_path.write_text(
            json.dumps(artifacts.backtest, default=str, indent=2), encoding='utf-8'
        )

    comparison_payload = {
        'model_type': 'current_year_model_comparison',
        'year': year,
        'top_players': artifacts.decision_table[
            [
                column
                for column in ['player_name', 'position', 'draft_score', 'draft_tier']
                if column in artifacts.decision_table.columns
            ]
        ]
        .head(25)
        .to_dict(orient='records'),
        'supporting_math': artifacts.dashboard_payload.get('supporting_math', {}),
        'league_settings': artifacts.league_settings.to_dict(),
    }
    comparison_path.write_text(
        json.dumps(comparison_payload, default=str, indent=2), encoding='utf-8'
    )

    return {
        'workbook_path': workbook_path,
        'payload_path': payload_path,
        'html_path': html_path,
        'compat_path': compat_path,
        'backtest_path': backtest_path,
        'comparison_path': comparison_path,
    }
