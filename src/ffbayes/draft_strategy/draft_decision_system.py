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
    roster_counts: dict[str, int] = field(default_factory=dict)

    def drafted_set(self) -> set[str]:
        return {name.strip().lower() for name in self.drafted_players if name}


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
    for column in df.columns:
        normalized = column.strip().lower().replace(' ', '_')
        if normalized in {'player', 'player_name', 'name', 'playername'}:
            rename_map[column] = 'player_name'
        elif normalized in {'pos', 'position', 'slot'}:
            rename_map[column] = 'position'
        elif normalized in {
            'fpts',
            'fantpt',
            'fantasy_points',
            'projected_points',
            'projected_fpts',
            'proj_points',
            'projection',
        }:
            rename_map[column] = 'proj_points_mean'
        elif normalized in {'adp', 'avg', 'average_draft_position', 'market_rank'}:
            rename_map[column] = 'adp'
        elif normalized in {'mean_projection', 'mean_proj', 'consensus_projection'}:
            rename_map[column] = 'proj_points_mean'
        elif normalized in {'std_projection', 'projection_std', 'projection_spread'}:
            rename_map[column] = 'std_projection'
        elif normalized in {'uncertainty_score', 'risk_score', 'volatility_score'}:
            rename_map[column] = 'uncertainty_score'
        elif normalized in {'vor', 'value_over_replacement'}:
            rename_map[column] = 'vor_value'
        elif normalized in {'valuerank', 'vor_rank', 'market_rank_numeric'}:
            rename_map[column] = 'market_rank'

    df = df.rename(columns=rename_map)

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
        'pick_mode',
        'rationale',
    ]
    combined = pd.concat([now[cols], wait[cols]], ignore_index=True)
    combined = combined.sort_values(
        ['pick_mode', 'draft_score'], ascending=[True, False]
    ).reset_index(drop=True)
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
        elif strategy_name == 'vor':
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
        by_strategy = {}
        for strategy in ['market', 'vor', 'consensus', 'draft_score']:
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
                ('vor', 'proj_points_mean'),
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
    for strategy in ['market', 'vor', 'consensus', 'draft_score']:
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

    top_targets = recommendations.head(10).to_dict(orient='records')
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
        'top_targets': top_targets,
        'decision_table': decision_table.to_dict(orient='records'),
        'position_summary': position_summary,
        'tier_cliffs': tier_cliffs.to_dict(orient='records'),
        'roster_scenarios': roster_scenarios.to_dict(orient='records'),
        'source_freshness': source_freshness.to_dict(orient='records'),
        'backtest': backtest,
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

    my_picks = recommendations.copy()
    _write_dataframe_sheet(wb, 'My Picks', my_picks)

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
) -> Path:
    """Write a local interactive HTML dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backtest = backtest or {}
    source_freshness = (
        source_freshness if source_freshness is not None else pd.DataFrame()
    )

    try:
        figs = _build_charts(decision_table, recommendations, league_settings)
    except Exception:
        figs = []

    if not figs:
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FFBayes Draft Board</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }}
    pre {{
      white-space: pre-wrap;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 16px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <h1>FFBayes Draft Board</h1>
  <p>Plotly is unavailable in this environment, so this is the lightweight fallback view.</p>
  <h2>Top Targets</h2>
  <pre>{json.dumps(recommendations.head(12).to_dict(orient='records'), default=str, indent=2)}</pre>
  <h2>Dashboard Payload</h2>
  <pre>{json.dumps({'league_settings': league_settings.to_dict(), 'backtest': backtest}, default=str, indent=2)}</pre>
</body>
</html>
"""
        output_path.write_text(html, encoding='utf-8')
        return output_path

    chart_divs = []
    for fig in figs:
        chart_divs.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    top_rows = recommendations.head(12).to_dict(orient='records')
    freshness_rows = source_freshness.to_dict(orient='records')
    backtest_summary = backtest.get('overall', {}).get('by_strategy', [])

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FFBayes Draft Board</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.25fr 0.85fr;
      gap: 20px;
      align-items: start;
    }}
    .card {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.24);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      color: #e2e8f0;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #93c5fd; }}
    .muted {{ color: #94a3b8; }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(59, 130, 246, 0.18);
      color: #bfdbfe;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .section {{ margin-top: 18px; }}
    .small {{ font-size: 12px; }}
    .charts > div {{ margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>FFBayes Draft Board</h1>
  <p class="muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | League {league_settings.league_size}-team, pick {league_settings.draft_position}</p>
  <div class="grid">
    <div class="card charts">
      {''.join(chart_divs)}
    </div>
    <div class="card">
      <h2>Top Targets</h2>
      <div class="small muted">Best current options and the reasons they show up on the board.</div>
      <table>
        <thead>
          <tr>
            <th>Player</th><th>Pos</th><th>Score</th><th>Avail</th><th>Why</th>
          </tr>
        </thead>
        <tbody>
          {''.join(
              f"<tr><td>{row.get('player_name', '')}</td><td>{row.get('position', '')}</td><td>{row.get('draft_score', 0):.2f}</td><td>{row.get('availability_to_next_pick', 0):.0%}</td><td>{row.get('why_flags', '')}</td></tr>"
              for row in top_rows
          )}
        </tbody>
      </table>
      <div class="section">
        <h3>Backtest</h3>
        <div class="small muted">Season-level snake-draft proxy results.</div>
        <table>
          <thead><tr><th>Strategy</th><th>Mean lineup points</th></tr></thead>
          <tbody>
            {''.join(
                f"<tr><td>{row.get('strategy', '')}</td><td>{row.get('mean_lineup_points', 0):.2f}</td></tr>"
                for row in backtest_summary
            )}
          </tbody>
        </table>
      </div>
      <div class="section">
        <h3>Source Freshness</h3>
        <table>
          <thead><tr><th>Source</th><th>Age (days)</th><th>Rows</th></tr></thead>
          <tbody>
            {''.join(
                f"<tr><td>{row.get('source_name', '')}</td><td>{row.get('freshness_days', '')}</td><td>{row.get('row_count', '')}</td></tr>"
                for row in freshness_rows
            )}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    window.FFBAYES_DASHBOARD = {json.dumps({'league_settings': league_settings.to_dict(), 'top_targets': top_rows, 'backtest': backtest}, default=str)};
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
) -> dict[str, Path]:
    """Write workbook, dashboard payload, and HTML dashboard to disk."""
    year = year or datetime.now().year
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = Path(dashboard_dir) if dashboard_dir is not None else output_dir
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = output_dir / f'draft_board_{filename_prefix}{year}.xlsx'
    payload_path = dashboard_dir / f'dashboard_payload_{filename_prefix}{year}.json'
    html_path = dashboard_dir / f'draft_board_{filename_prefix}{year}.html'
    backtest_years = artifacts.backtest.get('holdout_years', [])
    backtest_suffix = (
        f'{min(backtest_years)}-{max(backtest_years)}' if backtest_years else str(year)
    )
    backtest_path = (
        output_dir / f'draft_decision_backtest_{filename_prefix}{backtest_suffix}.json'
    )

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
    )
    if artifacts.backtest:
        backtest_path.write_text(
            json.dumps(artifacts.backtest, default=str, indent=2), encoding='utf-8'
        )

    return {
        'workbook_path': workbook_path,
        'payload_path': payload_path,
        'html_path': html_path,
        'backtest_path': backtest_path,
    }
