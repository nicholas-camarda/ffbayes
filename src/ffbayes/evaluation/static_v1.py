"""Frozen static VOR/ADP v1 support functions.

Copied from the first generic-rule implementation. Matches the original rank
formula for legacy profiles and extends replacement demand to configured
slot groups. Keep independent of mutable live-engine helpers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.sources import SemanticInputError

VALUED_POSITIONS = ('QB', 'RB', 'WR', 'TE', 'DST', 'K')


def _score_stats(
    stats: object, scoring_items: dict[str, float], position_overrides: dict[str, float]
) -> float:
    if not isinstance(stats, dict):
        return float('nan')
    effective_items = {**scoring_items, **position_overrides}
    return float(
        sum(
            float(stats.get(stat_id, 0.0)) * points
            for stat_id, points in effective_items.items()
        )
    )


def score_projections(players: pd.DataFrame, profile: LeagueProfile) -> pd.DataFrame:
    """Apply the complete configured scoring surface to raw projected stats."""
    if 'projection_stats' not in players:
        raise SemanticInputError('Projection stats are required for league scoring')
    frame = players.copy()
    frame['projected_points'] = frame.apply(
        lambda row: _score_stats(
            row['projection_stats'],
            profile.scoring_items,
            profile.scoring_overrides.get(str(row['position']), {}),
        ),
        axis=1,
    )
    if profile.bonuses:
        if 'weekly_projection_stats' not in frame:
            raise SemanticInputError(
                'Weekly projection stats are required for configured bonuses'
            )

        def bonus_points(weeks: object) -> float:
            if not isinstance(weeks, list) or not weeks:
                raise SemanticInputError(
                    'Weekly projection coverage is missing for configured bonuses'
                )
            total = 0.0
            for week in weeks:
                for bonus in profile.bonuses:
                    if float(week.get(bonus['stat_id'], 0.0)) >= bonus['threshold']:
                        total += bonus['points']
            return total

        frame['projected_points'] += frame['weekly_projection_stats'].map(bonus_points)
    points = pd.to_numeric(frame['projected_points'], errors='coerce')
    if points.isna().any() or (~np.isfinite(points)).any() or (points < 0).any():
        raise SemanticInputError('League scoring produced invalid projected points')
    frame['scoring_profile'] = profile.profile_id
    return frame


def _flex_allocation(
    scored: pd.DataFrame, base_demand: dict[str, int], profile: LeagueProfile
) -> dict[str, int]:
    # Preserve the historic return shape for FLEX while honoring a generic
    # slot eligibility map.  Overlapping SUPERFLEX groups are handled by the
    # roster assignment used for live marginal value; this replacement helper
    # still provides an interpretable league-wide scarcity baseline.
    eligible = profile.eligible_positions('FLEX') or profile.flex_eligible
    allocation = {position: 0 for position in eligible}
    flex_total = profile.roster_slots.get('FLEX', 0) * profile.team_count
    position_values = {
        position: scored.loc[scored['position'].eq(position), 'projected_points']
        .sort_values(ascending=False)
        .reset_index(drop=True)
        for position in profile.flex_eligible
    }
    for _ in range(flex_total):
        candidates: list[tuple[float, str]] = []
        for position, values in position_values.items():
            index = base_demand.get(position, 0) + allocation[position]
            if index < len(values):
                candidates.append((float(values.iloc[index]), position))
        if not candidates:
            raise SemanticInputError(
                'Projection depth cannot fill configured FLEX demand'
            )
        _, selected_position = max(candidates)
        allocation[selected_position] += 1
    return allocation


def _market_bench_allocation(
    scored: pd.DataFrame, occupied: dict[str, int], total_bench_slots: int
) -> dict[str, int]:
    """Allocate skill-position bench demand in current ESPN ADP order."""
    bench_positions = ('QB', 'RB', 'WR', 'TE')
    allocation = {position: 0 for position in bench_positions}
    market = {
        position: scored.loc[scored['position'].eq(position), 'adp']
        .sort_values()
        .reset_index(drop=True)
        for position in bench_positions
    }
    for _ in range(total_bench_slots):
        candidates: list[tuple[float, str]] = []
        for position, values in market.items():
            index = occupied.get(position, 0) + allocation[position]
            if index < len(values):
                adp = float(values.iloc[index])
                if math.isfinite(adp):
                    candidates.append((adp, position))
        if not candidates:
            raise SemanticInputError('Market depth cannot fill configured bench demand')
        _, selected_position = min(candidates)
        allocation[selected_position] += 1
    return allocation


def calculate_replacement_levels(
    scored: pd.DataFrame, profile: LeagueProfile
) -> dict[str, Any]:
    """Calculate league-specific replacement demand and valid position baselines."""
    if scored.empty:
        raise SemanticInputError(
            'Cannot calculate replacement levels from an empty pool'
        )
    configured_positions = tuple(
        sorted(
            {
                position
                for _, _, eligible in profile.slot_groups()
                for position in eligible
            }
            | set(str(value).upper() for value in scored['position'].dropna().unique())
        )
    )
    # Fixed-position starter demand is separated from overlapping groups; FLEX
    # allocation is added below.  Zero-demand positions are retained in the
    # diagnostics but never require a replacement level.
    base_demand = {
        position: 0 for position in configured_positions if position != 'FLEX'
    }
    for slot_name, count, eligible in profile.slot_groups():
        if slot_name == 'FLEX' or len(eligible) != 1:
            continue
        position = eligible[0]
        base_demand[position] = (
            base_demand.get(position, 0) + count * profile.team_count
        )
    flex = _flex_allocation(scored, base_demand, profile)
    # If an explicit SUPERFLEX exists, allocate it with the same deterministic
    # market-free point order used by FLEX.  ``flex_allocation`` remains the
    # legacy FLEX-only field; aggregate starter demand includes every group.
    for slot_name, count, eligible in profile.slot_groups():
        if slot_name == 'FLEX' or len(eligible) <= 1:
            continue
        allocation = {position: 0 for position in eligible}
        values = {
            position: scored.loc[scored['position'].eq(position), 'projected_points']
            .sort_values(ascending=False)
            .reset_index(drop=True)
            for position in eligible
        }
        for _ in range(count * profile.team_count):
            candidates = []
            for position, series in values.items():
                index = (
                    base_demand.get(position, 0)
                    + flex.get(position, 0)
                    + allocation[position]
                )
                if index < len(series):
                    candidates.append((float(series.iloc[index]), position))
            if not candidates:
                raise SemanticInputError(
                    f'Projection depth cannot fill configured {slot_name} demand'
                )
            _, selected = max(candidates)
            allocation[selected] += 1
        for position, amount in allocation.items():
            base_demand[position] = base_demand.get(position, 0) + amount
    occupied = {
        position: base_demand.get(position, 0) + flex.get(position, 0)
        for position in configured_positions
    }
    bench = _market_bench_allocation(
        scored, occupied, profile.bench_slots * profile.team_count
    )
    demand = {
        position: occupied.get(position, 0) + bench.get(position, 0)
        for position in configured_positions
    }

    levels: dict[str, float] = {}
    for position, required_count in demand.items():
        values = (
            scored.loc[scored['position'].eq(position), 'projected_points']
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        if required_count <= 0:
            continue
        if len(values) <= required_count:
            raise SemanticInputError(
                f'Invalid replacement demand for {position}: need depth beyond '
                f'{required_count}, found {len(values)}'
            )
        if values.nunique(dropna=True) < 2 or float(values.abs().max()) == 0.0:
            raise SemanticInputError(
                f'Projection scoring has no usable signal for {position}'
            )
        level = float(values.iloc[required_count - 1])
        if not math.isfinite(level):
            raise SemanticInputError(f'Replacement level for {position} is not finite')
        levels[position] = level

    return {
        'levels': levels,
        'demand': demand,
        'starter_demand': occupied,
        'flex_allocation': flex,
        'bench_allocation': bench,
    }


def _position_scarcity(frame: pd.DataFrame, replacement: dict[str, Any]) -> pd.Series:
    scarcity = pd.Series(0.0, index=frame.index)
    for position, demand in replacement['demand'].items():
        if int(demand) <= 0:
            continue
        indices = frame.index[frame['position'].eq(position)]
        values = frame.loc[indices, 'projected_points'].sort_values(ascending=False)
        if values.empty:
            continue

        def local_drop(index: int) -> float:
            start = max(0, index - 1)
            end = min(len(values), index + 5)
            tail = values.iloc[start:end]
            return (
                max(0.0, float(values.iloc[start] - tail.iloc[-1]))
                if len(tail) > 1
                else 0.0
            )

        starter_demand = int(replacement['starter_demand'][position])
        scarcity.loc[indices] = local_drop(demand) + local_drop(starter_demand)
    return scarcity
