"""League-aware scoring, replacement, VOR, scarcity, and pick decisions."""

from __future__ import annotations

import math
from typing import Any, Iterable

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
    allocation = {position: 0 for position in profile.flex_eligible}
    flex_total = profile.roster_slots['FLEX'] * profile.team_count
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
    base_demand = {
        position: profile.roster_slots.get(position, 0) * profile.team_count
        for position in VALUED_POSITIONS
    }
    flex = _flex_allocation(scored, base_demand, profile)
    occupied = {
        position: base_demand.get(position, 0) + flex.get(position, 0)
        for position in VALUED_POSITIONS
    }
    bench = _market_bench_allocation(
        scored, occupied, profile.bench_slots * profile.team_count
    )
    demand = {
        position: occupied.get(position, 0) + bench.get(position, 0)
        for position in VALUED_POSITIONS
    }

    levels: dict[str, float] = {}
    for position, required_count in demand.items():
        values = (
            scored.loc[scored['position'].eq(position), 'projected_points']
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        if required_count <= 0 or len(values) <= required_count:
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


def next_snake_pick(current_pick: int, profile: LeagueProfile) -> int | None:
    """Return the user's next pick strictly after the current overall pick.

    The clock may equal one past the final configured pick so a completed
    draft can be represented without inventing a future turn.
    """
    if profile.draft_slot is None:
        raise SemanticInputError('Draft slot is required for snake-pick timing')
    final_pick = profile.total_draft_picks()
    if isinstance(current_pick, bool) or not isinstance(current_pick, int):
        raise SemanticInputError('Current pick must be an integer')
    if not 1 <= current_pick <= final_pick + 1:
        raise SemanticInputError('Current pick is outside the configured draft')
    if current_pick == final_pick + 1:
        return None
    picks: list[int] = []
    for round_number in range(1, profile.active_roster_size() + 1):
        if round_number % 2:
            pick = (round_number - 1) * profile.team_count + profile.draft_slot
        else:
            pick = round_number * profile.team_count - profile.draft_slot + 1
        if pick > current_pick:
            picks.append(pick)
    return min(picks) if picks else None


def _availability_probability(adp: pd.Series, target_pick: int) -> pd.Series:
    means = pd.to_numeric(adp, errors='coerce')
    if means.isna().any():
        raise SemanticInputError(
            'Board contains missing ADP; availability cannot be estimated'
        )
    standard_deviation = np.maximum(6.0, means.to_numpy(dtype=float) * 0.18)
    z = (target_pick - means.to_numpy(dtype=float)) / standard_deviation
    cdf = np.array([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in z])
    return pd.Series(1.0 - cdf, index=adp.index).clip(0.0, 1.0)


def _position_scarcity(frame: pd.DataFrame, replacement: dict[str, Any]) -> pd.Series:
    scarcity = pd.Series(0.0, index=frame.index)
    for position, demand in replacement['demand'].items():
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


def build_draft_board(
    players: pd.DataFrame,
    profile: LeagueProfile,
    *,
    current_pick: int | None = None,
    taken_ids: Iterable[int] = (),
    your_ids: Iterable[int] = (),
) -> pd.DataFrame:
    """Build a league-specific board with no missing-input substitutions."""
    if pd.to_numeric(players.get('adp'), errors='coerce').isna().any():
        raise SemanticInputError(
            'Board contains missing ADP; no neutral fallback is allowed'
        )
    scored = score_projections(players, profile)
    replacement = calculate_replacement_levels(scored, profile)
    frame = scored.copy()
    frame['replacement_level'] = frame['position'].map(replacement['levels'])
    if frame['replacement_level'].isna().any():
        raise SemanticInputError('A position has no valid replacement level')
    frame['vor'] = frame['projected_points'] - frame['replacement_level']
    frame['scarcity'] = _position_scarcity(frame, replacement)

    if current_pick is None:
        # The draft clock is an overall-pick clock, not the user's roster slot.
        # A fresh board always starts before the first selection; the service
        # passes an explicit clock for later synchronization.
        active_pick = 1
    else:
        active_pick = current_pick
    if active_pick is None:
        next_pick = None
        frame['availability_next_pick'] = np.nan
    else:
        if not 1 <= active_pick <= profile.total_draft_picks() + 1:
            raise SemanticInputError('Current pick is outside the configured draft')
        if profile.draft_slot is None:
            next_pick = None
            frame['availability_next_pick'] = np.nan
        else:
            next_pick = next_snake_pick(active_pick, profile)
            if next_pick is None:
                frame['availability_next_pick'] = np.nan
            else:
                frame['availability_next_pick'] = _availability_probability(
                    frame['adp'], next_pick
                )

    vor_scale = max(float(frame['vor'].std(ddof=0)), 1.0)
    scarcity_scale = max(float(frame['scarcity'].std(ddof=0)), 1.0)
    frame['value_signal'] = (
        frame['vor'] / vor_scale + 0.25 * frame['scarcity'] / scarcity_scale
    )
    frame['model_rank'] = frame['value_signal'].rank(ascending=False, method='average')
    frame['market_rank'] = frame['adp'].rank(ascending=True, method='average')
    frame['board_priority'] = 0.6 * frame['model_rank'] + 0.4 * frame['market_rank']
    frame['decision_score'] = -frame['board_priority']
    frame = frame.sort_values(['board_priority', 'vor', 'adp']).reset_index(drop=True)
    frame['board_rank'] = np.arange(1, len(frame) + 1)
    frame['recommendation'] = np.where(
        frame['availability_next_pick'].isna(),
        'slot_required',
        np.where(frame['availability_next_pick'] < 0.35, 'draft_now', 'can_wait'),
    )

    if 'espn_id' in frame:
        espn_ids = pd.to_numeric(frame['espn_id'], errors='coerce')
    else:
        espn_ids = pd.Series(np.nan, index=frame.index)
    taken = {int(value) for value in taken_ids}
    yours = {int(value) for value in your_ids}
    if taken.intersection(yours):
        raise SemanticInputError('A player cannot be both taken and yours')
    if taken or yours:
        if espn_ids.isna().any():
            raise SemanticInputError('Stable espn_id values are required for draft state')
        known_ids = {int(value) for value in espn_ids}
        unknown = (taken | yours).difference(known_ids)
        if unknown:
            raise SemanticInputError(f'unknown player espn_id values: {sorted(unknown)}')
    frame['espn_id'] = espn_ids.astype('Int64')
    frame['roster_status'] = 'available'
    if taken:
        frame.loc[espn_ids.isin(taken), 'roster_status'] = 'taken'
    if yours:
        frame.loc[espn_ids.isin(yours), 'roster_status'] = 'mine'
    frame['is_available'] = frame['roster_status'].eq('available')
    frame.loc[frame['roster_status'].eq('taken'), 'recommendation'] = 'taken'
    frame.loc[frame['roster_status'].eq('mine'), 'recommendation'] = 'mine'
    frame.attrs.update(
        {
            'profile_id': profile.profile_id,
            'current_pick': active_pick,
            'next_pick': next_pick,
            'replacement': replacement,
        }
    )
    return frame
