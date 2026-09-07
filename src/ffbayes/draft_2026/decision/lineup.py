"""Exact starter assignment for configurable roster slot groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.rules import RosterRules, rules_for_profile


@dataclass(frozen=True)
class LineupAssignment:
    """Best legal lineup under one team's current player set."""

    assignments: tuple[tuple[str, int], ...]
    total_points: float
    filled_slots: int
    unfilled_slots: tuple[str, ...]

    @property
    def used_player_ids(self) -> tuple[int, ...]:
        return tuple(player_id for _, player_id in self.assignments)


def optimize_lineup(
    players: pd.DataFrame,
    profile: LeagueProfile | RosterRules,
    *,
    player_id_column: str = 'espn_id',
    points_column: str = 'projected_points',
) -> LineupAssignment:
    """Find the exact maximum-point legal assignment for owned players.

    The recursive matcher tracks selected counts by position and is exact
    for the supported position-based eligibility rules.  Missing
    starters are allowed while a draft is in progress and are reported in the
    result; all score columns are treated as non-negative.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    required = {player_id_column, 'position', points_column}
    missing = sorted(required.difference(players.columns))
    if missing:
        raise ValueError(f'lineup player frame is missing columns: {missing}')

    rows: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for _, row in players.iterrows():
        raw_id = row[player_id_column]
        if isinstance(raw_id, bool):
            raise ValueError('lineup player IDs must be integers')
        try:
            player_id = int(raw_id)
            position = str(row['position']).upper()
            points = float(row[points_column])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('lineup player rows contain invalid values') from exc
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, (int, float))
            or not math.isfinite(float(raw_id))
            or float(raw_id) != float(player_id)
            or player_id < 1
            or player_id in seen
        ):
            raise ValueError('lineup player IDs must be positive and unique')
        if not position or not math.isfinite(points):
            raise ValueError('lineup player rows contain invalid values')
        seen.add(player_id)
        rows.append((player_id, position, max(points, 0.0)))

    expanded = rules.expanded_slots()
    # Players of a position have identical eligibility. For any selected count,
    # an optimum uses the highest-valued players of that position. Tracking
    # counts rather than individual-player subsets preserves exactness while
    # avoiding exponential duplication among interchangeable bench players.
    positions = tuple(sorted({position for _, position, _ in rows}))
    groups = tuple(
        tuple(
            sorted(
                (i for i, (_, pos, _) in enumerate(rows) if pos == position),
                key=lambda i: (-rows[i][2], rows[i][0]),
            )
        )
        for position in positions
    )

    @lru_cache(maxsize=None)
    def best(index: int, counts: tuple[int, ...]) -> tuple[int, float, tuple[int, ...]]:
        if index >= len(expanded):
            return 0, 0.0, ()
        filled, points, tail = best(index + 1, counts)
        result = (filled, points, (-1,) + tail)
        for category, position in enumerate(positions):
            if position not in expanded[index][1] or counts[category] >= len(
                groups[category]
            ):
                continue
            row_index = groups[category][counts[category]]
            updated = list(counts)
            updated[category] += 1
            filled, points, tail = best(index + 1, tuple(updated))
            candidate = (filled + 1, points + rows[row_index][2], (row_index,) + tail)
            if candidate[:2] > result[:2]:
                result = candidate
        return result

    filled, total_points, chosen_indices = best(0, (0,) * len(positions))
    assignments = tuple(
        (expanded[index][0], rows[row_index][0])
        for index, row_index in enumerate(chosen_indices)
        if row_index >= 0
    )
    unfilled_slots = tuple(
        expanded[index][0]
        for index, row_index in enumerate(chosen_indices)
        if row_index < 0
    )
    return LineupAssignment(
        assignments=assignments,
        total_points=float(total_points),
        filled_slots=int(filled),
        unfilled_slots=tuple(unfilled_slots),
    )
