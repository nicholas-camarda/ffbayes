"""Marginal replacement-adjusted lineup utility; no fitted injury probabilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ffbayes.draft_2026.decision.lineup import optimize_lineup
from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.rules import RosterRules, roster_is_feasible, rules_for_profile

ROSTER_VALUE_VERSION = 'roster-marginal-v1'
DEFAULT_BENCH_INSURANCE_WEIGHT = 0.05


def calculate_marginal_roster_value(
    players: pd.DataFrame,
    profile: LeagueProfile | RosterRules,
    *,
    your_ids: Iterable[int] = (),
    taken_ids: Iterable[int] = (),
    bench_insurance_weight: float = DEFAULT_BENCH_INSURANCE_WEIGHT,
    replacement_levels: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Evaluate one coherent utility before and after each possible selection.

    U(R) = best legal sum of positive VOR + w * remaining positive VOR.
    The bench coefficient is an explicit uncalibrated insurance proxy. The
    same full-pool replacement anchors are used before/after every candidate.
    This is a utility proxy, not a forecast of actual weekly lineup points.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    if not 0 <= bench_insurance_weight <= 1 or not np.isfinite(bench_insurance_weight):
        raise ValueError(
            'bench_insurance_weight must be finite and between zero and one'
        )
    # Validate IDs and point values even for an empty current roster.
    required = {'espn_id', 'position', 'projected_points'}
    if not required.issubset(players):
        raise ValueError(f'roster value frame requires {sorted(required)}')
    ids = pd.to_numeric(players['espn_id'], errors='coerce')
    points = pd.to_numeric(players['projected_points'], errors='coerce')
    if (
        players['espn_id'].map(lambda x: isinstance(x, (bool, np.bool_))).any()
        or not np.isfinite(ids).all()
        or (ids == 0).any()
        or (ids != ids.round()).any()
        or ids.duplicated().any()
    ):
        raise ValueError('player IDs must be unique nonzero integers')
    if not np.isfinite(points).all() or (points < 0).any():
        raise ValueError('projected points must be finite and non-negative')

    def id_set(values):
        values = list(values)
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
            for x in values
        ):
            raise ValueError('player IDs must be integers')
        return set(values)

    mine, taken = id_set(your_ids), id_set(taken_ids)
    if mine & taken or (mine | taken) - set(ids):
        raise ValueError('overlapping or unknown player IDs')
    if replacement_levels is None:
        if not isinstance(profile, LeagueProfile):
            raise ValueError('explicit replacement_levels required for RosterRules')
        from ffbayes.draft_2026.engine import calculate_replacement_levels

        replacement_levels = calculate_replacement_levels(players, profile)['levels']
    if any(not np.isfinite(v) or v < 0 for v in replacement_levels.values()):
        raise ValueError('replacement levels must be finite and non-negative')
    frame = players.copy()
    frame['replacement_anchor'] = frame['position'].map(replacement_levels).fillna(0.0)
    frame['utility_points'] = (points - frame['replacement_anchor']).clip(lower=0)
    frame.loc[~frame['position'].isin(rules.positions), 'utility_points'] = 0.0
    owned = frame[frame.espn_id.isin(mine)]
    remaining = frame[~frame.espn_id.isin(mine | taken)]
    base = optimize_lineup(owned, rules, points_column='utility_points').total_points
    base_bench = float(owned.utility_points.sum()) - base
    rows = []
    for _, row in frame.iterrows():
        player_id = int(row.espn_id)
        gain = insurance = 0.0
        feasible = False
        if player_id not in mine | taken and len(mine) < rules.roster_size:
            added = pd.concat([owned, row.to_frame().T], ignore_index=True)
            after = optimize_lineup(
                added, rules, points_column='utility_points'
            ).total_points
            gain = max(0.0, after - base)
            insurance = bench_insurance_weight * (
                float(added.utility_points.sum()) - after - base_bench
            )
            feasible = row.position in rules.positions and roster_is_feasible(
                added.position,
                remaining.loc[remaining.espn_id != player_id, 'position'],
                rules,
            )
        rows.append(
            dict(
                espn_id=player_id,
                marginal_usable_points=max(0.0, gain + insurance),
                starter_slot_gain=gain,
                bench_insurance_value=insurance,
                replacement_anchor=float(row.replacement_anchor),
                roster_feasible=feasible,
                roster_value_version=ROSTER_VALUE_VERSION,
            )
        )
    return pd.DataFrame(rows).set_index('espn_id')
