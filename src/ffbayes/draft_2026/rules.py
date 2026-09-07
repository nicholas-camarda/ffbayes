"""Generic roster-slot rules shared by valuation and draft evaluation.

The live dashboard historically represented one FLEX slot as a special case.
This module treats every roster slot as an eligibility group.  A SUPERFLEX is
therefore just another group whose eligible positions include QB, and formats
without K/DST are represented by zero-demand or omitted optional groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import pandas as pd

from ffbayes.draft_2026.league import LeagueProfile


class RosterRulesError(ValueError):
    """Raised when a roster cannot satisfy its configured slot contract."""


@dataclass(frozen=True)
class SlotRule:
    """One slot group and the player positions it accepts."""

    name: str
    count: int
    eligible_positions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise RosterRulesError('slot name cannot be empty')
        if self.count < 0:
            raise RosterRulesError(f'{self.name} count cannot be negative')
        if not self.eligible_positions:
            raise RosterRulesError(f'{self.name} must accept at least one position')


@dataclass(frozen=True)
class RosterRules:
    """Normalized roster contract for one league and one team's roster."""

    team_count: int
    slots: tuple[SlotRule, ...]
    bench_slots: int
    ir_slots: int

    @classmethod
    def from_profile(cls, profile: LeagueProfile) -> 'RosterRules':
        slots = tuple(
            SlotRule(name, count, tuple(positions))
            for name, count, positions in profile.slot_groups()
        )
        return cls(
            team_count=profile.team_count,
            slots=slots,
            bench_slots=profile.bench_slots,
            ir_slots=profile.ir_slots,
        )

    @property
    def starter_count(self) -> int:
        return sum(slot.count for slot in self.slots)

    @property
    def roster_size(self) -> int:
        return self.starter_count + self.bench_slots

    @property
    def positions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    position
                    for slot in self.slots
                    for position in slot.eligible_positions
                }
            )
        )

    def expanded_slots(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Expand counts into one slot per assignment for exact matching."""
        result: list[tuple[str, tuple[str, ...]]] = []
        for slot in self.slots:
            result.extend(
                (slot.name, slot.eligible_positions) for _ in range(slot.count)
            )
        # Constrained slots first is only a speed optimization; the exact
        # assignment result does not depend on this order.
        result.sort(key=lambda item: (len(item[1]), item[0]))
        return tuple(result)

    def position_starter_demand(self) -> dict[str, int]:
        """Return fixed-position demand, leaving overlapping slots unassigned."""
        demand: dict[str, int] = {}
        for slot in self.slots:
            if len(slot.eligible_positions) == 1:
                position = slot.eligible_positions[0]
                demand[position] = demand.get(position, 0) + slot.count
        return demand


def rules_for_profile(profile: LeagueProfile) -> RosterRules:
    """Return normalized rules for a profile."""
    return RosterRules.from_profile(profile)


def _position_counts(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        position = str(value).upper()
        counts[position] = counts.get(position, 0) + 1
    return tuple(sorted(counts.items()))


def maximum_slot_match(
    positions: Iterable[str], profile: LeagueProfile | RosterRules
) -> int:
    """Return the exact maximum number of starter slots that can be filled.

    This count-level matcher is used for feasibility checks.  It is an exact
    maximum matching over position categories, including overlapping FLEX and
    SUPERFLEX groups; it does not rely on greedy slot assignment.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    expanded = rules.expanded_slots()
    raw_counts = dict(_position_counts(positions))
    all_positions = tuple(sorted(set(raw_counts) | set(rules.positions)))
    initial_counts = tuple(raw_counts.get(position, 0) for position in all_positions)

    @lru_cache(maxsize=None)
    def best(index: int, counts: tuple[int, ...]) -> int:
        if index >= len(expanded):
            return 0
        _, eligible = expanded[index]
        result = best(index + 1, counts)  # leave this slot empty
        for position in eligible:
            if position not in all_positions:
                continue
            position_index = all_positions.index(position)
            if counts[position_index] <= 0:
                continue
            updated = list(counts)
            updated[position_index] -= 1
            result = max(result, 1 + best(index + 1, tuple(updated)))
        return result

    return best(0, initial_counts)


def can_fill_starters(
    positions: Iterable[str], profile: LeagueProfile | RosterRules
) -> bool:
    """Return whether all configured starter groups can be filled."""
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    return maximum_slot_match(positions, rules) == rules.starter_count


def minimum_additional_players_for_starters(
    roster_positions: Iterable[str],
    available_positions: Iterable[str],
    profile: LeagueProfile | RosterRules,
) -> int | None:
    """Return the fewest future picks needed to fill every starter slot.

    The dynamic program tracks roster and available position counts separately,
    so an apparently adequate full player pool cannot hide an impossible final
    set of picks.  ``None`` means no legal assignment exists even with every
    available player.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    expanded = rules.expanded_slots()
    roster_counts = dict(_position_counts(roster_positions))
    available_counts = dict(_position_counts(available_positions))
    positions = tuple(
        sorted(set(roster_counts) | set(available_counts) | set(rules.positions))
    )
    roster_initial = tuple(roster_counts.get(position, 0) for position in positions)
    available_initial = tuple(
        available_counts.get(position, 0) for position in positions
    )

    @lru_cache(maxsize=None)
    def minimum(
        index: int, roster_state: tuple[int, ...], available_state: tuple[int, ...]
    ) -> int | None:
        if index >= len(expanded):
            return 0
        _, eligible = expanded[index]
        best_cost: int | None = None
        for position in eligible:
            if position not in positions:
                continue
            position_index = positions.index(position)
            if roster_state[position_index] > 0:
                next_roster = list(roster_state)
                next_roster[position_index] -= 1
                result = minimum(index + 1, tuple(next_roster), available_state)
                if result is not None:
                    best_cost = result if best_cost is None else min(best_cost, result)
            if available_state[position_index] > 0:
                next_available = list(available_state)
                next_available[position_index] -= 1
                result = minimum(index + 1, roster_state, tuple(next_available))
                if result is not None:
                    result += 1
                    best_cost = result if best_cost is None else min(best_cost, result)
        return best_cost

    return minimum(0, roster_initial, available_initial)


def roster_is_feasible(
    roster_positions: Iterable[str],
    available_positions: Iterable[str],
    profile: LeagueProfile | RosterRules,
) -> bool:
    """Check starter coverage and total-player capacity after a candidate pick.

    ``roster_positions`` includes players already selected by the team.  The
    available pool includes legal future candidates after the proposed pick.
    The check is deliberately independent of player scores and therefore can
    be shared by every baseline policy.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    roster = [str(value).upper() for value in roster_positions]
    available = [str(value).upper() for value in available_positions]
    if len(roster) > rules.roster_size:
        return False
    if len(roster) + len(available) < rules.roster_size:
        return False
    additional = minimum_additional_players_for_starters(roster, available, rules)
    if additional is None:
        return False
    return additional <= rules.roster_size - len(roster)


def validate_complete_roster(
    roster_positions: Iterable[str], profile: LeagueProfile | RosterRules
) -> None:
    """Raise when a completed roster is illegal or incomplete."""
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    values = [str(value).upper() for value in roster_positions]
    if len(values) != rules.roster_size:
        raise RosterRulesError(
            f'roster has {len(values)} players; expected {rules.roster_size}'
        )
    if not can_fill_starters(values, rules):
        raise RosterRulesError('roster cannot fill all configured starter slots')


def position_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count positions from a player frame for diagnostics and tests."""
    if 'position' not in frame:
        raise RosterRulesError('player frame requires a position column')
    return dict(_position_counts(frame['position'].tolist()))


def league_can_complete(
    rosters: Iterable[Iterable[str]],
    available_positions: Iterable[str],
    profile: LeagueProfile | RosterRules,
) -> bool:
    """Exact league-wide completion including bench capacity and ownership.

    Match all owned players to their own team's slots first, then augment
    with available players. Augmentation never drops a previously matched
    player, so every existing selection must fit the completed league.
    """
    rules = profile if isinstance(profile, RosterRules) else rules_for_profile(profile)
    teams = [list(roster) for roster in rosters]
    if len(teams) != rules.team_count:
        return False
    slots: list[tuple[int, tuple[str, ...]]] = []
    for team in range(len(teams)):
        slots.extend((team, eligible) for _, eligible in rules.expanded_slots())
        slots.extend((team, rules.positions) for _ in range(rules.bench_slots))
    edges: list[list[int]] = []
    occupied: dict[int, int] = {}

    def add(position: str, owner: int | None) -> bool:
        token = len(edges)
        edges.append(
            [
                i
                for i, (team, eligible) in enumerate(slots)
                if position in eligible and (owner is None or owner == team)
            ]
        )

        def augment(player: int, visited: set[int]) -> bool:
            for slot in edges[player]:
                if slot in visited:
                    continue
                visited.add(slot)
                if slot not in occupied or augment(occupied[slot], visited):
                    occupied[slot] = player
                    return True
            return False

        return augment(token, set())

    for team, roster in enumerate(teams):
        for position in roster:
            if not add(position, team):
                return False
    for position in available_positions:
        if len(occupied) == len(slots):
            return True
        add(position, None)
    return len(occupied) == len(slots)
