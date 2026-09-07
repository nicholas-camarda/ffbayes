"""Explicit league-profile contract for current-season draft valuation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class LeagueProfileError(ValueError):
    """Raised when a league profile is incomplete or internally inconsistent."""


# K and DST are common positions, but they are optional in many formats.  The
# five skill/utility groups are the minimum contract needed to describe a
# conventional roster; an explicit ``slot_eligibility`` mapping can add other
# groups (for example SUPERFLEX) without changing the scoring engine.
REQUIRED_ROSTER_SLOTS = ('QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K')
CORE_ROSTER_SLOTS = ('QB', 'RB', 'WR', 'TE', 'FLEX')


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise LeagueProfileError(f'{field} must be a non-negative integer')
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LeagueProfileError(f'{field} must be a non-negative integer') from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric) or numeric < 0:
        raise LeagueProfileError(f'{field} must be a non-negative integer')
    return int(numeric)


@dataclass(frozen=True)
class LeagueProfile:
    profile_id: str
    league_name: str
    season: int
    team_count: int
    draft_format: str
    draft_slot: int | None
    scoring_label: str
    scoring_items: dict[str, float]
    scoring_overrides: dict[str, dict[str, float]]
    bonuses: tuple[dict[str, Any], ...]
    roster_slots: dict[str, int]
    bench_slots: int
    ir_slots: int
    flex_eligible: tuple[str, ...]
    waiver_type: str
    waiver_constraints: tuple[str, ...]
    settings_source: str
    settings_verified_at: str
    # Maps slot group names to eligible player positions.  ``None`` is not
    # exposed by the dataclass: from_mapping always stores a normalized map.
    # The private flag keeps legacy profile serialization byte-for-byte stable
    # when the new field was not present in the input.
    slot_eligibility: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _slot_eligibility_explicit: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'LeagueProfile':
        required = (
            'profile_id',
            'league_name',
            'season',
            'team_count',
            'draft_format',
            'scoring_label',
            'scoring_items',
            'bonuses',
            'roster_slots',
            'bench_slots',
            'ir_slots',
            'waiver_type',
            'waiver_constraints',
            'settings_source',
            'settings_verified_at',
        )
        missing = [
            key
            for key in required
            if value.get(key) is None
            or value.get(key) == ''
            or (key not in ('bonuses', 'waiver_constraints') and value.get(key) == [])
            or (key == 'scoring_items' and not value.get(key))
        ]
        if missing:
            raise LeagueProfileError(
                f'League profile has unresolved required fields: {missing}'
            )
        if value.get('flex_eligible') is None and value.get('slot_eligibility') is None:
            raise LeagueProfileError(
                'League profile requires flex_eligible or slot_eligibility'
            )

        roster = {str(k).upper(): int(v) for k, v in value['roster_slots'].items()}
        missing_slots = [slot for slot in CORE_ROSTER_SLOTS if slot not in roster]
        if missing_slots:
            raise LeagueProfileError(f'roster_slots is missing: {missing_slots}')
        # Keep the normalized shape used by the existing payload contract while
        # allowing profiles to omit K/DST entirely for formats without them.
        roster.setdefault('DST', 0)
        roster.setdefault('K', 0)
        if any(count < 0 for count in roster.values()):
            raise LeagueProfileError('roster_slots cannot contain negative values')

        raw_slot_eligibility = value.get('slot_eligibility')
        explicit_slot_eligibility = raw_slot_eligibility is not None
        if raw_slot_eligibility is None:
            slot_eligibility: dict[str, tuple[str, ...]] = {
                position: (position,) for position in roster if position != 'FLEX'
            }
            slot_eligibility['FLEX'] = tuple(
                str(position).upper() for position in (value.get('flex_eligible') or ())
            )
        elif not isinstance(raw_slot_eligibility, Mapping):
            raise LeagueProfileError('slot_eligibility must be a mapping')
        else:
            slot_eligibility = {}
            for raw_name, raw_positions in raw_slot_eligibility.items():
                name = str(raw_name).upper().strip()
                if not name:
                    raise LeagueProfileError(
                        'slot_eligibility contains an empty slot name'
                    )
                if not isinstance(raw_positions, (list, tuple)) or not raw_positions:
                    raise LeagueProfileError(
                        f'slot_eligibility for {name} must contain positions'
                    )
                positions = tuple(
                    str(position).upper().strip() for position in raw_positions
                )
                if any(not position for position in positions):
                    raise LeagueProfileError(
                        f'slot_eligibility for {name} contains an empty position'
                    )
                if len(set(positions)) != len(positions):
                    raise LeagueProfileError(
                        f'slot_eligibility for {name} contains duplicate positions'
                    )
                slot_eligibility[name] = positions
            missing_eligibility = [
                slot
                for slot, count in roster.items()
                if count > 0 and slot not in slot_eligibility
            ]
            if missing_eligibility:
                raise LeagueProfileError(
                    f'slot_eligibility is missing roster slots: {missing_eligibility}'
                )

        team_count = int(value['team_count'])
        raw_draft_slot = value.get('draft_slot')
        draft_slot = None if raw_draft_slot is None else int(raw_draft_slot)
        if team_count < 2:
            raise LeagueProfileError('team_count must be at least 2')
        if draft_slot is not None and not 1 <= draft_slot <= team_count:
            raise LeagueProfileError('draft_slot must be within team_count')
        draft_format = str(value['draft_format']).strip().lower()
        if draft_format != 'snake':
            raise LeagueProfileError(
                'Only an explicitly configured snake draft is supported'
            )

        scoring_items = {
            str(stat_id): float(points)
            for stat_id, points in value['scoring_items'].items()
        }
        if not scoring_items:
            raise LeagueProfileError('scoring_items must not be empty')
        scoring_overrides: dict[str, dict[str, float]] = {}
        for position, raw_overrides in (value.get('scoring_overrides') or {}).items():
            normalized_position = str(position).upper()
            if normalized_position not in REQUIRED_ROSTER_SLOTS:
                raise LeagueProfileError(
                    f'Unknown scoring override position: {normalized_position}'
                )
            if not isinstance(raw_overrides, Mapping):
                raise LeagueProfileError(
                    f'scoring_overrides for {normalized_position} must be a mapping'
                )
            scoring_overrides[normalized_position] = {
                str(stat_id): float(points) for stat_id, points in raw_overrides.items()
            }
        bonuses: list[dict[str, Any]] = []
        for raw_bonus in value['bonuses']:
            required_bonus = {'stat_id', 'threshold', 'points', 'scope'}
            if not required_bonus.issubset(raw_bonus):
                raise LeagueProfileError(
                    f'bonus rule is missing fields: {sorted(required_bonus.difference(raw_bonus))}'
                )
            if raw_bonus['scope'] != 'weekly':
                raise LeagueProfileError(
                    'Only explicit weekly bonus rules are supported'
                )
            bonuses.append(
                {
                    'stat_id': str(raw_bonus['stat_id']),
                    'threshold': float(raw_bonus['threshold']),
                    'points': float(raw_bonus['points']),
                    'scope': 'weekly',
                }
            )
        flex_eligible = tuple(slot_eligibility.get('FLEX', ()))
        if roster['FLEX'] > 0 and not flex_eligible:
            raise LeagueProfileError('flex_eligible is required when FLEX slots exist')

        bench_slots = _nonnegative_integer(value['bench_slots'], 'bench_slots')
        ir_slots = _nonnegative_integer(value['ir_slots'], 'ir_slots')

        return cls(
            profile_id=str(value['profile_id']),
            league_name=str(value['league_name']),
            season=int(value['season']),
            team_count=team_count,
            draft_format=draft_format,
            draft_slot=draft_slot,
            scoring_label=str(value['scoring_label']),
            scoring_items=scoring_items,
            scoring_overrides=scoring_overrides,
            bonuses=tuple(bonuses),
            roster_slots=roster,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
            flex_eligible=flex_eligible,
            waiver_type=str(value['waiver_type']),
            waiver_constraints=tuple(value.get('waiver_constraints') or ()),
            settings_source=str(value['settings_source']),
            settings_verified_at=str(value['settings_verified_at']),
            slot_eligibility=slot_eligibility,
            _slot_eligibility_explicit=explicit_slot_eligibility,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        explicit = bool(value.pop('_slot_eligibility_explicit', False))
        # Profiles written before the generic slot contract must continue to
        # produce the same digest so existing checkpoints remain recoverable.
        if explicit:
            value['slot_eligibility'] = {
                name: list(positions)
                for name, positions in self.slot_eligibility.items()
            }
        else:
            value.pop('slot_eligibility', None)
        return value

    def eligible_positions(self, slot: str) -> tuple[str, ...]:
        """Return normalized eligibility for one configured roster slot."""
        return self.slot_eligibility.get(str(slot).upper(), ())

    def slot_groups(self) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
        """Return configured slot groups with positive demand in stable order."""
        return tuple(
            (name, int(self.roster_slots.get(name, 0)), positions)
            for name, positions in self.slot_eligibility.items()
            if int(self.roster_slots.get(name, 0)) > 0
        )

    def active_roster_size(self) -> int:
        return sum(self.roster_slots.values()) + self.bench_slots

    def total_draft_picks(self) -> int:
        """Return the number of picks in the configured snake draft."""
        return self.active_roster_size() * self.team_count

    def validate_runtime_slot(self, slot: int) -> None:
        """Validate a draft slot entered by the user on draft day."""
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise LeagueProfileError('draft_slot must be an integer')
        if not 1 <= slot <= self.team_count:
            raise LeagueProfileError('draft_slot must be within team_count')
