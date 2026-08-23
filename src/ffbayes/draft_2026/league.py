"""Explicit league-profile contract for current-season draft valuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class LeagueProfileError(ValueError):
    """Raised when a league profile is incomplete or internally inconsistent."""


REQUIRED_ROSTER_SLOTS = ('QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K')


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
            'flex_eligible',
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
            or (
                key not in ('bonuses', 'waiver_constraints')
                and value.get(key) == []
            )
            or (key == 'scoring_items' and not value.get(key))
        ]
        if missing:
            raise LeagueProfileError(
                f'League profile has unresolved required fields: {missing}'
            )

        roster = {str(k).upper(): int(v) for k, v in value['roster_slots'].items()}
        missing_slots = [slot for slot in REQUIRED_ROSTER_SLOTS if slot not in roster]
        if missing_slots:
            raise LeagueProfileError(f'roster_slots is missing: {missing_slots}')
        if any(roster[slot] < 0 for slot in REQUIRED_ROSTER_SLOTS):
            raise LeagueProfileError('roster_slots cannot contain negative values')

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
        flex_eligible = tuple(
            str(position).upper() for position in value['flex_eligible']
        )
        if roster['FLEX'] > 0 and not flex_eligible:
            raise LeagueProfileError('flex_eligible is required when FLEX slots exist')

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
            bench_slots=int(value['bench_slots']),
            ir_slots=int(value['ir_slots']),
            flex_eligible=flex_eligible,
            waiver_type=str(value['waiver_type']),
            waiver_constraints=tuple(value.get('waiver_constraints') or ()),
            settings_source=str(value['settings_source']),
            settings_verified_at=str(value['settings_verified_at']),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
