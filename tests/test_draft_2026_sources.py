from __future__ import annotations

import pandas as pd
import pytest

from ffbayes.draft_2026.sources import (
    CoverageRequirements,
    SemanticInputError,
    fetch_espn_player_payload,
    parse_espn_player_payload,
    reconcile_current_players,
    validate_source_coverage,
)


class _Response:
    status_code = 200
    content = b'{"players": []}'

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {'players': []}


class _Session:
    def get(self, *args: object, **kwargs: object) -> _Response:
        return _Response()


def test_espn_manifest_records_cache_off_provenance() -> None:
    _, manifest = fetch_espn_player_payload(2026, session=_Session())

    assert manifest['cache_mode'] == 'off'
    assert manifest['http_status'] == 200
    assert manifest['season'] == 2026


def _entry(
    player_id: int,
    name: str,
    position_id: int,
    *,
    active: bool = True,
    pro_team_id: int = 1,
    season: int = 2026,
    adp: float | None = 25.0,
    projected_points: float = 200.0,
) -> dict:
    ownership = {'date': 1_787_400_000_000}
    if adp is not None:
        ownership['averageDraftPosition'] = adp
    return {
        'player': {
            'id': player_id,
            'fullName': name,
            'active': active,
            'proTeamId': pro_team_id,
            'defaultPositionId': position_id,
            'ownership': ownership,
            'draftRanksByRankType': {'PPR': {'rank': 25}},
            'stats': [
                {
                    'seasonId': season,
                    'statSourceId': 1,
                    'statSplitTypeId': 0,
                    'scoringPeriodId': 0,
                    'appliedTotal': projected_points,
                    'stats': {'3': 4000.0, '4': 30.0},
                }
            ],
        }
    }


def test_parse_espn_payload_rejects_noncurrent_and_historical_entries() -> None:
    payload = {
        'players': [
            _entry(1, 'Current Runner', 2),
            _entry(2, 'Retired Legend', 1, active=False),
            _entry(3, 'Historical Only', 1, pro_team_id=0),
            _entry(4, 'Wrong Season', 3, season=2025),
        ]
    }

    parsed = parse_espn_player_payload(payload, season=2026)

    assert parsed['name'].tolist() == ['Current Runner']
    assert parsed['position'].tolist() == ['RB']
    assert parsed.loc[0, 'projection_season'] == 2026


def test_current_roster_reconciliation_excludes_retired_without_blacklist() -> None:
    espn = parse_espn_player_payload(
        {
            'players': [
                _entry(101, 'Active Quarterback', 1),
                _entry(102, 'Retired Quarterback', 1),
                _entry(103, 'Rookie Quarterback', 1),
                _entry(200, 'Bills D/ST', 16),
            ]
        },
        season=2026,
    )
    roster = pd.DataFrame(
        {
            'espn_id': [101, 102, None],
            'full_name': [
                'Active Quarterback',
                'Retired Quarterback',
                'Rookie Quarterback',
            ],
            'position': ['QB', 'QB', 'QB'],
            'team': ['BUF', None, 'LV'],
            'status': ['ACT', 'RET', 'RES'],
            'season': [2026, 2026, 2026],
        }
    )

    current = reconcile_current_players(espn, roster, season=2026)

    assert set(current['name']) == {
        'Active Quarterback',
        'Rookie Quarterback',
        'Bills D/ST',
    }
    assert not current['name'].str.contains('Retired').any()


def test_projection_coverage_fails_closed_for_truncated_pool() -> None:
    frame = pd.DataFrame(
        {
            'espn_id': list(range(40)),
            'name': [f'Player {i}' for i in range(40)],
            'position': ['QB'] * 10 + ['RB'] * 10 + ['WR'] * 10 + ['TE'] * 10,
            'projected_points': [200.0] * 40,
            'adp': list(range(1, 41)),
            'adp_updated_at': [pd.Timestamp('2026-08-22', tz='UTC')] * 40,
            'projection_season': [2026] * 40,
            'eligibility_status': ['current'] * 40,
        }
    )

    production = CoverageRequirements.production()
    requirements = CoverageRequirements(
        minimum_players={position: 0 for position in production.minimum_players},
        minimum_projections=production.minimum_projections,
        market_top_n=production.market_top_n,
        minimum_market_fraction=production.minimum_market_fraction,
        max_market_age_days=production.max_market_age_days,
    )
    with pytest.raises(SemanticInputError, match='Projection coverage'):
        validate_source_coverage(frame, 2026, requirements)


def test_adp_coverage_fails_closed_without_neutral_substitution() -> None:
    requirements = CoverageRequirements(
        minimum_players={'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'K': 1, 'DST': 1},
        minimum_projections={'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'K': 1, 'DST': 1},
        market_top_n=6,
        minimum_market_fraction=0.9,
        max_market_age_days=7,
    )
    frame = pd.DataFrame(
        {
            'espn_id': list(range(6)),
            'name': list('ABCDEF'),
            'position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'projected_points': [200, 190, 180, 170, 160, 150],
            'adp': [1.0, None, None, None, None, None],
            'adp_updated_at': [pd.Timestamp('2026-08-22', tz='UTC')] * 6,
            'projection_season': [2026] * 6,
            'eligibility_status': ['current'] * 6,
        }
    )

    with pytest.raises(SemanticInputError, match='ADP coverage'):
        validate_source_coverage(
            frame, 2026, requirements, as_of=pd.Timestamp('2026-08-22', tz='UTC')
        )


def test_coverage_rejects_missing_stable_player_identity() -> None:
    requirements = CoverageRequirements(
        minimum_players={'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'K': 1, 'DST': 1},
        minimum_projections={'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'K': 1, 'DST': 1},
        market_top_n=6,
        minimum_market_fraction=1.0,
        max_market_age_days=7,
    )
    frame = pd.DataFrame(
        {
            'name': list('ABCDEF'),
            'position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'projected_points': [200, 190, 180, 170, 160, 150],
            'adp': [1.0] * 6,
            'adp_updated_at': [pd.Timestamp('2026-08-22', tz='UTC')] * 6,
            'projection_season': [2026] * 6,
            'eligibility_status': ['current'] * 6,
        }
    )

    with pytest.raises(SemanticInputError, match='espn_id'):
        validate_source_coverage(
            frame, 2026, requirements, as_of=pd.Timestamp('2026-08-22', tz='UTC')
        )
