"""Current-season player, projection, and market source contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests

ESPN_POSITION_IDS = {1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST'}
CURRENT_ROSTER_STATUSES = frozenset({'ACT', 'RES', 'E14'})
ESPN_PLAYER_URL = (
    'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/'
    '{season}/segments/0/leaguedefaults/1?view=kona_player_info'
)


class SemanticInputError(RuntimeError):
    """Raised when an input can be read but is unsafe for draft decisions."""


@dataclass(frozen=True)
class CoverageRequirements:
    """Minimum source coverage required before valuation is allowed."""

    minimum_players: Mapping[str, int]
    minimum_projections: Mapping[str, int]
    market_top_n: int
    minimum_market_fraction: float
    max_market_age_days: int

    @classmethod
    def production(cls) -> 'CoverageRequirements':
        return cls(
            minimum_players={
                'QB': 48,
                'RB': 90,
                'WR': 150,
                'TE': 70,
                'K': 28,
                'DST': 28,
            },
            minimum_projections={
                'QB': 48,
                'RB': 90,
                'WR': 150,
                'TE': 70,
                'K': 28,
                'DST': 28,
            },
            market_top_n=200,
            minimum_market_fraction=0.95,
            max_market_age_days=7,
        )


def _normalize_name(value: object) -> str:
    name = ' '.join(str(value or '').strip().lower().split())
    for suffix in (' jr.', ' sr.', ' iii', ' ii', ' iv', ' v'):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _season_projection(
    player: Mapping[str, Any], season: int
) -> Mapping[str, Any] | None:
    matches = [
        stat
        for stat in player.get('stats', [])
        if stat.get('seasonId') == season
        and stat.get('statSourceId') == 1
        and stat.get('statSplitTypeId') == 0
        and stat.get('scoringPeriodId') == 0
        and bool(stat.get('stats'))
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _weekly_projections(
    player: Mapping[str, Any], season: int
) -> list[dict[str, float]]:
    weeks = [
        stat
        for stat in player.get('stats', [])
        if stat.get('seasonId') == season
        and stat.get('statSourceId') == 1
        and stat.get('statSplitTypeId') == 1
        and int(stat.get('scoringPeriodId') or 0) > 0
        and bool(stat.get('stats'))
    ]
    weeks.sort(key=lambda stat: int(stat['scoringPeriodId']))
    return [
        {str(key): float(value) for key, value in stat['stats'].items()}
        for stat in weeks
    ]


def parse_espn_player_payload(payload: Mapping[str, Any], season: int) -> pd.DataFrame:
    """Parse only current, rostered, projected players from ESPN's season feed."""
    players = payload.get('players')
    if not isinstance(players, list):
        raise SemanticInputError('ESPN player response has no players list')

    rows: list[dict[str, Any]] = []
    for entry in players:
        player = entry.get('player', entry) if isinstance(entry, Mapping) else {}
        raw_position = player.get('defaultPositionId')
        position = (
            ESPN_POSITION_IDS.get(raw_position)
            if isinstance(raw_position, int)
            else None
        )
        projection = _season_projection(player, season)
        if (
            position is None
            or player.get('active') is not True
            or not player.get('proTeamId')
            or projection is None
        ):
            continue

        ownership = player.get('ownership') or {}
        market_date = ownership.get('date')
        rows.append(
            {
                'espn_id': int(player['id']),
                'name': str(player.get('fullName') or '').strip(),
                'normalized_name': _normalize_name(player.get('fullName')),
                'position': position,
                'pro_team_id': int(player['proTeamId']),
                'projection_season': int(projection['seasonId']),
                'projected_points_standard': float(projection['appliedTotal']),
                'projection_stats': {
                    str(key): float(value)
                    for key, value in (projection.get('stats') or {}).items()
                },
                'weekly_projection_stats': _weekly_projections(player, season),
                'adp': pd.to_numeric(
                    ownership.get('averageDraftPosition'), errors='coerce'
                ),
                'adp_updated_at': (
                    pd.to_datetime(market_date, unit='ms', utc=True)
                    if market_date is not None
                    else pd.NaT
                ),
                'market_rank_standard': pd.to_numeric(
                    (player.get('draftRanksByRankType') or {})
                    .get('STANDARD', {})
                    .get('rank'),
                    errors='coerce',
                ),
                'market_rank_ppr': pd.to_numeric(
                    (player.get('draftRanksByRankType') or {})
                    .get('PPR', {})
                    .get('rank'),
                    errors='coerce',
                ),
                'eligibility_status': 'espn_current',
            }
        )

    if not rows:
        raise SemanticInputError('ESPN returned no current projected players')
    return pd.DataFrame(rows).sort_values(['position', 'name']).reset_index(drop=True)


def reconcile_current_players(
    espn_players: pd.DataFrame, nfl_roster: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Require a current NFL roster match for players; retain current team defenses."""
    required = {'espn_id', 'name', 'position', 'projection_season'}
    missing = required.difference(espn_players.columns)
    if missing:
        raise SemanticInputError(
            f'ESPN player frame is missing columns: {sorted(missing)}'
        )

    roster = nfl_roster.copy()
    roster['season'] = pd.to_numeric(roster.get('season'), errors='coerce')
    roster['status'] = roster.get('status', '').astype(str).str.upper()
    roster = roster[
        roster['season'].eq(season) & roster['status'].isin(CURRENT_ROSTER_STATUSES)
    ].copy()
    roster['normalized_name'] = roster.get('full_name', '').map(_normalize_name)
    roster_ids = set(
        pd.to_numeric(roster.get('espn_id'), errors='coerce').dropna().astype(int)
    )
    roster_names = set(roster['normalized_name'].dropna())

    frame = espn_players.copy()
    normalized = frame.get('normalized_name', frame['name'].map(_normalize_name))
    id_match = pd.to_numeric(frame['espn_id'], errors='coerce').isin(roster_ids)
    name_match = normalized.isin(roster_names)
    current = frame['position'].eq('DST') | id_match | name_match
    frame = frame[current].copy()
    frame['eligibility_status'] = 'current'
    if frame.empty:
        raise SemanticInputError('No ESPN players matched the current NFL roster')
    return frame.reset_index(drop=True)


def validate_source_coverage(
    frame: pd.DataFrame,
    season: int,
    requirements: CoverageRequirements,
    *,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Validate season, eligibility, positional depth, projections, and ADP."""
    if frame.empty:
        raise SemanticInputError('Current player universe is empty')
    if 'espn_id' not in frame:
        raise SemanticInputError('Current player universe is missing espn_id')
    stable_ids = pd.to_numeric(frame['espn_id'], errors='coerce')
    if stable_ids.isna().any() or stable_ids.duplicated().any():
        raise SemanticInputError('Current player universe has invalid espn_id values')
    if not frame['projection_season'].eq(season).all():
        raise SemanticInputError(f'Projection season does not equal {season}')
    if not frame['eligibility_status'].eq('current').all():
        raise SemanticInputError(
            'Player universe contains noncurrent eligibility records'
        )

    position_counts = frame.groupby('position')['name'].nunique().to_dict()
    shallow_players = {
        position: (position_counts.get(position, 0), minimum)
        for position, minimum in requirements.minimum_players.items()
        if position_counts.get(position, 0) < minimum
    }
    if shallow_players:
        raise SemanticInputError(
            f'Current-player coverage is inadequate: {shallow_players}'
        )

    projection_counts = (
        frame.assign(
            valid_projection=pd.to_numeric(
                frame.get('projected_points', frame.get('projected_points_standard')),
                errors='coerce',
            ).notna()
        )
        .groupby('position')['valid_projection']
        .sum()
        .astype(int)
        .to_dict()
    )
    shallow_projections = {
        position: (projection_counts.get(position, 0), minimum)
        for position, minimum in requirements.minimum_projections.items()
        if projection_counts.get(position, 0) < minimum
    }
    if shallow_projections:
        raise SemanticInputError(
            f'Projection coverage is inadequate: {shallow_projections}'
        )

    top_n = min(requirements.market_top_n, len(frame))
    market = frame.sort_values(
        frame.get('market_rank_ppr', pd.Series(index=frame.index, dtype=float)).name
        if 'market_rank_ppr' in frame
        else 'projected_points',
        na_position='last',
    ).head(top_n)
    market_fraction = pd.to_numeric(market['adp'], errors='coerce').notna().mean()
    if market_fraction < requirements.minimum_market_fraction:
        raise SemanticInputError(
            'ADP coverage is inadequate: '
            f'{market_fraction:.1%} < {requirements.minimum_market_fraction:.1%}'
        )

    reference_time = as_of or pd.Timestamp.now(tz='UTC')
    market_dates = pd.to_datetime(market['adp_updated_at'], errors='coerce', utc=True)
    if market_dates.isna().any():
        raise SemanticInputError('ADP provenance timestamps are missing')
    oldest_age = reference_time - market_dates.min()
    if oldest_age > pd.Timedelta(days=requirements.max_market_age_days):
        raise SemanticInputError(
            f'ADP market data is stale by {oldest_age.total_seconds() / 86400:.1f} days'
        )

    return {
        'season': season,
        'rows': int(len(frame)),
        'position_counts': position_counts,
        'projection_counts': projection_counts,
        'market_top_n': top_n,
        'market_coverage': float(market_fraction),
        'oldest_market_timestamp': market_dates.min().isoformat(),
        'status': 'passed',
    }


def fetch_espn_player_payload(
    season: int, *, session: requests.Session | None = None, timeout: float = 30.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch ESPN data without fallback and return response provenance."""
    client = session or requests.Session()
    url = ESPN_PLAYER_URL.format(season=season)
    player_filter = {
        'players': {
            'filterActive': {'value': True},
            'limit': 2000,
            'sortPercOwned': {'sortPriority': 1, 'sortAsc': False},
        }
    }
    response = client.get(
        url,
        headers={
            'x-fantasy-filter': json.dumps(player_filter, separators=(',', ':')),
            'User-Agent': 'ffbayes/0.1 current-season draft validation',
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.content
    try:
        payload = response.json()
    except ValueError as exc:
        raise SemanticInputError('ESPN player response was not valid JSON') from exc
    fetched_at = datetime.now(timezone.utc).isoformat()
    canonical_payload = json.dumps(payload, sort_keys=True).encode('utf-8')
    provenance = {
        'source': 'espn_fantasy_players',
        'url': url,
        'season': season,
        'fetched_at': fetched_at,
        'cache_mode': 'off',
        'http_status': response.status_code,
        'bytes': len(raw),
        'sha256': hashlib.sha256(canonical_payload).hexdigest(),
        'transport_sha256': hashlib.sha256(raw).hexdigest(),
        'reported_players': len(payload.get('players', [])),
    }
    return payload, provenance


def write_source_snapshot(
    payload: Mapping[str, Any], provenance: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Write a run-scoped source snapshot and its matching manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    payload_path = output_dir / 'espn_players.json'
    manifest_path = output_dir / 'source_manifest.json'
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    manifest_path.write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True), encoding='utf-8'
    )
    return payload_path, manifest_path
