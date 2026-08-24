"""Isolated current-season draft pipeline and output provenance contract."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nflreadpy as nfl
import pandas as pd
import requests
from nflreadpy.config import CacheMode, get_config, update_config

from ffbayes.draft_2026.engine import build_draft_board
from ffbayes.draft_2026.league import LeagueProfile, LeagueProfileError
from ffbayes.draft_2026.sources import (
    CoverageRequirements,
    SemanticInputError,
    fetch_espn_player_payload,
    parse_espn_player_payload,
    reconcile_current_players,
    validate_source_coverage,
)

SCHEMA_VERSION = 'draft_2026_v1'
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = PROJECT_ROOT / 'config' / 'leagues'
EXAMPLE_PROFILE = PROFILE_ROOT / 'example_2026.json'


def default_profile_paths() -> tuple[Path, ...]:
    """Return local profiles when present, otherwise the portable example."""
    local_profiles = tuple(sorted(PROFILE_ROOT.glob('*.local.json')))
    return local_profiles or (EXAMPLE_PROFILE,)


class OutputProvenanceError(ValueError):
    """Raised when a board output cannot be tied to its validated inputs."""


@dataclass(frozen=True)
class FreshInputs:
    """One validated source snapshot shared by every league board in a run."""

    payload: dict[str, Any]
    source_manifest: dict[str, Any]
    roster: pd.DataFrame
    roster_manifest: dict[str, Any]
    players: pd.DataFrame
    coverage: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(frame.to_dict(orient='records'))


def build_dashboard_payload(
    board: pd.DataFrame,
    profile: LeagueProfile,
    *,
    source_manifest: Mapping[str, Any],
    roster_manifest: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
    code_revision: str,
) -> dict[str, Any]:
    """Build a payload bound to exact source, profile, code, and board inputs."""
    profile_value = profile.to_dict()
    table_columns = [
        'board_rank',
        'espn_id',
        'name',
        'position',
        'projected_points',
        'replacement_level',
        'vor',
        'scarcity',
        'adp',
        'availability_next_pick',
        'recommendation',
        'roster_status',
        'is_available',
    ]
    decision_table = _records(board[[c for c in table_columns if c in board]])
    if not decision_table or any(row.get('espn_id') is None for row in decision_table):
        raise OutputProvenanceError('Decision table requires stable espn_id values')
    runtime_state = dict(
        board.attrs.get(
            'runtime_state',
            {
                'draft_slot': profile.draft_slot,
                'current_pick': board.attrs.get('current_pick'),
                'taken_ids': [],
                'your_ids': [],
                'queue_ids': [],
            },
        )
    )
    for key in ('taken_ids', 'your_ids', 'queue_ids'):
        runtime_state[key] = sorted({int(value) for value in runtime_state.get(key, [])})
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        'schema_version': SCHEMA_VERSION,
        'season': profile.season,
        'generated_at': generated_at,
        'league_profile': profile_value,
        'current_pick': board.attrs.get('current_pick'),
        'next_pick': board.attrs.get('next_pick'),
        'runtime_state': runtime_state,
        'replacement': board.attrs['replacement'],
        'coverage_report': dict(coverage_report),
        'decision_table': decision_table,
        'provenance': {
            'code_revision': code_revision,
            'profile_sha256': _sha256_json(profile_value),
            'state_sha256': _sha256_json(runtime_state),
            'board_sha256': _sha256_json(decision_table),
            'source_manifests': [dict(source_manifest), dict(roster_manifest)],
        },
    }
    validate_output_provenance(payload)
    return payload


def validate_output_provenance(payload: Mapping[str, Any]) -> None:
    """Fail closed when output and input provenance are inconsistent."""
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise OutputProvenanceError('Unsupported dashboard schema version')
    season = payload.get('season')
    profile = payload.get('league_profile') or {}
    if profile.get('season') != season:
        raise OutputProvenanceError(
            'League profile season does not match output season'
        )
    if (payload.get('coverage_report') or {}).get('status') != 'passed':
        raise OutputProvenanceError('Coverage validation did not pass')
    provenance = payload.get('provenance') or {}
    if not provenance.get('code_revision'):
        raise OutputProvenanceError('Code revision is missing')
    generated_at = pd.Timestamp(payload['generated_at'])
    for manifest in provenance.get('source_manifests') or []:
        if manifest.get('season') != season:
            raise OutputProvenanceError(
                'Source manifest season does not match output season'
            )
        digest = manifest.get('sha256')
        if not isinstance(digest, str) or not digest:
            raise OutputProvenanceError('Source manifest digest is missing')
        fetched_at = pd.Timestamp(manifest.get('fetched_at'))
        if fetched_at > generated_at:
            raise OutputProvenanceError(
                'Source timestamp is later than output timestamp'
            )
    if provenance.get('profile_sha256') != _sha256_json(profile):
        raise OutputProvenanceError('League profile digest is inconsistent')
    table = payload.get('decision_table')
    if not isinstance(table, list) or not table:
        raise OutputProvenanceError('Decision table is empty')
    if provenance.get('board_sha256') != _sha256_json(table):
        raise OutputProvenanceError('Decision table digest is inconsistent')
    runtime_state = payload.get('runtime_state')
    if not isinstance(runtime_state, Mapping):
        raise OutputProvenanceError('Runtime state is missing')
    if provenance.get('state_sha256') != _sha256_json(runtime_state):
        raise OutputProvenanceError('Runtime state digest is inconsistent')
    if any(row.get('espn_id') is None for row in table if isinstance(row, Mapping)):
        raise OutputProvenanceError('Decision table contains a missing espn_id')


def render_dashboard_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained, read-only board for the validated payload."""
    validate_output_provenance(payload)
    title = html.escape(str(payload['league_profile']['league_name']))
    rows = []
    for player in payload['decision_table'][:100]:
        availability = player.get('availability_next_pick')
        availability_text = (
            'Draft slot required'
            if availability is None
            else f'{float(availability):.0%}'
        )
        rows.append(
            '<tr>'
            f'<td>{int(player["board_rank"])}</td>'
            f'<td>{html.escape(str(player["name"]))}</td>'
            f'<td>{html.escape(str(player["position"]))}</td>'
            f'<td>{float(player["projected_points"]):.1f}</td>'
            f'<td>{float(player["vor"]):.1f}</td>'
            f'<td>{float(player["adp"]):.1f}</td>'
            f'<td>{availability_text}</td>'
            f'<td>{html.escape(str(player["recommendation"]))}</td>'
            '</tr>'
        )
    embedded = (
        _canonical_json(payload)
        .replace('&', '\\u0026')
        .replace('<', '\\u003c')
        .replace('>', '\\u003e')
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} 2026 Draft Board</title>
<style>body{{font-family:system-ui;margin:2rem;background:#07111f;color:#f8fafc}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #334155;text-align:left}}th{{position:sticky;top:0;background:#0f172a}}.meta{{color:#94a3b8}}</style></head>
<body><h1>{title} — 2026 Draft Board</h1>
<p class="meta">Generated {html.escape(str(payload['generated_at']))}; next pick {payload['next_pick'] if payload['next_pick'] is not None else 'slot required'}; validated current inputs only.</p>
<table id="draft-board"><thead><tr><th>Rank</th><th>Player</th><th>Pos</th><th>Proj</th><th>VOR</th><th>ADP</th><th>Next-pick availability</th><th>Action</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script id="draft-2026-payload" type="application/json">{embedded}</script></body></html>"""


def _git_revision(project_root: Path) -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _roster_manifest(roster: pd.DataFrame, season: int) -> dict[str, Any]:
    canonical = roster.sort_values(
        [column for column in ['full_name', 'team', 'position'] if column in roster]
    ).to_csv(index=False)
    return {
        'source': 'nflverse_rosters_via_nflreadpy',
        'url': (
            'https://github.com/nflverse/nflverse-data/releases/download/'
            f'rosters/roster_{season}.parquet'
        ),
        'season': season,
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'cache_mode': 'off',
        'rows': int(len(roster)),
        'sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
    }


def _load_fresh_roster(season: int) -> pd.DataFrame:
    previous_cache_mode = get_config().cache_mode
    update_config(cache_mode=CacheMode.OFF)
    try:
        return nfl.load_rosters([season]).to_pandas()
    finally:
        update_config(cache_mode=previous_cache_mode)


def load_fresh_inputs(season: int) -> FreshInputs:
    """Fetch and validate the public inputs once, without a cache fallback."""
    payload, source_manifest = fetch_espn_player_payload(season)
    espn = parse_espn_player_payload(payload, season)
    roster = _load_fresh_roster(season)
    roster_manifest = _roster_manifest(roster, season)
    players = reconcile_current_players(espn, roster, season)
    coverage = validate_source_coverage(
        players, season, CoverageRequirements.production()
    )
    return FreshInputs(
        payload=payload,
        source_manifest=source_manifest,
        roster=roster,
        roster_manifest=roster_manifest,
        players=players,
        coverage=coverage,
    )


def run_profile(
    profile: LeagueProfile,
    output_dir: Path,
    *,
    project_root: Path,
    fresh_inputs: FreshInputs | None = None,
) -> dict[str, Path]:
    """Fetch fresh inputs and build one isolated league board."""
    inputs = fresh_inputs or load_fresh_inputs(profile.season)
    if inputs.source_manifest.get('season') != profile.season:
        raise OutputProvenanceError('Fresh input season does not match league profile')
    board = build_draft_board(inputs.players, profile)
    dashboard_payload = build_dashboard_payload(
        board,
        profile,
        source_manifest=inputs.source_manifest,
        roster_manifest=inputs.roster_manifest,
        coverage_report=inputs.coverage,
        code_revision=_git_revision(project_root),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = output_dir / 'inputs'
    inputs_dir.mkdir()
    outputs_dir = output_dir / 'outputs'
    outputs_dir.mkdir()
    (inputs_dir / 'espn_players.json').write_text(
        json.dumps(inputs.payload, sort_keys=True), encoding='utf-8'
    )
    (inputs_dir / 'source_manifests.json').write_text(
        json.dumps([inputs.source_manifest, inputs.roster_manifest], indent=2),
        encoding='utf-8',
    )
    inputs.roster.sort_values(
        [
            column
            for column in ['full_name', 'team', 'position']
            if column in inputs.roster
        ]
    ).to_csv(inputs_dir / 'nfl_roster.csv', index=False)
    board.to_csv(outputs_dir / 'draft_board.csv', index=False)
    (outputs_dir / 'dashboard_payload.json').write_text(
        json.dumps(dashboard_payload, indent=2, default=_json_value), encoding='utf-8'
    )
    (outputs_dir / 'draft_board.html').write_text(
        render_dashboard_html(dashboard_payload), encoding='utf-8'
    )
    return {
        'run_dir': output_dir,
        'board': outputs_dir / 'draft_board.csv',
        'payload': outputs_dir / 'dashboard_payload.json',
        'dashboard': outputs_dir / 'draft_board.html',
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build fail-closed, league-specific 2026 draft boards.'
    )
    parser.add_argument('--profile', action='append', type=Path)

    parser.add_argument(
        '--output-root', type=Path, default=PROJECT_ROOT / 'runtime' / 'runs' / 'draft_2026'
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = PROJECT_ROOT
    profile_paths = tuple(args.profile or default_profile_paths())
    profiles: list[LeagueProfile] = []
    failures: list[dict[str, Any]] = []
    for path in profile_paths:
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            profiles.append(LeagueProfile.from_mapping(raw))
        except (OSError, json.JSONDecodeError, LeagueProfileError) as exc:
            failures.append(
                {
                    'profile': str(path),
                    'failure_mode': str(exc),
                    'failure_class': 'incomplete_local_configuration',
                    'configuration_source': 'explicit local league profile',
                    'external_dependency': False,
                    'pipeline_dependencies': [
                        'league scoring',
                        'replacement levels',
                        'scarcity',
                        'pick recommendations',
                    ],
                    'resolution': 'supply the missing settings explicitly in the profile',
                    'status': 'blocked',
                }
            )
    if failures:
        print(json.dumps({'status': 'blocked', 'profiles': failures}, indent=2))
        return 2

    seasons = {profile.season for profile in profiles}
    if len(seasons) != 1:
        print(
            json.dumps(
                {
                    'status': 'blocked',
                    'failure_mode': 'League profiles do not share one source season',
                },
                indent=2,
            )
        )
        return 2

    try:
        fresh_inputs = load_fresh_inputs(seasons.pop())
        for profile in profiles:
            build_draft_board(fresh_inputs.players, profile)
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        print(
            json.dumps(
                {
                    'status': 'blocked',
                    'source': 'ESPN public current-season fantasy player feed',
                    'failure_mode': f'{type(exc).__name__}: {exc}',
                    'http_status': status,
                    'transience': (
                        'possibly transient'
                        if status is None or status >= 500
                        else 'structural until source access or URL is repaired'
                    ),
                    'pipeline_dependencies': [
                        'player universe',
                        'projections',
                        'ADP',
                        'replacement levels',
                        'recommendations',
                    ],
                    'alternate_source': 'none used; silent fallback is prohibited',
                },
                indent=2,
            )
        )
        return 3
    except (ConnectionError, OSError) as exc:
        print(
            json.dumps(
                {
                    'status': 'blocked',
                    'source': 'nflverse current-season roster release',
                    'failure_mode': f'{type(exc).__name__}: {exc}',
                    'transience': 'possibly transient network or upstream release failure',
                    'pipeline_dependencies': [
                        'current-player eligibility',
                        'inactive-player exclusion',
                        'recommendations',
                    ],
                    'alternate_source': 'none used; silent fallback is prohibited',
                },
                indent=2,
            )
        )
        return 3
    except (SemanticInputError, OutputProvenanceError) as exc:
        print(
            json.dumps(
                {
                    'status': 'blocked',
                    'source': '2026 public player, projection, ADP, or roster inputs',
                    'failure_mode': str(exc),
                    'transience': (
                        'unknown; source content or league model is semantically inadequate'
                    ),
                    'pipeline_dependencies': [
                        'replacement levels',
                        'recommendations',
                        'dashboard payload',
                    ],
                    'alternate_source': 'none used; silent fallback is prohibited',
                },
                indent=2,
            )
        )
        return 4

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for profile in profiles:
        run_profile(
            profile,
            args.output_root / timestamp / profile.profile_id,
            project_root=project_root,
            fresh_inputs=fresh_inputs,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
