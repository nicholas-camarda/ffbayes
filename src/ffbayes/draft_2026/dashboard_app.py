"""One-command, loopback-only 2026 draft dashboard service."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import webbrowser
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

from ffbayes.draft_2026.engine import build_draft_board
from ffbayes.draft_2026.league import LeagueProfile, LeagueProfileError
from ffbayes.draft_2026.pipeline import (
    PROJECT_ROOT,
    FreshInputs,
    OutputProvenanceError,
    _git_revision,
    build_dashboard_payload,
    default_profile_paths,
    load_fresh_inputs,
)
from ffbayes.draft_2026.sources import SemanticInputError


class DashboardRequestError(ValueError):
    """Raised when a dashboard request is malformed or unsafe."""


class DashboardBlockedError(RuntimeError):
    """Raised when required source or profile validation has failed."""


def _integer_list(value: object, field: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DashboardRequestError(f'{field} must be a JSON list of integers')
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise DashboardRequestError(f'{field} must contain only integer IDs')
        result.append(raw)
    if len(result) != len(set(result)):
        raise DashboardRequestError(f'{field} contains duplicate IDs')
    return result


def _safe_filename(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-') or 'snapshot'


class DashboardService:
    """Own one validated source snapshot and all independent league state."""

    def __init__(
        self,
        fresh_inputs: FreshInputs | None,
        profiles: Sequence[LeagueProfile],
        *,
        project_root: Path,
        run_root: Path,
        code_revision: str,
        blocked_error: str | None = None,
        blocked_details: Mapping[str, Any] | None = None,
    ) -> None:
        self.fresh_inputs = fresh_inputs
        self.profiles = {profile.profile_id: profile for profile in profiles}
        self.project_root = project_root
        self.run_root = run_root
        self.code_revision = code_revision
        self.blocked_error = blocked_error
        self.blocked_details = dict(blocked_details or {})
        self._state: dict[str, dict[str, Any]] = {
            profile.profile_id: {
                'draft_slot': profile.draft_slot,
                'current_pick': None,
                'taken_ids': [],
                'your_ids': [],
                'queue_ids': [],
            }
            for profile in profiles
        }

    def status_payload(self) -> dict[str, Any]:
        if self.blocked_error:
            payload = {
                'status': 'blocked',
                'season': 2026,
                'error': self.blocked_error,
                'dependent_features': [
                    'current-player universe',
                    'projections',
                    'ADP',
                    'replacement levels',
                    'draft recommendations',
                ],
                'fallback': False,
            }
            payload.update(self.blocked_details)
            return payload
        if self.fresh_inputs is None:
            return {'status': 'blocked', 'season': 2026, 'error': 'Inputs are unavailable'}
        return {
            'status': 'ready',
            'season': 2026,
            'coverage': self.fresh_inputs.coverage,
            'source_manifests': [
                self.fresh_inputs.source_manifest,
                self.fresh_inputs.roster_manifest,
            ],
            'fallback': False,
        }

    def leagues_payload(self) -> dict[str, Any]:
        return {
            'leagues': [
                {
                    'profile_id': profile.profile_id,
                    'league_name': profile.league_name,
                    'season': profile.season,
                    'team_count': profile.team_count,
                    'draft_format': profile.draft_format,
                    'scoring_label': profile.scoring_label,
                    'roster_slots': profile.roster_slots,
                    'bench_slots': profile.bench_slots,
                    'ir_slots': profile.ir_slots,
                    'flex_eligible': profile.flex_eligible,
                    'waiver_type': profile.waiver_type,
                }
                for profile in self.profiles.values()
            ]
        }

    def _known_ids(self) -> set[int]:
        if self.fresh_inputs is None or 'espn_id' not in self.fresh_inputs.players:
            return set()
        values = pd.to_numeric(self.fresh_inputs.players['espn_id'], errors='coerce').dropna()
        return {int(value) for value in values}

    def _request_state(self, request: Mapping[str, Any]) -> tuple[LeagueProfile, dict[str, Any]]:
        if self.blocked_error or self.fresh_inputs is None:
            raise DashboardBlockedError(self.blocked_error or 'Fresh inputs are unavailable')
        profile_id = request.get('profile_id')
        if not isinstance(profile_id, str) or profile_id not in self.profiles:
            raise DashboardRequestError('Unknown profile_id')
        profile = self.profiles[profile_id]
        raw_slot = request.get('draft_slot')
        if raw_slot is not None and (isinstance(raw_slot, bool) or not isinstance(raw_slot, int)):
            raise DashboardRequestError('draft_slot must be an integer or null')
        slot = raw_slot
        if slot is not None:
            try:
                profile.validate_runtime_slot(slot)
            except LeagueProfileError as exc:
                raise DashboardRequestError(str(exc)) from exc
        raw_current = request.get('current_pick')
        if raw_current is not None and (
            isinstance(raw_current, bool) or not isinstance(raw_current, int)
        ):
            raise DashboardRequestError('current_pick must be an integer or null')
        current_pick = raw_current
        if current_pick is None and slot is not None:
            current_pick = slot
        if current_pick is not None and not 1 <= current_pick <= profile.total_draft_picks():
            raise DashboardRequestError('current_pick is outside the configured draft')
        taken_ids = _integer_list(request.get('taken_ids'), 'taken_ids')
        your_ids = _integer_list(request.get('your_ids'), 'your_ids')
        queue_ids = _integer_list(request.get('queue_ids'), 'queue_ids')
        if set(taken_ids).intersection(your_ids):
            raise DashboardRequestError('A player cannot be both taken and yours')
        unknown = (set(taken_ids) | set(your_ids) | set(queue_ids)).difference(self._known_ids())
        if unknown:
            raise DashboardRequestError(f'unknown player espn_id values: {sorted(unknown)}')
        state = {
            'draft_slot': slot,
            'current_pick': current_pick,
            'taken_ids': taken_ids,
            'your_ids': your_ids,
            'queue_ids': queue_ids,
        }
        return replace(profile, draft_slot=slot), state

    def handle_board(self, request: Mapping[str, Any]) -> dict[str, Any]:
        profile, state = self._request_state(request)
        fresh_inputs = self.fresh_inputs
        if fresh_inputs is None:
            raise DashboardBlockedError('Fresh inputs are unavailable')
        board = build_draft_board(
            fresh_inputs.players,
            profile,
            current_pick=state['current_pick'],
            taken_ids=state['taken_ids'],
            your_ids=state['your_ids'],
        )
        board.attrs['runtime_state'] = state
        payload = build_dashboard_payload(
            board,
            profile,
            source_manifest=fresh_inputs.source_manifest,
            roster_manifest=fresh_inputs.roster_manifest,
            coverage_report=fresh_inputs.coverage,
            code_revision=self.code_revision,
        )
        self._state[profile.profile_id] = state
        return payload

    def write_snapshot(self, request: Mapping[str, Any]) -> Path:
        payload = self.handle_board(request)
        snapshot_dir = self.run_root / 'snapshots'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        profile_id = _safe_filename(str(payload['league_profile']['profile_id']))
        state_sha = _safe_filename(str(payload['provenance']['state_sha256']))
        destination = snapshot_dir / f'{stamp}-{profile_id}-{state_sha}.json'
        fd, temporary_name = tempfile.mkstemp(
            dir=snapshot_dir, prefix=f'.{destination.name}.', suffix='.tmp'
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write('\n')
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FFBayes 2026 Draft Dashboard</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif}body{margin:0;background:#07111f;color:#e2e8f0}main{max-width:1500px;margin:0 auto;padding:24px}h1{margin:0 0 6px}.muted{color:#94a3b8}.blocked{background:#451a1a;border:1px solid #ef4444;padding:16px;border-radius:8px}.panel{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:16px;margin:14px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.control{display:flex;flex-direction:column;gap:5px}input,select,button{font:inherit;background:#111827;color:#f8fafc;border:1px solid #475569;border-radius:5px;padding:7px}button{cursor:pointer}button:hover{border-color:#38bdf8}.table-wrap{overflow:auto;max-height:65vh}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #334155;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#111827}.status-available{color:#86efac}.status-taken{color:#fca5a5}.status-mine{color:#93c5fd}.notice{color:#fbbf24}.hidden{display:none}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:3px 8px;margin-right:5px;font-size:12px}</style></head>
<body><main><h1>FFBayes 2026 Draft Dashboard</h1><p class="muted">Fresh public inputs, Python-calculated values, local loopback only.</p>
<section id="blocked" class="blocked hidden" role="alert"></section>
<section id="ready" class="hidden">
<div class="panel"><div class="controls"><label class="control">League<select id="league"></select></label><label class="control">Draft slot<input id="draft-slot" type="number" min="1"></label><label class="control">Current overall pick<input id="current-pick" type="number" min="1"></label><button id="recalculate">Recalculate board</button><button id="snapshot">Export snapshot</button></div><p id="league-settings" class="muted"></p><p id="status"></p></div>
<div class="panel"><strong>Provenance</strong><pre id="provenance" class="muted"></pre></div>
<div class="panel"><p class="muted">Use the row buttons to mark a player taken, yours, or queue them. Every change is recalculated by the Python service.</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Player</th><th>Pos</th><th>Proj</th><th>Replacement</th><th>VOR</th><th>Scarcity</th><th>ADP</th><th>Next pick</th><th>Action</th><th>State</th></tr></thead><tbody id="board"></tbody></table></div></div>
</section>
<script>
const stateByLeague = new Map(); let leagues = []; let currentPayload = null;
const $ = (id) => document.getElementById(id);
function showBlocked(message){ $('blocked').classList.remove('hidden'); $('ready').classList.add('hidden'); $('blocked').textContent = 'Dashboard blocked: ' + message; }
function currentState(){ const id=$('league').value; return stateByLeague.get(id) || {profile_id:id,draft_slot:null,current_pick:null,taken_ids:[],your_ids:[],queue_ids:[]}; }
function setSettings(){ const league=leagues.find(item=>item.profile_id===$('league').value); if(!league)return; $('draft-slot').max=league.team_count; $('current-pick').max=league.team_count*(Object.values(league.roster_slots).reduce((a,b)=>a+b,0)+league.bench_slots); $('league-settings').textContent=`${league.league_name} · ${league.team_count}-team ${league.draft_format} · ${league.scoring_label} · FLEX ${league.roster_slots.FLEX} · bench ${league.bench_slots} · IR ${league.ir_slots}`; const s=currentState(); $('draft-slot').value=s.draft_slot ?? ''; $('current-pick').value=s.current_pick ?? ''; }
function requestState(){ const state=currentState(); return {...state,draft_slot:$('draft-slot').value===''?null:Number($('draft-slot').value),current_pick:$('current-pick').value===''?null:Number($('current-pick').value)}; }
function renderBoard(){ const body=$('board'); body.replaceChildren(); if(!currentPayload)return; const statusClasses={available:'status-available',taken:'status-taken',mine:'status-mine'}; const textCell=(value)=>{ const cell=document.createElement('td'); cell.textContent=String(value); return cell; }; for(const row of currentPayload.decision_table.slice(0,100)){ const tr=document.createElement('tr'); tr.dataset.playerId=String(row.espn_id); const status=row.roster_status||'available'; tr.append(textCell(row.board_rank),textCell(row.name),textCell(row.position),textCell(Number(row.projected_points).toFixed(1)),textCell(Number(row.replacement_level).toFixed(1)),textCell(Number(row.vor).toFixed(1)),textCell(Number(row.scarcity).toFixed(1)),textCell(Number(row.adp).toFixed(1)),textCell(row.availability_next_pick==null?'Draft slot required':(Number(row.availability_next_pick)*100).toFixed(0)+'%')); const actionCell=document.createElement('td'); for(const [action,label] of [['taken','Taken'],['mine','Mine'],['queue','Queue']]){ if(actionCell.childNodes.length)actionCell.appendChild(document.createTextNode(' ')); const button=document.createElement('button'); button.type='button'; button.dataset.action=action; button.dataset.id=String(row.espn_id); button.textContent=label; actionCell.appendChild(button); } tr.appendChild(actionCell); const statusCell=textCell(`${status} · ${row.recommendation}`); if(statusClasses[status])statusCell.classList.add(statusClasses[status]); tr.appendChild(statusCell); body.appendChild(tr); } $('status').textContent=`Current pick ${currentPayload.current_pick ?? '—'} · next pick ${currentPayload.next_pick ?? '—'} · ${currentPayload.decision_table.length} validated players`; $('provenance').textContent=JSON.stringify(currentPayload.provenance,null,2); }
async function recalculate(){ const response=await fetch('/api/board',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(requestState())}); const data=await response.json(); if(!response.ok){$('status').textContent=data.error||'Board recalculation failed';$('status').className='notice';return;} currentPayload=data; const state=requestState(); stateByLeague.set(state.profile_id,state); renderBoard(); }
$('league').addEventListener('change',()=>{setSettings(); currentPayload=null; recalculate();}); $('recalculate').addEventListener('click',recalculate); $('snapshot').addEventListener('click',async()=>{const response=await fetch('/api/snapshot',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(requestState())}); const data=await response.json(); $('status').textContent=response.ok?'Snapshot written to '+data.path:(data.error||'Snapshot failed');}); $('board').addEventListener('click',(event)=>{const button=event.target.closest('button[data-action]');if(!button)return;const state=currentState();const id=Number(button.dataset.id);const action=button.dataset.action;for(const key of ['taken_ids','your_ids'])state[key]=state[key].filter(value=>value!==id);if(action==='taken')state.taken_ids.push(id);if(action==='mine')state.your_ids.push(id);if(action==='queue'){state.queue_ids=state.queue_ids.includes(id)?state.queue_ids.filter(value=>value!==id):[...state.queue_ids,id];}stateByLeague.set(state.profile_id,state);recalculate();});
(async()=>{try{const status=await (await fetch('/api/status')).json();if(status.status!=='ready'){showBlocked(status.error||'source validation failed');return;}leagues=(await (await fetch('/api/leagues')).json()).leagues||[];if(!leagues.length){showBlocked('no league profiles are available');return;}for(const league of leagues){stateByLeague.set(league.profile_id,{profile_id:league.profile_id,draft_slot:null,current_pick:null,taken_ids:[],your_ids:[],queue_ids:[]});const option=document.createElement('option');option.value=league.profile_id;option.textContent=league.league_name;$('league').appendChild(option);} $('ready').classList.remove('hidden');setSettings();await recalculate();}catch(error){showBlocked(error instanceof Error?error.message:String(error));}})();
</script></main></body></html>"""


def create_http_server(service: DashboardService, port: int = 0) -> ThreadingHTTPServer:
    """Create a loopback-only HTTP server for the dashboard service."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, value: object, content_type: str = 'application/json') -> None:
            if content_type == 'application/json':
                body = json.dumps(value, sort_keys=True).encode('utf-8')
            else:
                body = str(value).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', f'{content_type}; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == '/':
                self._send(200, _dashboard_html(), 'text/html')
            elif self.path == '/api/status':
                self._send(200, service.status_payload())
            elif self.path == '/api/leagues':
                self._send(200, service.leagues_payload())
            else:
                self._send(404, {'error': 'Not found'})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {'/api/board', '/api/snapshot'}:
                self._send(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise DashboardRequestError('Request body must be a JSON object')
                if self.path == '/api/board':
                    result = service.handle_board(request)
                else:
                    result = {'path': str(service.write_snapshot(request))}
                self._send(200, result)
            except (DashboardRequestError, DashboardBlockedError, OutputProvenanceError, ValueError) as exc:
                self._send(400, {'status': 'blocked', 'error': str(exc)})
            except Exception as exc:  # pragma: no cover - final HTTP safety net
                self._send(500, {'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}'})

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve_dashboard(
    service: DashboardService, *, port: int = 0, open_browser: bool = True
) -> int:
    server = create_http_server(service, port)
    address = server.server_address
    host = address[0].decode() if isinstance(address[0], bytes) else address[0]
    url = f'http://{host}:{address[1]}/'
    print(url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _load_profiles(paths: Sequence[Path]) -> tuple[list[LeagueProfile], str | None]:
    profiles: list[LeagueProfile] = []
    for path in paths:
        try:
            profiles.append(LeagueProfile.from_mapping(json.loads(path.read_text(encoding='utf-8'))))
        except (OSError, json.JSONDecodeError, LeagueProfileError) as exc:
            return profiles, f'Profile {path} is blocked: {exc}'
    return profiles, None


def _source_failure_details(exc: Exception) -> dict[str, Any]:
    """Describe a blocked fresh-input fetch without inventing a fallback."""
    if isinstance(exc, requests.RequestException):
        status = exc.response.status_code if exc.response is not None else None
        return {
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
        }
    if isinstance(exc, (ConnectionError, OSError)):
        return {
            'source': 'nflverse current-season roster release',
            'failure_mode': f'{type(exc).__name__}: {exc}',
            'transience': 'possibly transient network or upstream release failure',
            'pipeline_dependencies': [
                'current-player eligibility',
                'inactive-player exclusion',
                'recommendations',
            ],
            'alternate_source': 'none used; silent fallback is prohibited',
        }
    if isinstance(exc, SemanticInputError):
        return {
            'source': '2026 public player, projection, ADP, or roster inputs',
            'failure_mode': str(exc),
            'transience': 'unknown; source content is semantically inadequate',
            'pipeline_dependencies': [
                'replacement levels',
                'recommendations',
                'dashboard payload',
            ],
            'alternate_source': 'none used; silent fallback is prohibited',
        }
    return {
        'source': '2026 dashboard input orchestration',
        'failure_mode': f'{type(exc).__name__}: {exc}',
        'transience': 'unknown; manual diagnosis required',
        'pipeline_dependencies': [
            'player universe',
            'projections',
            'ADP',
            'replacement levels',
            'recommendations',
        ],
        'alternate_source': 'none used; silent fallback is prohibited',
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Start the local 2026 draft dashboard.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--profile', action='append', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.year != 2026:
        print(json.dumps({'status': 'blocked', 'error': 'Only --year 2026 is supported'}))
        return 2
    profile_paths = tuple(args.profile or default_profile_paths())
    profiles, profile_error = _load_profiles(profile_paths)
    blocked_error = profile_error
    blocked_details: dict[str, Any] = {}
    fresh_inputs: FreshInputs | None = None
    if blocked_error is None:
        try:
            fresh_inputs = load_fresh_inputs(args.year)
            for profile in profiles:
                build_draft_board(fresh_inputs.players, profile)
        except Exception as exc:
            blocked_error = f'{type(exc).__name__}: {exc}'
            blocked_details = _source_failure_details(exc)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_root = args.output_root or PROJECT_ROOT / 'runtime' / 'runs' / 'dashboard_2026'
    run_root = output_root / timestamp
    try:
        revision = _git_revision(PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    service = DashboardService(
        fresh_inputs,
        profiles,
        project_root=PROJECT_ROOT,
        run_root=run_root,
        code_revision=revision,
        blocked_error=blocked_error,
        blocked_details=blocked_details,
    )
    return serve_dashboard(service, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    raise SystemExit(main())
