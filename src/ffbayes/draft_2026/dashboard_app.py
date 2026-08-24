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

from ffbayes.draft_2026.draft_state import DraftState
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
        profile_ids = [profile.profile_id for profile in profiles]
        duplicates = sorted(
            {profile_id for profile_id in profile_ids if profile_ids.count(profile_id) > 1}
        )
        if duplicates:
            raise LeagueProfileError(
                f'duplicate profile_id values are not allowed: {duplicates}'
            )
        self.profiles = {profile.profile_id: profile for profile in profiles}
        self.project_root = project_root
        self.run_root = run_root
        self.code_revision = code_revision
        self.blocked_error = blocked_error
        self.blocked_details = dict(blocked_details or {})
        self._state: dict[str, DraftState] = {
            profile.profile_id: DraftState(draft_slot=profile.draft_slot)
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

    def _request_profile(self, request: Mapping[str, Any]) -> tuple[LeagueProfile, DraftState]:
        if self.blocked_error or self.fresh_inputs is None:
            raise DashboardBlockedError(self.blocked_error or 'Fresh inputs are unavailable')
        profile_id = request.get('profile_id')
        if not isinstance(profile_id, str) or profile_id not in self.profiles:
            raise DashboardRequestError('Unknown profile_id')
        if any(key in request for key in ('taken_ids', 'your_ids', 'queue_ids', 'actions')):
            raise DashboardRequestError(
                'Client-owned draft arrays are not accepted; use /api/action'
            )
        state = self._state[profile_id]
        if 'draft_slot' in request:
            raw_slot = request.get('draft_slot')
            if raw_slot is not None and (
                isinstance(raw_slot, bool) or not isinstance(raw_slot, int)
            ):
                raise DashboardRequestError('draft_slot must be an integer or null')
            if raw_slot is not None:
                try:
                    self.profiles[profile_id].validate_runtime_slot(raw_slot)
                except LeagueProfileError as exc:
                    raise DashboardRequestError(str(exc)) from exc
            state = replace(state, draft_slot=raw_slot)
        return replace(self.profiles[profile_id], draft_slot=state.draft_slot), state

    def _validate_pick(self, profile: LeagueProfile, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DashboardRequestError('current_pick must be an integer')
        if not 1 <= value <= profile.total_draft_picks() + 1:
            raise DashboardRequestError('current_pick is outside the configured draft')
        return value

    def _validate_player_id(self, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DashboardRequestError('player_id must be a positive integer')
        if value not in self._known_ids():
            raise DashboardRequestError(f'unknown player espn_id value: {value}')
        return value

    @staticmethod
    def _runtime_state(state: DraftState) -> dict[str, Any]:
        return {
            'draft_slot': state.draft_slot,
            'current_pick': state.current_pick,
            'taken_ids': list(state.taken_ids),
            'your_ids': list(state.your_ids),
            'queue_ids': list(state.queue_ids),
            'actions': [
                {
                    'pick': action.pick,
                    'player_id': action.player_id,
                    'disposition': action.disposition,
                }
                for action in state.actions
            ],
        }

    def _render(self, profile: LeagueProfile, state: DraftState) -> dict[str, Any]:
        fresh_inputs = self.fresh_inputs
        if fresh_inputs is None:
            raise DashboardBlockedError('Fresh inputs are unavailable')
        board = build_draft_board(
            fresh_inputs.players,
            profile,
            current_pick=state.current_pick,
            taken_ids=state.taken_ids,
            your_ids=state.your_ids,
        )
        board.attrs['runtime_state'] = self._runtime_state(state)
        payload = build_dashboard_payload(
            board,
            profile,
            source_manifest=fresh_inputs.source_manifest,
            roster_manifest=fresh_inputs.roster_manifest,
            coverage_report=fresh_inputs.coverage,
            code_revision=self.code_revision,
        )
        return payload

    def handle_board(self, request: Mapping[str, Any]) -> dict[str, Any]:
        profile, state = self._request_profile(request)
        if 'current_pick' in request:
            state = state.sync_clock(self._validate_pick(profile, request['current_pick']))
        payload = self._render(profile, state)
        self._state[profile.profile_id] = state
        return payload

    def handle_action(self, request: Mapping[str, Any]) -> dict[str, Any]:
        profile, state = self._request_profile(request)
        action = request.get('action')
        if not isinstance(action, Mapping):
            raise DashboardRequestError('action must be an object')
        action_type = action.get('type')
        if action_type == 'record':
            player_id = self._validate_player_id(action.get('player_id'))
            disposition = action.get('disposition')
            if disposition not in ('taken', 'mine'):
                raise DashboardRequestError("disposition must be 'taken' or 'mine'")
            state = state.record(player_id, disposition)
        elif action_type == 'queue':
            state = state.toggle_queue(self._validate_player_id(action.get('player_id')))
        elif action_type == 'undo':
            state = state.undo()
        elif action_type == 'sync':
            state = state.sync_clock(self._validate_pick(profile, action.get('current_pick')))
        else:
            raise DashboardRequestError('Unsupported action type')
        payload = self._render(profile, state)
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
<title>FFBayes 2026 Draft War Room</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,-apple-system,sans-serif}body{margin:0;background:#07111f;color:#e2e8f0}main{max-width:1600px;margin:auto;padding:24px}h1,h2,h3{margin:0 0 8px}.muted{color:#94a3b8}.blocked{background:#451a1a;border:1px solid #ef4444;padding:16px;border-radius:10px}.panel{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;margin:14px 0}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end}.control{display:flex;flex-direction:column;gap:5px}input,select,button{font:inherit;background:#111827;color:#f8fafc;border:1px solid #475569;border-radius:6px;padding:8px}button{cursor:pointer}button:hover{border-color:#38bdf8}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.wide{grid-column:span 2}.metric{font-size:1.6rem;font-weight:700}.notice{color:#fbbf24}.good{color:#86efac}.warn{color:#fca5a5}.table-wrap{overflow:auto;max-height:58vh}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #334155;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#111827}.status-taken{color:#fca5a5}.status-mine{color:#93c5fd}.bar{height:7px;background:#1e293b;border-radius:6px;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:#38bdf8}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:3px 8px;margin:2px;font-size:12px}.hidden{display:none}.list{display:grid;gap:7px}.list-item{padding:8px;border:1px solid #334155;border-radius:7px}.small{font-size:12px}.cliff{display:flex;align-items:center;gap:8px;margin:7px 0}.cliff strong{width:40px}.frontier{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}.frontier .list-item{background:#111c31}</style></head>
<body><main><h1>FFBayes 2026 Draft War Room</h1><p class="muted">One canonical Python board, live server-confirmed actions, fresh public 2026 inputs, loopback only.</p>
<section id="blocked" class="blocked hidden" role="alert"></section><section id="ready" class="hidden">
<div class="panel"><div class="controls"><label class="control">League<select id="league"></select></label><label class="control">Draft slot<input id="draft-slot" type="number" min="1"></label><label class="control">Current overall pick<input id="current-pick" type="number" min="1"></label><button id="sync-clock">Sync clock</button><button id="recalculate">Recalculate board</button><button id="undo">Undo last pick</button><button id="snapshot">Export snapshot</button></div><p id="league-settings" class="muted"></p><p id="status"></p></div>
<div class="grid"><section class="panel" id="recommendation-panel"><h2>Recommendation</h2><div id="recommendation"></div></section><section class="panel" id="roster-panel"><h2>My roster</h2><div id="roster" class="list"></div></section><section class="panel" id="queue-panel"><h2>Queue</h2><div id="queue" class="list"></div></section><section class="panel wide" id="timing-frontier"><h2>Timing frontier</h2><p class="muted small">Pick-now value, next-pick survival, and regret are calculated by Python.</p><div id="frontier" class="frontier"></div></section><section class="panel" id="positional-cliffs"><h2>Positional cliffs</h2><div id="cliffs"></div></section><section class="panel" id="comparative-explainer"><h2>Comparative explainer</h2><div id="comparative" class="list"></div></section><section class="panel wide" id="freshness-panel"><h2>Freshness and provenance</h2><pre id="freshness" class="muted small"></pre></section></div>
<section class="panel"><h2>Draft board</h2><p class="muted small">Taken and Mine advance the server clock once. Queue never advances it. Repeating or correcting a player is idempotent and Undo restores the consumed pick.</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Player</th><th>Pos</th><th>Proj</th><th>VOR</th><th>ADP</th><th>Survival</th><th>Action</th><th>State</th></tr></thead><tbody id="board"></tbody></table></div></section></section></main>
<script>
let leagues=[];let currentPayload=null;const $=id=>document.getElementById(id);const fmt=(v,d=1)=>v==null?'—':Number(v).toFixed(d);const esc=v=>String(v==null?'—':v);
function showBlocked(message){$('blocked').classList.remove('hidden');$('ready').classList.add('hidden');$('blocked').textContent='Dashboard blocked: '+message;}
function selectedLeague(){return leagues.find(x=>x.profile_id===$('league').value);}
function setSettings(){const league=selectedLeague();if(!league)return;$('draft-slot').max=league.team_count;$('current-pick').max=league.team_count*(Object.values(league.roster_slots).reduce((a,b)=>a+b,0)+league.bench_slots);$('league-settings').textContent=league.league_name+' · '+league.team_count+' teams · '+league.draft_format+' · '+league.scoring_label+' · FLEX '+league.roster_slots.FLEX+' · bench '+league.bench_slots+' · IR '+league.ir_slots;if(currentPayload){$('draft-slot').value=currentPayload.runtime_state.draft_slot??'';$('current-pick').value=currentPayload.current_pick??1;}}
function requestClock(){return{profile_id:$('league').value,draft_slot:$('draft-slot').value===''?null:Number($('draft-slot').value),current_pick:Number($('current-pick').value||1)};}
async function post(path,body){const response=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;}
function cell(value){const node=document.createElement('td');node.textContent=esc(value);return node;}
function item(row,extra){const node=document.createElement('div');node.className='list-item';node.textContent=row.name+' · '+row.position+' · VOR '+fmt(row.vor)+' · '+(extra||row.recommendation);return node;}
function renderAnalytics(){const a=currentPayload.analytics||{};const rec=$('recommendation');rec.replaceChildren();const primary=a.recommendation&&a.recommendation.primary;if(primary){const title=document.createElement('div');title.className='metric';title.textContent=primary.name;rec.appendChild(title);const why=document.createElement('p');why.textContent=primary.rationale+' VOR '+fmt(primary.vor)+' · survival '+fmt(primary.availability_next_pick*100,0)+'%';rec.appendChild(why);}else rec.textContent='No available players.';const fallback=document.createElement('div');for(const row of (a.recommendation?.fallbacks||[]))fallback.appendChild(item(row,'fallback · regret '+fmt(row.expected_regret)));rec.appendChild(fallback);$('roster').replaceChildren(...(a.roster||[]).map(row=>item(row,'mine')));$('queue').replaceChildren(...(a.queue||[]).map(row=>item(row,'queued')));$('frontier').replaceChildren(...(a.timing_frontier||[]).slice(0,12).map(row=>item(row,'lane '+row.lane+' · regret '+fmt(row.expected_regret))));const cliffs=$('cliffs');cliffs.replaceChildren();for(const row of (a.positional_cliffs||[])){const wrap=document.createElement('div');wrap.className='cliff';const label=document.createElement('strong');label.textContent=row.position;const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=Math.min(100,Number(row.strongest_cliff||0))+'%';bar.appendChild(fill);const text=document.createElement('span');text.className='small';text.textContent='cliff '+fmt(row.strongest_cliff)+' after '+esc(row.cliff_after_rank);wrap.append(label,bar,text);cliffs.appendChild(wrap);}const comparative=$('comparative');comparative.replaceChildren();for(const row of (a.comparative||[]).filter(x=>Math.abs(Number(x.rank_gap))>=3).slice(0,8)){comparative.appendChild(item(row,'model '+fmt(row.model_rank,0)+' vs market '+fmt(row.market_rank,0)+' (gap '+fmt(row.rank_gap,0)+')'));}}
function renderBoard(){const body=$('board');body.replaceChildren();if(!currentPayload)return;const queued=new Set(currentPayload.runtime_state.queue_ids||[]);for(const row of currentPayload.decision_table.slice(0,100)){const tr=document.createElement('tr');tr.dataset.playerId=String(row.espn_id);const status=row.roster_status||'available';tr.append(cell(row.board_rank),cell(row.name),cell(row.position),cell(fmt(row.projected_points)),cell(fmt(row.vor)),cell(fmt(row.adp)),cell(row.availability_next_pick==null?'slot required':fmt(Number(row.availability_next_pick)*100,0)+'%'));const actions=document.createElement('td');for(const [type,label] of [['taken','Taken'],['mine','Mine'],['queue',queued.has(row.espn_id)?'Unqueue':'Queue']]){const button=document.createElement('button');button.type='button';button.dataset.type=type;button.dataset.action=type;button.dataset.id=String(row.espn_id);button.textContent=label;actions.appendChild(button);if(type!=='queue')actions.appendChild(document.createTextNode(' '));}tr.appendChild(actions);const state=cell((queued.has(row.espn_id)?'queued · ':'')+status+' · '+row.recommendation);state.className=status==='mine'?'status-mine':status==='taken'?'status-taken':'';tr.appendChild(state);body.appendChild(tr);}$('status').textContent='Current pick '+esc(currentPayload.current_pick)+' · next pick '+esc(currentPayload.next_pick)+' · '+currentPayload.decision_table.length+' validated players';$('freshness').textContent=JSON.stringify({generated_at:currentPayload.generated_at,coverage:currentPayload.coverage_report,provenance:currentPayload.provenance},null,2);renderAnalytics();}
async function refresh(){try{currentPayload=await post('/api/board',requestClock());setSettings();renderBoard();$('status').className='';}catch(error){$('status').textContent=error.message;$('status').className='notice';}}
$('league').addEventListener('change',()=>{currentPayload=null;setSettings();refresh();});$('sync-clock').addEventListener('click',refresh);$('recalculate').addEventListener('click',refresh);$('undo').addEventListener('click',async()=>{try{currentPayload=await post('/api/action',{profile_id:$('league').value,action:{type:'undo'}});setSettings();renderBoard();}catch(error){$('status').textContent=error.message;$('status').className='notice';}});$('snapshot').addEventListener('click',async()=>{try{const result=await post('/api/snapshot',requestClock());$('status').textContent='Snapshot written to '+result.path;}catch(error){$('status').textContent=error.message;$('status').className='notice';}});$('board').addEventListener('click',async event=>{const button=event.target.closest('button[data-type]');if(!button)return;const type=button.dataset.type;try{currentPayload=await post('/api/action',{profile_id:$('league').value,action:{type:type==='queue'?'queue':'record',player_id:Number(button.dataset.id),disposition:type==='mine'?'mine':'taken'}});setSettings();renderBoard();}catch(error){$('status').textContent=error.message;$('status').className='notice';}});
(async()=>{try{const status=await(await fetch('/api/status')).json();if(status.status!=='ready'){showBlocked(status.error||'source validation failed');return;}leagues=(await(await fetch('/api/leagues')).json()).leagues||[];if(!leagues.length){showBlocked('no league profiles are available');return;}for(const league of leagues){const option=document.createElement('option');option.value=league.profile_id;option.textContent=league.league_name;$('league').appendChild(option);} $('ready').classList.remove('hidden');await refresh();}catch(error){showBlocked(error instanceof Error?error.message:String(error));}})();
</script></body></html>"""


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
            if self.path not in {'/api/board', '/api/action', '/api/snapshot'}:
                self._send(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise DashboardRequestError('Request body must be a JSON object')
                if self.path == '/api/board':
                    result = service.handle_board(request)
                elif self.path == '/api/action':
                    result = service.handle_action(request)
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
    profile_ids = [profile.profile_id for profile in profiles]
    duplicates = sorted(
        {profile_id for profile_id in profile_ids if profile_ids.count(profile_id) > 1}
    )
    if duplicates:
        return profiles, f'duplicate profile_id values are not allowed: {duplicates}'
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
        mismatched_profiles = [
            profile
            for profile in profiles
            if profile.season != args.year
        ]
        if mismatched_profiles:
            blocked_error = (
                f'Profile season does not match requested year {args.year}: '
                + ', '.join(
                    f'{profile.profile_id} has season {profile.season}'
                    for profile in mismatched_profiles
                )
            )
            blocked_details = {
                'source': 'local league profile configuration',
                'failure_mode': blocked_error,
                'external_dependency': False,
                'pipeline_dependencies': ['dashboard readiness', 'board requests'],
                'resolution': 'use a profile whose season matches the requested year',
            }
        else:
            try:
                fresh_inputs = load_fresh_inputs(args.year)
            except Exception as exc:
                blocked_error = f'{type(exc).__name__}: {exc}'
                blocked_details = _source_failure_details(exc)
            else:
                try:
                    for profile in profiles:
                        build_draft_board(fresh_inputs.players, profile)
                except SemanticInputError as exc:
                    blocked_error = f'{type(exc).__name__}: {exc}'
                    blocked_details = {
                        'source': 'local league profile scoring configuration',
                        'failure_mode': str(exc),
                        'external_dependency': False,
                        'pipeline_dependencies': [
                            'replacement levels',
                            'recommendations',
                            'dashboard payload',
                        ],
                        'resolution': (
                            'supply scoring rules compatible with the current '
                            'public projection statistic IDs'
                        ),
                    }
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_root = args.output_root or PROJECT_ROOT / 'runtime' / 'runs' / 'dashboard_2026'
    run_root = output_root / timestamp
    try:
        revision = _git_revision(PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    service = DashboardService(
        fresh_inputs,
        profiles if profile_error is None else [],
        project_root=PROJECT_ROOT,
        run_root=run_root,
        code_revision=revision,
        blocked_error=blocked_error,
        blocked_details=blocked_details,
    )
    return serve_dashboard(service, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    raise SystemExit(main())
