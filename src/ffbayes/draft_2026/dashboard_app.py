"""One-command, loopback-only 2026 draft dashboard service."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
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
        self._state_lock = threading.RLock()
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
        with self._state_lock:
            profile, state = self._request_profile(request)
            if 'current_pick' in request:
                state = state.sync_clock(self._validate_pick(profile, request['current_pick']))
            payload = self._render(profile, state)
            self._state[profile.profile_id] = state
            return payload

    def handle_action(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._state_lock:
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
:root{color-scheme:dark;font-family:'Avenir Next','DM Sans',Inter,system-ui,-apple-system,sans-serif;--bg:#07111f;--panel:#0d192b;--panel-strong:#111f34;--line:#263b56;--muted:#8fa3ba;--text:#edf5ff;--accent:#55d6be;--accent-2:#65a7ff;--danger:#ff9b9b}*{box-sizing:border-box}body{margin:0;min-width:320px;background:radial-gradient(circle at 80% -10%,#183657 0,#0b1a2c 36%,var(--bg) 72%);color:var(--text)}main{max-width:1540px;margin:auto;padding:30px 34px 56px}h1,h2,h3{font-family:'Avenir Next','DM Sans',Inter,system-ui,sans-serif;margin:0 0 8px;letter-spacing:-.02em}h1{font-size:clamp(1.9rem,3vw,2.7rem)}h2{font-size:1.1rem}.muted{color:var(--muted)}.eyebrow{font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 7px}.blocked{background:#451a1a;border:1px solid #ef4444;padding:16px;border-radius:12px}.app-header{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:24px}.app-header>div:first-child{max-width:780px}.app-header p.muted{margin:7px 0 0;font-size:.94rem}.clock-hero{min-width:210px;padding:15px 18px;border:1px solid #2b5770;border-radius:14px;background:linear-gradient(135deg,#0d2635,#10243d);box-shadow:0 12px 30px #02081266}.clock-hero strong{display:block;font-family:'Avenir Next','DM Sans',Inter,system-ui,sans-serif;font-size:1.9rem;line-height:1.1}.clock-hero span:last-child{display:block;margin-top:5px;font-size:.82rem}.panel{background:linear-gradient(160deg,#102039e8,#0b1729f2);border:1px solid var(--line);border-radius:14px;padding:18px;margin:0;box-shadow:0 10px 26px #0208122b}.control-panel{margin-bottom:18px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.control{display:flex;flex-direction:column;gap:6px;font-size:.78rem;color:#b9c9db;font-weight:600}.control select,.control input{margin-top:0}.control:first-child{min-width:220px}.control:nth-child(2),.control:nth-child(3){min-width:145px}input,select,button{font:inherit;background:#0a1526;color:var(--text);border:1px solid #3a5270;border-radius:8px;padding:9px 11px;min-height:40px}input:focus,select:focus,button:focus{outline:2px solid #55d6be66;outline-offset:1px}button{cursor:pointer;font-weight:600;transition:border-color .15s,background .15s,transform .15s}button:hover{border-color:var(--accent-2);background:#142b45;transform:translateY(-1px)}#sync-clock,#recalculate{background:#165d61;border-color:#2cbba8}#sync-clock:hover,#recalculate:hover{background:#1c7b7b}#undo,#snapshot{background:#101c30}.toolbar-note{margin:12px 0 0}.status-line{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin:12px 0 0}.status-line #status{margin:0;font-size:.9rem}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:16px}.primary-grid{grid-template-columns:minmax(0,1.45fr) minmax(220px,.8fr) minmax(220px,.8fr)}.wide{grid-column:span 2}.feature{border-color:#2c6975;background:linear-gradient(145deg,#102d3a,#102039)}.metric{font-family:'Avenir Next','DM Sans',Inter,system-ui,sans-serif;font-size:clamp(1.45rem,2.5vw,2.05rem);font-weight:700;line-height:1.15;color:#fff}.recommendation-copy{max-width:650px;color:#c0d1e3;margin:9px 0 16px}.recommendation-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);font-weight:700}.panel-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.health-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border:1px solid #2c876e;background:#123d3b;color:#91f0d0;border-radius:999px;font-size:.75rem;font-weight:700;white-space:nowrap}.health-badge::before{content:'';width:7px;height:7px;border-radius:50%;background:currentColor}.health-badge.warn{border-color:#8f4c4c;background:#3b1e27;color:var(--danger)}.health-summary{margin:5px 0 0}.health-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:16px 0}.health-metric{padding:10px 11px;border:1px solid #29415d;border-radius:9px;background:#0b1729}.health-metric strong{display:block;color:#f5fbff;font-size:1.05rem}.health-metric span{display:block;color:var(--muted);font-size:.72rem;margin-top:3px}.source-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.source-chip{border:1px solid #35506f;border-radius:999px;padding:5px 9px;color:#c3d4e6;font-size:.72rem;background:#0a1728}.provenance-details{margin-top:14px;border-top:1px solid #243a55;padding-top:11px}.provenance-details summary{cursor:pointer;color:#99b6d3;font-size:.8rem}.provenance-details[open] summary{color:var(--accent)}pre{white-space:pre-wrap;overflow:auto;max-height:360px;margin:11px 0 0}.notice{color:#fbbf24}.good{color:#86efac}.warn{color:var(--danger)}.table-panel{margin-top:0}.table-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px}.table-wrap{overflow:auto;max-height:560px;margin-top:12px;border:1px solid #243a55;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid #20344d;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#0d1d31;color:#a9c1d9;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;z-index:1}tbody tr:nth-child(even){background:#0b192b99}tbody tr:hover{background:#17324a}.status-taken{color:var(--danger)}.status-mine{color:#9cc4ff}.bar{height:7px;background:#1e3047;border-radius:6px;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent-2),var(--accent))}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:3px 8px;margin:2px;font-size:12px}.hidden{display:none}.list{display:grid;gap:8px}.list-item{padding:9px 10px;border:1px solid #29415d;border-radius:9px;background:#0b1729}.small{font-size:12px}.cliff{display:flex;align-items:center;gap:8px;margin:9px 0}.cliff strong{width:40px}.frontier{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}.frontier .list-item{background:#0d2035}#freshness-panel{align-self:start}@media(max-width:980px){main{padding:22px 18px 40px}.app-header{align-items:flex-start;flex-direction:column}.clock-hero{width:100%;min-width:0}.grid,.primary-grid{grid-template-columns:1fr}.wide{grid-column:auto}.health-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.health-grid{grid-template-columns:1fr}.controls>*{width:100%}.control:first-child,.control:nth-child(2),.control:nth-child(3){min-width:0}.controls button{width:100%}}</style></head>
<style>.full-panel{margin-bottom:16px}.analysis-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.analysis-heading .eyebrow{margin-bottom:6px}.legend{display:flex;flex-wrap:wrap;gap:10px 16px;color:#a9bfd5;font-size:.75rem;white-space:nowrap}.legend span{display:inline-flex;align-items:center;gap:6px}.legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent-2)}.legend-dot.vor,.rank-marker.model{background:var(--accent-2)}.legend-dot.survival,.rank-marker.market{background:var(--accent)}.legend-dot.regret{background:#f3a35c}.frontier-chart{margin-top:18px}.frontier-header,.frontier-row{display:grid;grid-template-columns:minmax(230px,1.3fr) repeat(3,minmax(180px,1fr));gap:16px;align-items:center}.frontier-header{padding:0 10px 8px;color:#8fa8c2;font-size:.68rem;letter-spacing:.09em;text-transform:uppercase}.frontier-row{padding:11px 10px;border-top:1px solid #203850}.frontier-row:hover{background:#112a3e}.frontier-player{min-width:0}.frontier-name{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:700;color:#f5fbff}.frontier-subline{display:flex;align-items:center;gap:7px;margin-top:4px;color:var(--muted);font-size:.72rem}.metric-cell{min-width:0}.metric-caption{display:none}.metric-line{display:flex;align-items:center;gap:9px}.metric-track{height:9px;flex:1;min-width:60px;border-radius:99px;background:#1b3047;overflow:hidden}.metric-track i{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#4d8dff,#65a7ff)}.metric-track.survival i{background:linear-gradient(90deg,#36b79f,#65e0be)}.metric-track.regret i{background:linear-gradient(90deg,#c87745,#f3a35c)}.metric-track.unavailable{border:1px dashed #46627b;background:transparent}.metric-value{min-width:43px;text-align:right;color:#dce9f7;font-size:.78rem;font-variant-numeric:tabular-nums}.lane-badge{display:inline-flex;padding:3px 7px;border-radius:999px;border:1px solid #35516d;font-size:.67rem;color:#b9cee1}.lane-badge.pick-now{border-color:#3d7895;color:#a8d7ff;background:#123149}.lane-badge.wait{border-color:#397b70;color:#a4edda;background:#123633}.lane-badge.unavailable{border-color:#6c5360;color:#e6b5bd;background:#30212b}.cliff-chart{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px;margin-top:16px}.cliff-card{padding:12px;border:1px solid #29445e;border-radius:10px;background:#0b192b}.cliff-card header{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.cliff-card strong{font-size:1rem}.cliff-card header span{color:#91a9c0;font-size:.7rem}.cliff-bar{height:9px;margin:16px 0 10px;border-radius:99px;background:#1b3047;overflow:hidden}.cliff-bar i{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#4d8dff,#55d6be)}.cliff-card footer{color:#a9bfd5;font-size:.72rem}.comparative-summary{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 3px}.summary-chip{padding:6px 10px;border:1px solid #29445e;border-radius:999px;background:#0b192b;color:#b9cee1;font-size:.74rem}.summary-chip strong{color:#f5fbff}.summary-chip.model{border-color:#38678c}.summary-chip.market{border-color:#397b70}.rank-axis{display:flex;justify-content:space-between;margin:17px 12px 5px 285px;color:#7f99b2;font-size:.67rem;text-transform:uppercase;letter-spacing:.06em}.comparative-chart{display:grid}.comparative-row{display:grid;grid-template-columns:minmax(230px,265px) minmax(0,1fr);gap:20px;align-items:center;padding:12px 10px;border-top:1px solid #203850}.comparative-row:hover{background:#112a3e}.comparative-player strong{display:block;color:#f5fbff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.comparative-player span{display:block;margin-top:5px;color:#a9bfd5;font-size:.73rem}.gap-badge{display:inline-block!important;width:max-content;margin-top:5px;padding:2px 6px;border-radius:999px;border:1px solid #3a607a;color:#abd7ff!important;font-size:.65rem!important}.gap-badge.market-early{border-color:#397b70;color:#a4edda!important}.rank-track{position:relative;height:24px;border-radius:8px;background:repeating-linear-gradient(90deg,#193048 0,#193048 1px,transparent 1px,transparent 10%);border-bottom:1px solid #36516b}.rank-marker{position:absolute;top:50%;width:12px;height:12px;border:2px solid #07111f;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px #ffffff18}.rank-marker::after{content:attr(aria-label);position:absolute;top:16px;left:50%;transform:translateX(-50%);font-size:.63rem;color:#c1d4e6;white-space:nowrap}.rank-values{display:flex;gap:18px;margin-top:14px;color:#91a9c0;font-size:.72rem}.rank-values b{font-weight:700;color:#b9cee1}.rank-values .model-value b{color:#83b8ff}.rank-values .market-value b{color:#7de2c2}@media(max-width:980px){.frontier-header{display:none}.frontier-row{grid-template-columns:1fr;gap:8px}.metric-caption{display:block;margin-bottom:4px;color:#8fa8c2;font-size:.68rem;text-transform:uppercase;letter-spacing:.07em}.cliff-chart{grid-template-columns:repeat(3,minmax(130px,1fr))}.rank-axis{margin-left:220px}.comparative-row{grid-template-columns:minmax(180px,220px) minmax(0,1fr)}}@media(max-width:620px){.analysis-heading{display:block}.legend{margin-top:12px}.cliff-chart{grid-template-columns:repeat(2,minmax(130px,1fr))}.rank-axis{margin-left:0}.comparative-row{grid-template-columns:1fr;gap:8px}.rank-values{margin-top:12px}}</style>
<body><main><header class="app-header"><div><p class="eyebrow">Live draft assistant</p><h1>FFBayes 2026 Draft War Room</h1><p class="muted">A live, server-confirmed board for making the next pick with context.</p></div><div class="clock-hero"><p class="eyebrow">On the clock</p><strong id="current-pick-hero">Pick 1</strong><span id="next-pick-hero" class="muted">Next user pick —</span></div></header>
<section id="blocked" class="blocked hidden" role="alert"></section><section id="ready" class="hidden">
<div class="panel control-panel"><div class="controls"><label class="control">League<select id="league"></select></label><label class="control">Draft slot<input id="draft-slot" type="number" min="1"></label><label class="control">Current overall pick<input id="current-pick" type="number" min="1"></label><button id="sync-clock">Sync clock</button><button id="recalculate">Recalculate board</button><button id="undo">Undo last pick</button><button id="snapshot">Export snapshot</button></div><div class="status-line"><p id="league-settings" class="muted toolbar-note"></p><p id="status" class="toolbar-note"></p></div></div>
<div class="grid primary-grid"><section class="panel feature" id="recommendation-panel"><p class="recommendation-label">Best available now</p><h2>Recommendation</h2><div id="recommendation"></div></section><section class="panel" id="roster-panel"><h2>My roster</h2><div id="roster" class="list"></div><p id="roster-counts" class="muted small"></p></section><section class="panel" id="queue-panel"><h2>Queue</h2><div id="queue" class="list"></div></section></div>
<section class="panel table-panel" id="board-panel"><div class="table-heading"><div><h2>Draft board</h2><p class="muted small">Use Taken or Mine to advance the server clock. Queue is a watchlist only.</p></div><span class="health-badge">Live board</span></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Player</th><th>Pos</th><th>Proj</th><th>VOR</th><th>ADP</th><th>Survival</th><th>Action</th><th>State</th></tr></thead><tbody id="board"></tbody></table></div></section>
<section class="panel full-panel" id="timing-frontier"><div class="analysis-heading"><div><p class="eyebrow">Decision timing</p><h2>Timing frontier</h2><p class="muted small">The visual tradeoff behind “draft now” versus “can wait.”</p></div><div class="legend"><span><i class="legend-dot vor"></i>VOR</span><span><i class="legend-dot survival"></i>Survival</span><span><i class="legend-dot regret"></i>Regret</span></div></div><div class="frontier-chart"><div class="frontier-header"><span>Player</span><span>Pick-now value</span><span>Next-pick survival</span><span>Expected regret</span></div><div id="frontier"></div></div></section>
<section class="panel full-panel" id="positional-cliffs"><div class="analysis-heading"><div><p class="eyebrow">Roster construction</p><h2>Positional cliffs</h2><p class="muted small">Where the next player at a position costs the most value.</p></div></div><div id="cliffs" class="cliff-chart"></div></section>
<section class="panel full-panel" id="comparative-explainer"><div class="analysis-heading"><div><p class="eyebrow">Market signal</p><h2>Market vs model</h2><p class="muted small">Lower rank is earlier. The two markers show where the model and market place each outlier.</p></div><div class="legend"><span><i class="legend-dot model"></i>Model</span><span><i class="legend-dot market"></i>Market</span></div></div><div id="comparative-summary" class="comparative-summary"></div><div class="rank-axis"><span>1</span><span>Earlier</span><span>Later</span><span>50+</span></div><div id="comparative" class="comparative-chart"></div></section>
<section class="panel full-panel" id="freshness-panel"><div class="panel-heading"><div><p class="eyebrow">Input status</p><h2>Data health</h2><p id="freshness" class="muted small health-summary"></p></div><span id="freshness-status" class="health-badge">Validated</span></div><div id="freshness-metrics" class="health-grid"></div><div id="source-list" class="source-list"></div><details id="provenance-details" class="provenance-details"><summary>View technical provenance</summary><pre id="provenance" class="muted small"></pre></details></section></section></main>
<script>
let leagues=[];let currentPayload=null;const $=id=>document.getElementById(id);const fmt=(v,d=1)=>v==null?'—':Number(v).toFixed(d);const esc=v=>String(v==null?'—':v);
function showBlocked(message){$('blocked').classList.remove('hidden');$('ready').classList.add('hidden');$('blocked').textContent='Dashboard blocked: '+message;}
function selectedLeague(){return leagues.find(x=>x.profile_id===$('league').value);}
function setSettings(){const league=selectedLeague();if(!league)return;$('draft-slot').max=league.team_count;$('current-pick').max=league.team_count*(Object.values(league.roster_slots).reduce((a,b)=>a+b,0)+league.bench_slots);$('league-settings').textContent=league.league_name+' · '+league.team_count+' teams · '+league.draft_format+' · '+league.scoring_label+' · FLEX '+league.roster_slots.FLEX+' · bench '+league.bench_slots+' · IR '+league.ir_slots;if(currentPayload){$('draft-slot').value=currentPayload.runtime_state.draft_slot??'';$('current-pick').value=currentPayload.current_pick??1;}}
function requestClock(){return{profile_id:$('league').value,draft_slot:$('draft-slot').value===''?null:Number($('draft-slot').value),current_pick:Number($('current-pick').value||1)};}
async function post(path,body){const response=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;}
function cell(value){const node=document.createElement('td');node.textContent=esc(value);return node;}
function item(row,extra){const node=document.createElement('div');node.className='list-item';node.textContent=row.name+' · '+row.position+' · score '+fmt(row.vor??row.contextual_score)+' · '+(extra||row.recommendation);return node;}
function metric(label,value){const node=document.createElement('div');node.className='health-metric';const strong=document.createElement('strong');strong.textContent=value;const text=document.createElement('span');text.textContent=label;node.append(strong,text);return node;}
function renderHealth(){const report=currentPayload.coverage_report||{};const provenance=currentPayload.provenance||{};const rows=Number(report.rows||currentPayload.decision_table?.length||0);const market=report.market_coverage==null?'—':fmt(Number(report.market_coverage)*100,0)+'%';const projectionCounts=Object.values(report.projection_counts||{}).map(Number);const projectionRows=projectionCounts.length?projectionCounts.reduce((a,b)=>a+b,0):rows;const date=currentPayload.generated_at?new Date(currentPayload.generated_at).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'}):'—';$('freshness').textContent='Fresh '+rows+'-player board · generated '+date+' · season '+esc(report.season||'2026');const passed=report.status==='passed';$('freshness-status').textContent=passed?'Validated':'Blocked';$('freshness-status').className='health-badge '+(passed?'':'warn');const metrics=$('freshness-metrics');metrics.replaceChildren(metric('Validated players',rows),metric('Market / ADP coverage',market),metric('Projected player rows',projectionRows),metric('Source season',report.season||'—'));const sources=$('source-list');sources.replaceChildren();const manifests=provenance.source_manifests||[];const defaults=['Player pool','Roster status'];for(const [index,manifest] of manifests.entries()){const chip=document.createElement('span');chip.className='source-chip';const name=String(manifest.source||defaults[index]||'Public input').replaceAll('_',' ');chip.textContent=name+(manifest.cache_mode?' · '+manifest.cache_mode:'');sources.appendChild(chip);}if(!manifests.length){const chip=document.createElement('span');chip.className='source-chip';chip.textContent='Public inputs';sources.appendChild(chip);}$('provenance').textContent=JSON.stringify(provenance,null,2);}
function laneLabel(lane){return lane==='pick_now'?'Draft now':lane==='wait'?'Can wait':'Slot required';}
function metricCell(label,value,max,kind,formatter,unavailable=false){const cellNode=document.createElement('div');cellNode.className='metric-cell';const caption=document.createElement('span');caption.className='metric-caption';caption.textContent=label;const line=document.createElement('div');line.className='metric-line';const track=document.createElement('div');track.className='metric-track '+kind;const fill=document.createElement('i');if(unavailable){track.classList.add('unavailable');}else{fill.style.width=Math.max(0,Math.min(100,(Number(value)/max)*100))+'%';track.appendChild(fill);}const valueNode=document.createElement('span');valueNode.className='metric-value';valueNode.textContent=unavailable?'slot required':formatter(Number(value));line.append(track,valueNode);cellNode.append(caption,line);return cellNode;}
function renderTiming(rows){const frontier=$('frontier');frontier.replaceChildren();const visible=rows.slice(0,12);const maxVor=Math.max(1,...visible.map(row=>Number(row.vor)||0));const maxRegret=Math.max(1,...visible.map(row=>Number(row.expected_regret)||0));for(const row of visible){const wrap=document.createElement('div');wrap.className='frontier-row';const player=document.createElement('div');player.className='frontier-player';const name=document.createElement('span');name.className='frontier-name';name.textContent='#'+esc(row.board_rank)+' · '+row.name;const subline=document.createElement('span');subline.className='frontier-subline';const pos=document.createElement('span');pos.textContent=row.position;const badge=document.createElement('span');badge.className='lane-badge '+String(row.lane||'unavailable').replace('_','-');badge.textContent=laneLabel(row.lane);subline.append(pos,badge);player.append(name,subline);const survivalMissing=row.availability_next_pick==null;wrap.append(player,metricCell('Pick-now value',row.vor,maxVor,'vor',value=>fmt(value),false),metricCell('Next-pick survival',row.availability_next_pick,1,'survival',value=>fmt(value*100,0)+'%',survivalMissing),metricCell('Expected regret',row.expected_regret,maxRegret,'regret',value=>fmt(value)));frontier.appendChild(wrap);}}
function renderCliffs(rows){const cliffs=$('cliffs');cliffs.replaceChildren();const maxCliff=Math.max(1,...rows.map(row=>Number(row.strongest_cliff)||0));for(const row of rows){const card=document.createElement('article');card.className='cliff-card';const header=document.createElement('header');const position=document.createElement('strong');position.textContent=row.position;const available=document.createElement('span');available.textContent=esc(row.players_available)+' available';header.append(position,available);const bar=document.createElement('div');bar.className='cliff-bar';const fill=document.createElement('i');fill.style.width=Math.max(0,Math.min(100,(Number(row.strongest_cliff||0)/maxCliff)*100))+'%';bar.appendChild(fill);const footer=document.createElement('footer');footer.textContent=fmt(row.strongest_cliff)+' pt drop after '+esc(row.cliff_after_rank);card.append(header,bar,footer);cliffs.appendChild(card);}}
function summaryChip(label,count,kind){const chip=document.createElement('span');chip.className='summary-chip '+kind;const strong=document.createElement('strong');strong.textContent=count;chip.append(strong,document.createTextNode(' '+label));return chip;}
function renderComparative(rows){const comparative=$('comparative');comparative.replaceChildren();const outliers=rows.filter(row=>Math.abs(Number(row.rank_gap))>=3).slice(0,8);const modelEarly=outliers.filter(row=>Number(row.rank_gap)>0).length;const marketEarly=outliers.filter(row=>Number(row.rank_gap)<0).length;const summary=$('comparative-summary');summary.replaceChildren(summaryChip('model-early outliers',modelEarly,'model'),summaryChip('market-early outliers',marketEarly,'market'));const maxRank=Math.max(50,...outliers.flatMap(row=>[Number(row.model_rank)||0,Number(row.market_rank)||0]));for(const row of outliers){const wrap=document.createElement('div');wrap.className='comparative-row';const player=document.createElement('div');player.className='comparative-player';const name=document.createElement('strong');name.textContent=row.name;const info=document.createElement('span');info.textContent=row.position;const gap=document.createElement('span');const gapValue=Number(row.rank_gap);gap.className='gap-badge '+(gapValue<0?'market-early':'');gap.textContent=gapValue>0?'Model '+fmt(gapValue,0)+' earlier':'Market '+fmt(Math.abs(gapValue),0)+' earlier';info.append(document.createTextNode(' · '),gap);player.append(name,info);const visual=document.createElement('div');const track=document.createElement('div');track.className='rank-track';const modelMarker=document.createElement('span');modelMarker.className='rank-marker model';modelMarker.style.left=((Number(row.model_rank)-1)/(maxRank-1)*100)+'%';modelMarker.setAttribute('aria-label','Model '+fmt(row.model_rank,0));modelMarker.title='Model rank '+fmt(row.model_rank,0);const marketMarker=document.createElement('span');marketMarker.className='rank-marker market';marketMarker.style.left=((Number(row.market_rank)-1)/(maxRank-1)*100)+'%';marketMarker.setAttribute('aria-label','Market '+fmt(row.market_rank,0));marketMarker.title='Market rank '+fmt(row.market_rank,0);track.append(modelMarker,marketMarker);const values=document.createElement('div');values.className='rank-values';const modelValue=document.createElement('span');modelValue.className='model-value';const modelStrong=document.createElement('b');modelStrong.textContent=fmt(row.model_rank,0);modelValue.append(document.createTextNode('Model '),modelStrong);const marketValue=document.createElement('span');marketValue.className='market-value';const marketStrong=document.createElement('b');marketStrong.textContent=fmt(row.market_rank,0);marketValue.append(document.createTextNode('Market '),marketStrong);values.append(modelValue,marketValue);visual.append(track,values);wrap.append(player,visual);comparative.appendChild(wrap);}}
function renderAnalytics(){const a=currentPayload.analytics||{};const rec=$('recommendation');rec.replaceChildren();const primary=a.recommendation&&a.recommendation.primary;if(primary){const title=document.createElement('div');title.className='metric';title.textContent=primary.name;rec.appendChild(title);const why=document.createElement('p');const survival=primary.availability_next_pick==null?'slot required':fmt(primary.availability_next_pick*100,0)+'%';why.textContent=primary.rationale+' VOR '+fmt(primary.vor)+' · survival '+survival;rec.appendChild(why);}else rec.textContent='No available players.';const fallback=document.createElement('div');for(const row of (a.recommendation?.fallbacks||[]))fallback.appendChild(item(row,'fallback · regret '+fmt(row.expected_regret)));rec.appendChild(fallback);$('roster').replaceChildren(...(a.roster||[]).map(row=>item(row,'mine')));$('roster-counts').textContent=Object.entries(a.roster_position_counts||{}).map(([position,count])=>position+' '+count).join(' · ')||'No players recorded';$('queue').replaceChildren(...(a.queue||[]).map(row=>item(row,'queued')));renderTiming(a.timing_frontier||[]);renderCliffs(a.positional_cliffs||[]);renderComparative(a.comparative||[]);}
function renderBoard(){const body=$('board');body.replaceChildren();if(!currentPayload)return;const queued=new Set(currentPayload.runtime_state.queue_ids||[]);for(const row of currentPayload.decision_table.slice(0,100)){const tr=document.createElement('tr');tr.dataset.playerId=String(row.espn_id);const status=row.roster_status||'available';tr.append(cell(row.board_rank),cell(row.name),cell(row.position),cell(fmt(row.projected_points)),cell(fmt(row.vor)),cell(fmt(row.adp)),cell(row.availability_next_pick==null?'slot required':fmt(Number(row.availability_next_pick)*100,0)+'%'));const actions=document.createElement('td');for(const [type,label] of [['taken','Taken'],['mine','Mine'],['queue',queued.has(row.espn_id)?'Unqueue':'Queue']]){const button=document.createElement('button');button.type='button';button.dataset.type=type;button.dataset.action=type;button.dataset.id=String(row.espn_id);button.textContent=label;actions.appendChild(button);if(type!=='queue')actions.appendChild(document.createTextNode(' '));}tr.appendChild(actions);const state=cell((queued.has(row.espn_id)?'queued · ':'')+status+' · '+row.recommendation);state.className=status==='mine'?'status-mine':status==='taken'?'status-taken':'';tr.appendChild(state);body.appendChild(tr);}$('status').textContent='Current pick '+esc(currentPayload.current_pick)+' · next pick '+esc(currentPayload.next_pick)+' · '+currentPayload.decision_table.length+' validated players';$('current-pick-hero').textContent='Pick '+esc(currentPayload.current_pick);$('next-pick-hero').textContent=currentPayload.next_pick==null?'No later user pick':'Next user pick '+esc(currentPayload.next_pick);renderHealth();renderAnalytics();}
async function refresh(body=requestClock()){try{currentPayload=await post('/api/board',body);setSettings();renderBoard();$('status').className='';}catch(error){$('status').textContent=error.message;$('status').className='notice';}}
$('league').addEventListener('change',()=>{currentPayload=null;refresh({profile_id:$('league').value});});$('sync-clock').addEventListener('click',()=>refresh());$('recalculate').addEventListener('click',()=>refresh());$('undo').addEventListener('click',async()=>{try{currentPayload=await post('/api/action',{profile_id:$('league').value,action:{type:'undo'}});setSettings();renderBoard();}catch(error){$('status').textContent=error.message;$('status').className='notice';}});$('snapshot').addEventListener('click',async()=>{try{const result=await post('/api/snapshot',requestClock());$('status').textContent='Snapshot written to '+result.path;}catch(error){$('status').textContent=error.message;$('status').className='notice';}});$('board').addEventListener('click',async event=>{const button=event.target.closest('button[data-type]');if(!button)return;const type=button.dataset.type;try{currentPayload=await post('/api/action',{profile_id:$('league').value,action:{type:type==='queue'?'queue':'record',player_id:Number(button.dataset.id),disposition:type==='mine'?'mine':'taken'}});setSettings();renderBoard();}catch(error){$('status').textContent=error.message;$('status').className='notice';}});
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
