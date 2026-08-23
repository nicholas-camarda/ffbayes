"""Print a validated draft-2026 dashboard fixture for the Playwright smoke test."""

from __future__ import annotations

import pandas as pd

from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.pipeline import build_dashboard_payload, render_dashboard_html

profile = LeagueProfile.from_mapping(
    {
        'profile_id': 'browser-smoke',
        'league_name': 'Browser Smoke League',
        'season': 2026,
        'team_count': 10,
        'draft_format': 'snake',
        'draft_slot': 5,
        'scoring_label': 'fixture',
        'scoring_items': {'53': 1.0},
        'scoring_overrides': {},
        'bonuses': [],
        'roster_slots': {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1,
            'DST': 1,
            'K': 1,
        },
        'bench_slots': 7,
        'ir_slots': 1,
        'flex_eligible': ['RB', 'WR', 'TE'],
        'waiver_type': 'fixture',
        'waiver_constraints': [],
        'settings_source': 'browser smoke fixture',
        'settings_verified_at': '2026-08-22T12:00:00Z',
    }
)
board = pd.DataFrame(
    {
        'board_rank': [1],
        'espn_id': [101],
        'name': ['Player One'],
        'position': ['WR'],
        'projected_points': [250.0],
        'replacement_level': [150.0],
        'vor': [100.0],
        'scarcity': [5.0],
        'adp': [5.0],
        'availability_next_pick': [0.1],
        'recommendation': ['draft_now'],
        'roster_status': ['available'],
        'is_available': [True],
    }
)
board.attrs['replacement'] = {'levels': {'WR': 150.0}, 'demand': {'WR': 30}}
board.attrs['current_pick'] = 5
board.attrs['next_pick'] = 16
board.attrs['runtime_state'] = {
    'draft_slot': 5,
    'current_pick': 5,
    'taken_ids': [],
    'your_ids': [],
    'queue_ids': [],
}
manifest = {
    'season': 2026,
    'sha256': 'fixture-digest',
    'fetched_at': '2026-08-22T12:00:00Z',
}
payload = build_dashboard_payload(
    board,
    profile,
    source_manifest=manifest,
    roster_manifest={**manifest, 'sha256': 'roster-fixture-digest'},
    coverage_report={'status': 'passed'},
    code_revision='browser-smoke-fixture',
)
print(render_dashboard_html(payload))
