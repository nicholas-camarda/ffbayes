from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_draft_2026_dashboard_service import _players, _profile

from ffbayes.draft_2026.dashboard_app import DashboardService, create_http_server
from ffbayes.draft_2026.pipeline import FreshInputs, _sha256_json

players = _players()
players.loc[players.index[0], 'name'] = (
    '<img src=x onerror="window.__hostileExecuted=true"> hostile-player'
)
inputs = FreshInputs(
    payload={'players': []},
    source_manifest={'season': 2026, 'sha256': 'espn-hostile-fixture', 'fetched_at': '2026-08-23T00:00:00Z'},
    roster=pd.DataFrame(),
    roster_manifest={'season': 2026, 'sha256': 'roster-hostile-fixture', 'fetched_at': '2026-08-23T00:00:00Z'},
    players=players,
    coverage={'status': 'passed', 'rows': len(players)},
)


class HostileDashboardService(DashboardService):
    def handle_board(self, request):
        payload = super().handle_board(request)
        row = next(
            row for row in payload['decision_table'] if 'hostile-player' in row['name']
        )
        row['recommendation'] = (
            '<button data-action="evil" onclick="window.__hostileExecuted=true">pwned</button>'
        )
        payload['provenance']['board_sha256'] = _sha256_json(payload['decision_table'])
        return payload


service = HostileDashboardService(
    inputs,
    [_profile('bill', 2)],
    project_root=Path.cwd(),
    run_root=Path.cwd() / 'runtime' / 'runs' / 'browser-hostile-fixture',
    code_revision='browser-hostile-fixture',
)
server = create_http_server(service, 0)
print(f'PORT={server.server_address[1]}', flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
