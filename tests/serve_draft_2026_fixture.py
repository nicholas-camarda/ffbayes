from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_draft_2026_dashboard_service import _players, _profile

from ffbayes.draft_2026.dashboard_app import DashboardService, create_http_server
from ffbayes.draft_2026.pipeline import FreshInputs

players = _players()
inputs = FreshInputs(
    payload={'players': []},
    source_manifest={'season': 2026, 'sha256': 'espn-fixture', 'fetched_at': '2026-08-23T00:00:00Z'},
    roster=pd.DataFrame(),
    roster_manifest={'season': 2026, 'sha256': 'roster-fixture', 'fetched_at': '2026-08-23T00:00:00Z'},
    players=players,
    coverage={'status': 'passed', 'rows': len(players)},
)
service = DashboardService(
    inputs,
    [_profile('bill', 2), _profile('family', 1)],
    project_root=Path.cwd(),
    run_root=Path.cwd() / 'runtime' / 'runs' / 'browser-fixture',
    code_revision='browser-fixture',
)
server = create_http_server(service, 0)
print(f'PORT={server.server_address[1]}', flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
