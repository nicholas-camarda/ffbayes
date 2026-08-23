import os
import subprocess
import sys
from pathlib import Path


def test_dashboard_smoke_targets_canonical_fixture_not_site():
    repo_root = Path(__file__).resolve().parents[1]
    smoke_script = (repo_root / 'tests' / 'dashboard_smoke.mjs').read_text(
        encoding='utf-8'
    )
    assert 'test_draft_2026_dashboard_browser.mjs' in smoke_script
    assert 'site' not in smoke_script

    env = os.environ.copy()
    env['PYTHON'] = sys.executable
    result = subprocess.run(
        ['node', 'tests/dashboard_smoke.mjs'],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '2026 interactive dashboard browser smoke passed' in result.stdout
