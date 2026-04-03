import subprocess
from pathlib import Path


def test_dashboard_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ['node', 'tests/dashboard_smoke.mjs'],
        cwd=repo_root,
        check=True,
    )
