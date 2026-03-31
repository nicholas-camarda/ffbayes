import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATH_PREFIXES = (
    'logs/',
    'results/',
    'plots/',
    'docs/images/',
    'my_ff_teams/',
    'misc-datasets/',
    '.playwright-mcp/',
)


def test_generated_artifacts_are_not_tracked():
    tracked_files = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'ls-files'],
        text=True,
    ).splitlines()

    tracked_generated = [
        path for path in tracked_files if path.startswith(GENERATED_PATH_PREFIXES)
    ]

    assert not tracked_generated, (
        'Generated artifact paths must not be tracked in Git: '
        f'{tracked_generated}'
    )


def test_runtime_code_does_not_import_nfl_data_py():
    runtime_sources = (REPO_ROOT / 'src').rglob('*.py')
    offenders = [
        path
        for path in runtime_sources
        if 'nfl_data_py' in path.read_text(encoding='utf-8')
    ]

    assert not offenders, f'Runtime code still references nfl_data_py: {offenders}'
