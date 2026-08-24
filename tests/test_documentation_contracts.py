from __future__ import annotations

import shlex
from pathlib import Path

import ffbayes.cli as cli

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / 'docs'
CURRENT_DOCS = [
    REPO_ROOT / 'README.md',
    DOCS_DIR / 'README.md',
    DOCS_DIR / 'DASHBOARD_OPERATOR_GUIDE.md',
    DOCS_DIR / 'DATA_LINEAGE_AND_PATHS.md',
    DOCS_DIR / 'METRIC_REFERENCE.md',
    DOCS_DIR / 'OUTPUT_EXAMPLES.md',
    DOCS_DIR / 'DASHBOARD_FRONTEND_ARCHITECTURE.md',
]


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _bash_commands(path: Path) -> list[str]:
    commands: list[str] = []
    in_bash = False
    active = ''
    for raw in _read(path).splitlines():
        line = raw.strip()
        if line == '```bash':
            in_bash = True
            continue
        if in_bash and line == '```':
            if active:
                commands.append(active)
            active = ''
            in_bash = False
            continue
        if in_bash and line and not line.startswith('#'):
            active = f'{active} {line}'.strip() if active else line
            if active.endswith('\\'):
                active = active[:-1].rstrip()
            elif active:
                commands.append(active)
                active = ''
    return [command for command in commands if command.startswith('ffbayes ')]


def test_docs_index_links_current_guides() -> None:
    text = _read(DOCS_DIR / 'README.md')
    for filename in [
        'DASHBOARD_OPERATOR_GUIDE.md',
        'DATA_LINEAGE_AND_PATHS.md',
        'METRIC_REFERENCE.md',
        'OUTPUT_EXAMPLES.md',
        'DASHBOARD_FRONTEND_ARCHITECTURE.md',
    ]:
        assert filename in text


def test_operator_guide_has_shared_conventions() -> None:
    text = _read(DOCS_DIR / 'DASHBOARD_OPERATOR_GUIDE.md')
    for marker in [
        '## One command',
        '## First-time setup',
        '## Draft-day workflow',
        '## What the board checks before it opens',
        'runtime/runs/dashboard_2026',
    ]:
        assert marker in text


def test_current_docs_document_one_operator_command_and_generic_profiles() -> None:
    combined = '\n'.join(_read(path) for path in CURRENT_DOCS)
    assert combined.count('ffbayes dashboard --year 2026') >= 3
    assert 'example_2026.json' in combined
    assert '*.local.json' in combined
    assert "Bill's Underbit" not in combined
    assert 'Camarda-Klein Family' not in combined
    assert 'runtime/runs/dashboard_2026' in combined
    assert 'loopback' in combined.lower()
    assert 'nflverse' in combined


def test_current_docs_do_not_advertise_legacy_board_commands() -> None:
    operator_surface = '\n'.join(
        _read(path)
        for path in [
            REPO_ROOT / 'README.md',
            DOCS_DIR / 'README.md',
            DOCS_DIR / 'DASHBOARD_OPERATOR_GUIDE.md',
        ]
    )
    for term in [
        'ffbayes pre-draft',
        'ffbayes stage-dashboard',
        'ffbayes draft-strategy',
        'dashboard/index.html',
        '--stage-pages',
    ]:
        assert term not in operator_surface


def test_documented_ffbayes_commands_match_the_public_cli() -> None:
    public_commands = {'dashboard'}
    for path in CURRENT_DOCS:
        for command in _bash_commands(path):
            tokens = shlex.split(command)
            assert tokens[0] == 'ffbayes'
            assert tokens[1] in public_commands
            assert '--year' in tokens
            assert tokens[tokens.index('--year') + 1] == '2026'

    help_text = cli.build_parser().format_help()
    assert 'ffbayes dashboard --year 2026' in help_text
    assert 'pre-draft' not in help_text


def test_metric_reference_defines_current_model_labels() -> None:
    text = _read(DOCS_DIR / 'METRIC_REFERENCE.md')
    for label in [
        'Scoring',
        'Roster demand and replacement',
        'VOR',
        'Scarcity',
        'ADP timing',
        'draft_now',
        'slot_required',
        'board_priority',
    ]:
        assert label in text
    assert 'does not mean a measured zero relationship' in text


def test_output_example_records_stable_ids_and_provenance() -> None:
    text = _read(DOCS_DIR / 'OUTPUT_EXAMPLES.md')
    for marker in [
        '"schema_version": "draft_2026_v1"',
        '"espn_id"',
        '"runtime_state"',
        '"state_sha256"',
        '"source_manifests"',
    ]:
        assert marker in text


def test_current_docs_do_not_contain_private_workflow_or_stale_trust_language() -> None:
    combined = '\n'.join(_read(path) for path in CURRENT_DOCS)
    assert 'public GitHub' not in combined
    assert 'unpushed worktree' not in combined
    assert 'Trust boundary:' not in combined
    assert "Bill's Underbit" not in combined
    assert 'Camarda-Klein Family' not in combined
