from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import ffbayes.cli as cli


def test_top_level_help_exposes_only_the_operator_dashboard(capsys) -> None:
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert 'ffbayes dashboard --year 2026' in output
    assert 'pre-draft' not in output
    assert 'stage-dashboard' not in output
    assert 'draft-strategy' not in output


def test_dashboard_dispatch_is_strictly_forwarded(monkeypatch) -> None:
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    assert cli.main(['dashboard', '--year', '2026', '--no-browser']) == 0
    assert captured == {
        'module_name': 'ffbayes.draft_2026.dashboard_app',
        'argv': [
            'ffbayes.draft_2026.dashboard_app',
            '--year',
            '2026',
            '--no-browser',
        ],
    }


def test_dashboard_unknown_option_is_rejected_by_its_parser(monkeypatch, capsys) -> None:
    assert cli.main(['dashboard', '--year', '2026', '--bogus']) == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_legacy_help_does_not_import_or_run_work(monkeypatch, capsys) -> None:
    imported = []

    def fail_import(module_name):
        imported.append(module_name)
        raise AssertionError('legacy help imported a work-producing module')

    monkeypatch.setattr(cli.importlib, 'import_module', fail_import)
    assert cli.main(['draft-backtest', '--help']) == 0
    assert imported == []
    assert 'developer/maintenance' in capsys.readouterr().out


def test_only_ffbayes_console_script_is_declared() -> None:
    pyproject = Path('pyproject.toml').read_text(encoding='utf-8')
    scripts = pyproject.split('[project.scripts]', 1)[1].split('[tool.ruff', 1)[0]
    assert scripts.count(' = ') == 1
    assert 'ffbayes = "ffbayes.cli:main"' in scripts
