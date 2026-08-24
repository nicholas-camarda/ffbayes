import sys
from types import SimpleNamespace

import pytest

import ffbayes.cli as cli


def test_dashboard_command_forwards_extra_arguments(monkeypatch):
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return None

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    exit_code = cli.main(['dashboard', '--year', '2026', '--no-browser'])

    assert exit_code == 0
    assert captured['module_name'] == 'ffbayes.draft_2026.dashboard_app'
    assert captured['argv'] == [
        'ffbayes.draft_2026.dashboard_app',
        '--year',
        '2026',
        '--no-browser',
    ]


def test_unknown_commands_are_rejected_without_importing_modules(monkeypatch):
    monkeypatch.setattr(
        cli.importlib,
        'import_module',
        lambda _: (_ for _ in ()).throw(AssertionError('unexpected import')),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(['pre-draft', '--year', '2026'])
    assert exc.value.code == 2
