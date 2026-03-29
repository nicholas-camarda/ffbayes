import sys
from types import SimpleNamespace

import ffbayes.cli as cli


def test_collect_command_forwards_extra_arguments(monkeypatch):
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return None

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    exit_code = cli.main(['collect', '--years', '2022,2023'])

    assert exit_code == 0
    assert captured['module_name'] == 'ffbayes.data_pipeline.collect_data'
    assert captured['argv'] == [
        'ffbayes.data_pipeline.collect_data',
        '--years',
        '2022,2023',
    ]


def test_pre_draft_shortcut_injects_phase_argument(monkeypatch):
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    exit_code = cli.main(['pre-draft', '--year', '2025'])

    assert exit_code == 0
    assert captured['module_name'] == 'ffbayes.run_pipeline_split'
    assert captured['argv'] == [
        'ffbayes.run_pipeline_split',
        'pre_draft',
        '--year',
        '2025',
    ]


def test_split_command_requires_phase_argument(capsys):
    exit_code = cli.main(['split'])

    assert exit_code == 2
    assert 'requires a phase argument' in capsys.readouterr().err
