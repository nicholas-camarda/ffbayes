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


def test_split_command_forwards_without_phase_argument(monkeypatch):
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    exit_code = cli.main(['split', '--year', '2025'])

    assert exit_code == 0
    assert captured['module_name'] == 'ffbayes.run_pipeline_split'
    assert captured['argv'] == [
        'ffbayes.run_pipeline_split',
        '--year',
        '2025',
    ]


def test_publish_pages_command_forwards_extra_arguments(monkeypatch):
    captured = {}

    def fake_import(module_name):
        captured['module_name'] = module_name

        def fake_main():
            captured['argv'] = sys.argv[:]
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, 'import_module', fake_import)

    exit_code = cli.main(['publish-pages', '--year', '2026'])

    assert exit_code == 0
    assert captured['module_name'] == 'ffbayes.publish_pages'
    assert captured['argv'] == [
        'ffbayes.publish_pages',
        '--year',
        '2026',
    ]
