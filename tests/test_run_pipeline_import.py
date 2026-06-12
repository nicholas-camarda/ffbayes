import importlib
import os
import subprocess
import sys


def test_run_pipeline_import_has_no_bootstrap_side_effects(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - only used if regression reappears
        raise AssertionError('Import-time bootstrap should not run')

    monkeypatch.setattr(subprocess, 'run', fail)
    monkeypatch.setattr(os, 'execv', fail)

    module = importlib.import_module('ffbayes.run_pipeline')
    importlib.reload(module)

    assert callable(module.ensure_conda_environment)


def test_run_pipeline_rejects_unimplemented_phase(monkeypatch, tmp_path):
    module = importlib.import_module('ffbayes.run_pipeline')
    monkeypatch.setattr(module, 'ensure_conda_environment', lambda: None)
    monkeypatch.setattr(module, 'create_required_directories', lambda: None)
    monkeypatch.setattr(module, 'cleanup_empty_directories', lambda: None)
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(tmp_path / 'project'))
    monkeypatch.setattr(sys, 'argv', ['ffbayes-pipeline', '--phase', 'draft'])

    assert module.main() == 2


def test_run_pipeline_full_phase_delegates_to_pre_draft(monkeypatch, tmp_path):
    module = importlib.import_module('ffbayes.run_pipeline')
    captured: dict[str, object] = {}

    def fake_run_pre_draft(argv):
        captured['argv'] = list(argv)
        return 0

    monkeypatch.setattr(module, 'ensure_conda_environment', lambda: None)
    monkeypatch.setattr(module, 'create_required_directories', lambda: None)
    monkeypatch.setattr(module, 'cleanup_empty_directories', lambda: None)
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(tmp_path / 'project'))
    monkeypatch.setattr(
        'ffbayes.run_pipeline_split.run_pre_draft', fake_run_pre_draft
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['ffbayes-pipeline', '--phase', 'full', '--year', '2026', '--stage-pages'],
    )

    assert module.main() == 0
    assert captured['argv'] == ['--year', '2026', '--stage-pages']
