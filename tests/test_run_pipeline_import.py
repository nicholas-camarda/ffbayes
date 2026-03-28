import importlib
import os
import subprocess


def test_run_pipeline_import_has_no_bootstrap_side_effects(monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - only used if regression reappears
        raise AssertionError('Import-time bootstrap should not run')

    monkeypatch.setattr(subprocess, 'run', fail)
    monkeypatch.setattr(os, 'execv', fail)

    module = importlib.import_module('ffbayes.run_pipeline')
    importlib.reload(module)

    assert callable(module.ensure_conda_environment)
