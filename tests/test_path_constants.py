from pathlib import Path

import pytest

import ffbayes.utils.path_constants as path_constants


def test_get_runtime_root_honors_explicit_env_override(monkeypatch, tmp_path):
    override = tmp_path / 'runtime-override'
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(override))

    assert path_constants.get_runtime_root() == override.resolve()


def test_get_runtime_root_falls_back_when_default_is_unwritable(
    monkeypatch, tmp_path
):
    monkeypatch.delenv('FFBAYES_RUNTIME_ROOT', raising=False)
    monkeypatch.setattr(path_constants, '_path_is_writable', lambda _: False)

    fallback = path_constants.get_runtime_root()
    expected = Path(path_constants.__file__).resolve().parents[3] / '.ffbayes_runtime'

    assert fallback == expected


def test_get_runtime_root_prefers_default_when_writable(monkeypatch, tmp_path):
    monkeypatch.delenv('FFBAYES_RUNTIME_ROOT', raising=False)
    monkeypatch.setattr(path_constants, '_path_is_writable', lambda _: True)

    expected = Path.home() / 'ProjectsRuntime' / 'ffbayes'

    assert path_constants.get_runtime_root() == expected


def test_pre_draft_paths_live_under_artifacts_and_diagnostics(monkeypatch, tmp_path):
    monkeypatch.delenv('FFBAYES_RUNTIME_ROOT', raising=False)
    monkeypatch.setattr(path_constants, '_path_is_writable', lambda _: True)
    monkeypatch.setattr(path_constants, 'RUNTIME_DIR', tmp_path / 'runtime', raising=False)
    monkeypatch.setattr(path_constants, 'BASE_DIR', tmp_path / 'project', raising=False)

    artifacts_dir = path_constants.get_pre_draft_artifacts_dir(2026)
    diagnostics_dir = path_constants.get_pre_draft_diagnostics_dir(2026)
    pages_dir = path_constants.get_pages_site_dir()

    assert artifacts_dir == tmp_path / 'runtime' / 'runs' / '2026' / 'pre_draft' / 'artifacts'
    assert diagnostics_dir == tmp_path / 'runtime' / 'runs' / '2026' / 'pre_draft' / 'diagnostics'
    assert path_constants.get_dashboard_payload_path(2026).parent == (
        tmp_path / 'runtime' / 'runs' / '2026' / 'pre_draft' / 'artifacts' / 'draft_strategy'
    )
    assert path_constants.get_dashboard_html_path(2026).parent == (
        tmp_path / 'runtime' / 'runs' / '2026' / 'pre_draft' / 'artifacts' / 'draft_strategy'
    )
    assert pages_dir == tmp_path / 'project' / 'site'
    assert path_constants.get_results_dir(2026) == artifacts_dir
    assert path_constants.get_plots_dir(2026) == diagnostics_dir


def test_get_phase_name_rejects_non_pre_draft(monkeypatch):
    monkeypatch.delenv('FFBAYES_PIPELINE_PHASE', raising=False)

    assert path_constants.get_phase_name('pre_draft') == 'pre_draft'
    with pytest.raises(ValueError):
        path_constants.get_phase_name('post_draft')

    monkeypatch.setenv('FFBAYES_PIPELINE_PHASE', 'post_draft')
    with pytest.raises(ValueError):
        path_constants.get_phase_name()


def test_get_cloud_root_falls_back_when_default_is_unwritable(monkeypatch):
    monkeypatch.delenv('FFBAYES_CLOUD_ROOT', raising=False)
    monkeypatch.setattr(path_constants, '_path_is_writable', lambda _: False)

    fallback = path_constants.get_cloud_root()
    expected = Path(path_constants.__file__).resolve().parents[3] / '.ffbayes_cloud'

    assert fallback == expected
