from pathlib import Path

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


def test_get_cloud_root_falls_back_when_default_is_unwritable(monkeypatch):
    monkeypatch.delenv('FFBAYES_CLOUD_ROOT', raising=False)
    monkeypatch.setattr(path_constants, '_path_is_writable', lambda _: False)

    fallback = path_constants.get_cloud_root()
    expected = Path(path_constants.__file__).resolve().parents[3] / '.ffbayes_cloud'

    assert fallback == expected
