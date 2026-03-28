import importlib
from pathlib import Path

import pandas as pd

from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset


def _reload_path_constants(monkeypatch, tmp_path):
    project_root = tmp_path / 'Projects' / 'ffbayes'
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'

    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_CLOUD_ROOT', str(cloud_root))

    import ffbayes.utils.path_constants as path_constants

    return importlib.reload(path_constants)


def test_unified_loader_prefers_runtime_dataset(monkeypatch, tmp_path):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    runtime_dataset = path_constants.get_unified_dataset_path()
    runtime_dataset.parent.mkdir(parents=True, exist_ok=True)
    expected = pd.DataFrame(
        {
            'Name': ['Runtime Player'],
            'Position': ['QB'],
            'Season': [2026],
            'FantPt': [21.5],
        }
    )
    expected.to_json(runtime_dataset)

    # A poison file in the cwd should be ignored.
    poison_dataset = Path.cwd() / 'datasets' / 'unified_dataset' / 'unified_dataset.json'
    poison_dataset.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            'Name': ['Poison Player'],
            'Position': ['RB'],
            'Season': [2026],
            'FantPt': [1.0],
        }
    ).to_json(poison_dataset)

    loaded_default = load_unified_dataset()
    loaded_legacy = load_unified_dataset('datasets')

    assert loaded_default.equals(expected)
    assert loaded_legacy.equals(expected)


def test_unified_loader_supports_explicit_override(monkeypatch, tmp_path):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    _reload_path_constants(monkeypatch, tmp_path)

    override_root = tmp_path / 'explicit_override'
    override_dataset = override_root / 'unified_dataset' / 'unified_dataset.json'
    override_dataset.parent.mkdir(parents=True, exist_ok=True)
    expected = pd.DataFrame(
        {
            'Name': ['Override Player'],
            'Position': ['WR'],
            'Season': [2026],
            'FantPt': [18.25],
        }
    )
    expected.to_json(override_dataset)

    loaded_override = load_unified_dataset(str(override_root))

    assert loaded_override.equals(expected)
