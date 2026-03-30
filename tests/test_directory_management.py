import importlib

from ffbayes.utils.visualization_manager import manage_visualizations


def _reload_path_constants(monkeypatch, tmp_path):
    project_root = tmp_path / 'Projects' / 'ffbayes'
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = (
        tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'
    )

    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_CLOUD_ROOT', str(cloud_root))

    import ffbayes.utils.path_constants as path_constants

    return importlib.reload(path_constants)


def test_create_all_required_directories_uses_runtime_only(tmp_path, monkeypatch):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    current_year = 2026
    assert path_constants.get_project_root() == tmp_path / 'Projects' / 'ffbayes'
    assert path_constants.get_runtime_root() == tmp_path / 'ProjectsRuntime' / 'ffbayes'
    assert (
        path_constants.get_cloud_root()
        == tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'
    )

    path_constants.create_all_required_directories(current_year)

    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = (
        tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'
    )

    assert (
        runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'artifacts'
    ).exists()
    assert (
        runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'diagnostics'
    ).exists()
    assert (runtime_root / 'data' / 'processed' / 'unified_dataset').exists()
    assert not (cloud_root / 'data').exists()
    assert not (cloud_root / 'results').exists()
    assert not (cloud_root / 'plots').exists()
    assert not (cloud_root / 'docs').exists()
    assert not (tmp_path / 'workspace' / 'results').exists()
    assert not (tmp_path / 'workspace' / 'plots').exists()


def test_manage_visualizations_publishes_selected_phase(tmp_path, monkeypatch):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    current_year = 2026
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = (
        tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'
    )

    pre_draft_artifact_dir = (
        runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'artifacts'
    )
    pre_draft_artifact_dir.mkdir(parents=True, exist_ok=True)
    (pre_draft_artifact_dir / 'vor_strategy' / 'example.json').parent.mkdir(
        parents=True, exist_ok=True
    )
    (pre_draft_artifact_dir / 'vor_strategy' / 'example.json').write_text(
        '{"ok": true}', encoding='utf-8'
    )
    (pre_draft_artifact_dir / 'draft_strategy' / f'dashboard_payload_{current_year}.json').parent.mkdir(
        parents=True, exist_ok=True
    )
    (pre_draft_artifact_dir / 'draft_strategy' / f'dashboard_payload_{current_year}.json').write_text(
        '{"ok": true}', encoding='utf-8'
    )

    pre_draft_plot_dir = (
        runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'diagnostics' / 'visualizations'
    )
    pre_draft_plot_dir.mkdir(parents=True, exist_ok=True)
    (pre_draft_plot_dir / 'chart.png').write_bytes(b'png')

    pre_draft_dashboard_dir = (
        runtime_root / 'runs' / str(current_year) / 'pre_draft' / 'artifacts' / 'draft_strategy'
    )
    pre_draft_dashboard_dir.mkdir(parents=True, exist_ok=True)

    result = manage_visualizations(current_year, phase='pre_draft')

    assert result['readme_updated'] is False
    assert result['phase'] == 'pre_draft'
    assert len(result['copied_files']) >= 4

    assert (
        cloud_root
        / 'results'
        / str(current_year)
        / 'pre_draft'
        / 'vor_strategy'
        / 'example.json'
    ).exists()
    assert (
        cloud_root
        / 'plots'
        / str(current_year)
        / 'pre_draft'
        / 'visualizations'
        / 'chart.png'
    ).exists()
    assert (
        cloud_root
        / 'dashboard'
        / str(current_year)
        / 'pre_draft'
        / 'draft_strategy'
        / f'dashboard_payload_{current_year}.json'
    ).exists()
    assert (cloud_root / 'docs' / 'images' / 'pre_draft_chart.png').exists()
    assert not (tmp_path / 'workspace' / 'docs' / 'images').exists()
