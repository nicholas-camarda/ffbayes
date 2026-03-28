import importlib

from ffbayes.utils.visualization_manager import manage_visualizations


def _reload_path_constants(monkeypatch, tmp_path):
    project_root = tmp_path / 'Projects' / 'ffbayes'
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'

    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_CLOUD_ROOT', str(cloud_root))

    import ffbayes.utils.path_constants as path_constants

    return importlib.reload(path_constants)


def test_create_all_required_directories_uses_runtime_and_cloud_roots(tmp_path, monkeypatch):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    current_year = 2026
    assert path_constants.get_project_root() == tmp_path / 'Projects' / 'ffbayes'
    assert path_constants.get_runtime_root() == tmp_path / 'ProjectsRuntime' / 'ffbayes'
    assert path_constants.get_cloud_root() == tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'

    path_constants.create_all_required_directories(current_year)

    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'

    assert (runtime_root / 'results' / str(current_year) / 'pre_draft').exists()
    assert (runtime_root / 'plots' / str(current_year) / 'post_draft').exists()
    assert (runtime_root / 'datasets' / 'unified_dataset').exists()
    assert (cloud_root / 'data' / 'raw').exists()
    assert (cloud_root / 'data' / 'raw' / 'season_datasets').exists()
    assert (cloud_root / 'data' / 'raw' / 'combined_datasets').exists()
    assert (cloud_root / 'data' / 'processed').exists()
    assert (cloud_root / 'results' / str(current_year) / 'pre_draft').exists()
    assert (cloud_root / 'plots' / str(current_year) / 'post_draft').exists()
    assert (cloud_root / 'docs' / 'images').exists()
    assert not (tmp_path / 'workspace' / 'results').exists()
    assert not (tmp_path / 'workspace' / 'plots').exists()


def test_manage_visualizations_publishes_to_cloud_workspace(tmp_path, monkeypatch):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    current_year = 2026
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'

    pre_draft_result = runtime_root / 'results' / str(current_year) / 'pre_draft' / 'vor_strategy'
    pre_draft_result.mkdir(parents=True, exist_ok=True)
    (pre_draft_result / 'example.json').write_text('{"ok": true}', encoding='utf-8')

    post_draft_result = runtime_root / 'results' / str(current_year) / 'post_draft' / 'team_aggregation'
    post_draft_result.mkdir(parents=True, exist_ok=True)
    (post_draft_result / 'example.tsv').write_text('player\tvalue\n', encoding='utf-8')

    pre_draft_plot_dir = runtime_root / 'plots' / str(current_year) / 'pre_draft' / 'visualizations'
    pre_draft_plot_dir.mkdir(parents=True, exist_ok=True)
    (pre_draft_plot_dir / 'chart.png').write_bytes(b'png')

    post_draft_plot_dir = runtime_root / 'plots' / str(current_year) / 'post_draft'
    post_draft_plot_dir.mkdir(parents=True, exist_ok=True)
    (post_draft_plot_dir / 'summary.png').write_bytes(b'png')

    result = manage_visualizations(current_year)

    assert result['readme_updated'] is False
    assert len(result['copied_files']) == 6

    assert (cloud_root / 'results' / str(current_year) / 'pre_draft' / 'vor_strategy' / 'example.json').exists()
    assert (cloud_root / 'results' / str(current_year) / 'post_draft' / 'team_aggregation' / 'example.tsv').exists()
    assert (cloud_root / 'plots' / str(current_year) / 'pre_draft' / 'visualizations' / 'chart.png').exists()
    assert (cloud_root / 'plots' / str(current_year) / 'post_draft' / 'summary.png').exists()
    assert (cloud_root / 'docs' / 'images' / 'pre_draft_chart.png').exists()
    assert (cloud_root / 'docs' / 'images' / 'post_draft_summary.png').exists()
    assert not (tmp_path / 'workspace' / 'docs' / 'images').exists()
