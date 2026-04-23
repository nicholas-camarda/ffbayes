import importlib
from datetime import datetime

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

    assert (runtime_root / 'seasons' / str(current_year)).exists()
    assert (runtime_root / 'seasons' / str(current_year) / 'diagnostics').exists()
    assert (runtime_root / 'inputs' / 'processed' / 'unified_dataset').exists()
    assert not (runtime_root / 'data').exists()
    assert not (runtime_root / 'datasets').exists()
    assert not (runtime_root / 'runs').exists()
    assert not (cloud_root / 'data').exists()
    assert not (cloud_root / 'results').exists()
    assert not (cloud_root / 'plots').exists()
    assert not (cloud_root / 'docs').exists()
    assert not (tmp_path / 'workspace' / 'results').exists()
    assert not (tmp_path / 'workspace' / 'plots').exists()


def test_create_all_required_directories_rejects_legacy_runtime_roots(
    tmp_path, monkeypatch
):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    (runtime_root / 'data').mkdir(parents=True, exist_ok=True)

    try:
        path_constants.create_all_required_directories(2026)
    except RuntimeError as exc:
        assert 'Legacy runtime directories are still present' in str(exc)
    else:
        raise AssertionError('Expected legacy runtime directories to be rejected')


def test_manage_visualizations_publishes_selected_phase(tmp_path, monkeypatch):
    (tmp_path / 'workspace').mkdir()
    monkeypatch.chdir(tmp_path / 'workspace')
    path_constants = _reload_path_constants(monkeypatch, tmp_path)

    current_year = 2026
    runtime_root = tmp_path / 'ProjectsRuntime' / 'ffbayes'
    cloud_root = (
        tmp_path / 'CloudStorage' / 'OneDrive-Personal' / 'SideProjects' / 'ffbayes'
    )

    pre_draft_artifact_dir = runtime_root / 'seasons' / str(current_year)
    pre_draft_artifact_dir.mkdir(parents=True, exist_ok=True)
    (runtime_root / 'inputs' / 'raw' / 'season_datasets').mkdir(parents=True, exist_ok=True)
    (runtime_root / 'inputs' / 'raw' / 'season_datasets' / '2025season.csv').write_text(
        'Name,Season\nPatrick Mahomes,2025\n', encoding='utf-8'
    )
    (runtime_root / 'inputs' / 'raw' / 'collection_manifest.json').write_text(
        '{"ok": true}', encoding='utf-8'
    )
    (runtime_root / 'inputs' / 'processed' / 'combined_datasets').mkdir(
        parents=True, exist_ok=True
    )
    (
        runtime_root
        / 'inputs'
        / 'processed'
        / 'combined_datasets'
        / '2021-2025season_modern.csv'
    ).write_text('Name,Season\nPatrick Mahomes,2025\n', encoding='utf-8')
    (runtime_root / 'inputs' / 'processed' / 'snake_draft_datasets').mkdir(
        parents=True, exist_ok=True
    )
    (
        runtime_root
        / 'inputs'
        / 'processed'
        / 'snake_draft_datasets'
        / 'snake-draft_ppr-0.5.csv'
    ).write_text('Name,VOR\nPatrick Mahomes,10\n', encoding='utf-8')
    (runtime_root / 'inputs' / 'processed' / 'unified_dataset').mkdir(
        parents=True, exist_ok=True
    )
    (
        runtime_root / 'inputs' / 'processed' / 'unified_dataset' / 'unified_dataset.csv'
    ).write_text('Name,Season\nPatrick Mahomes,2025\n', encoding='utf-8')
    (
        runtime_root / 'inputs' / 'processed' / 'unified_dataset' / 'unified_dataset.json'
    ).write_text('{"rows": 1}', encoding='utf-8')

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
    (pre_draft_artifact_dir / 'draft_strategy' / f'draft_board_{current_year}.html').write_text(
        '<html><body>dashboard</body></html>', encoding='utf-8'
    )

    pre_draft_plot_dir = (
        runtime_root / 'seasons' / str(current_year) / 'diagnostics' / 'visualizations'
    )
    pre_draft_plot_dir.mkdir(parents=True, exist_ok=True)
    (pre_draft_plot_dir / 'chart.png').write_bytes(b'png')

    pre_draft_dashboard_dir = runtime_root / 'seasons' / str(current_year) / 'draft_strategy'
    pre_draft_dashboard_dir.mkdir(parents=True, exist_ok=True)
    (pre_draft_dashboard_dir / 'notes.txt').write_text('do not publish', encoding='utf-8')
    (pre_draft_dashboard_dir / f'draft_board_{current_year}.xlsx').write_bytes(b'xlsx')

    result = manage_visualizations(current_year, phase='pre_draft')

    assert result['readme_updated'] is False
    assert result['phase'] == 'pre_draft'
    assert len(result['copied_files']) >= 8

    snapshot_dir = cloud_root / 'Analysis' / datetime.now().strftime('%Y-%m-%d')
    assert result['snapshot_dir'] == str(snapshot_dir)

    assert (cloud_root / 'data' / 'raw' / 'season_datasets' / '2025season.csv').exists()
    assert (
        cloud_root / 'data' / 'raw' / 'manifests' / f'collection_manifest_{current_year}.json'
    ).exists()
    assert (
        cloud_root
        / 'data'
        / 'processed'
        / 'combined_datasets'
        / '2021-2025season_modern.csv'
    ).exists()
    assert (
        cloud_root / 'data' / 'processed' / 'unified_dataset' / f'unified_dataset_{current_year}.csv'
    ).exists()
    assert (
        cloud_root / 'data' / 'processed' / 'unified_dataset' / f'unified_dataset_{current_year}.json'
    ).exists()
    assert (snapshot_dir / 'vor_strategy' / 'example.json').exists()
    assert (snapshot_dir / 'diagnostics' / 'visualizations' / 'chart.png').exists()
    assert (snapshot_dir / 'dashboard' / 'dashboard_payload.json').exists()
    assert (snapshot_dir / 'dashboard' / 'index.html').exists()
    assert not (
        snapshot_dir
        / 'dashboard'
        / 'notes.txt'
    ).exists()
    assert not (
        snapshot_dir
        / 'dashboard'
        / f'draft_board_{current_year}.xlsx'
    ).exists()
    assert not (snapshot_dir / 'data').exists()
    assert not (snapshot_dir / 'runs').exists()
    assert not (cloud_root / 'results').exists()
    assert not (cloud_root / 'plots').exists()
    assert not (cloud_root / 'docs').exists()
    assert not (tmp_path / 'workspace' / 'docs' / 'images').exists()
