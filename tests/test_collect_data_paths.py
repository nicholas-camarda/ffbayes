import json

from ffbayes.data_pipeline.collect_data import (
    build_collection_manifest,
    write_collection_manifest,
)


def test_collection_manifest_captures_runtime_outputs_only(tmp_path):
    runtime_season_dir = tmp_path / 'runtime' / 'season_datasets'
    runtime_combined_dir = tmp_path / 'runtime' / 'combined_datasets'
    runtime_manifest_path = tmp_path / 'runtime' / 'collection_manifest.json'
    freshness_manifest_path = tmp_path / 'runtime' / 'freshness_manifest.json'

    runtime_season_dir.mkdir(parents=True, exist_ok=True)
    runtime_combined_dir.mkdir(parents=True, exist_ok=True)

    season_csv = 'G#,Season,Name\n1,2025,Player A\n'
    combined_csv = 'Season,Name\n2025,Player A\n'
    (runtime_season_dir / '2025season.csv').write_text(season_csv, encoding='utf-8')
    (runtime_combined_dir / 'combined_data.csv').write_text(
        combined_csv, encoding='utf-8'
    )

    manifest = build_collection_manifest(
        requested_years=[2025, 2024],
        successful_years=[2025],
        runtime_season_dir=runtime_season_dir,
        runtime_combined_dir=runtime_combined_dir,
        source_manifest={'status': 'fresh'},
    )

    written_runtime_path, written_freshness_path = write_collection_manifest(
        manifest,
        runtime_manifest_path=runtime_manifest_path,
        freshness_manifest_path=freshness_manifest_path,
    )

    assert written_runtime_path == runtime_manifest_path
    assert written_freshness_path == freshness_manifest_path
    assert runtime_manifest_path.exists()
    assert freshness_manifest_path.exists()
    assert json.loads(runtime_manifest_path.read_text(encoding='utf-8')) == json.loads(
        freshness_manifest_path.read_text(encoding='utf-8')
    )
    assert manifest['requested_years'] == [2025, 2024]
    assert manifest['successful_years'] == [2025]
    assert manifest['runtime']['season_files'][0]['rows'] == 1
    assert manifest['runtime']['combined_file']['rows'] == 1
    assert manifest['source_manifest']['status'] == 'fresh'
    assert 'cloud_raw' not in manifest
