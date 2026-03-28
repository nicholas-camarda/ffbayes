import json

from ffbayes.data_pipeline.collect_data import (
    build_collection_manifest,
    mirror_raw_dataset,
    write_collection_manifest,
)


def test_mirror_raw_dataset_copies_to_destination(tmp_path):
    source_file = tmp_path / 'source' / '2025season.csv'
    destination_dir = tmp_path / 'cloud' / 'data' / 'raw' / 'season_datasets'

    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text('season,data\n2025,ok\n', encoding='utf-8')

    destination_file = mirror_raw_dataset(source_file, destination_dir)

    assert destination_file == destination_dir / source_file.name
    assert destination_file.exists()
    assert destination_file.read_text(encoding='utf-8') == source_file.read_text(encoding='utf-8')


def test_collection_manifest_captures_runtime_and_raw_outputs(tmp_path):
    runtime_season_dir = tmp_path / 'runtime' / 'season_datasets'
    raw_season_dir = tmp_path / 'cloud' / 'data' / 'raw' / 'season_datasets'
    runtime_combined_dir = tmp_path / 'runtime' / 'combined_datasets'
    raw_combined_dir = tmp_path / 'cloud' / 'data' / 'raw' / 'combined_datasets'
    runtime_manifest_path = tmp_path / 'runtime' / 'collection_manifest.json'
    raw_manifest_path = tmp_path / 'cloud' / 'data' / 'raw' / 'collection_manifest.json'

    runtime_season_dir.mkdir(parents=True, exist_ok=True)
    raw_season_dir.mkdir(parents=True, exist_ok=True)
    runtime_combined_dir.mkdir(parents=True, exist_ok=True)
    raw_combined_dir.mkdir(parents=True, exist_ok=True)

    season_csv = 'G#,Season,Name\n1,2025,Player A\n'
    combined_csv = 'Season,Name\n2025,Player A\n'
    for directory in (runtime_season_dir, raw_season_dir):
        (directory / '2025season.csv').write_text(season_csv, encoding='utf-8')
    for directory in (runtime_combined_dir, raw_combined_dir):
        (directory / 'combined_data.csv').write_text(combined_csv, encoding='utf-8')

    manifest = build_collection_manifest(
        requested_years=[2025, 2024],
        successful_years=[2025],
        runtime_season_dir=runtime_season_dir,
        raw_season_dir=raw_season_dir,
        runtime_combined_dir=runtime_combined_dir,
        raw_combined_dir=raw_combined_dir,
    )

    written_runtime_path, written_raw_path = write_collection_manifest(
        manifest,
        runtime_manifest_path=runtime_manifest_path,
        raw_manifest_path=raw_manifest_path,
    )

    assert written_runtime_path == runtime_manifest_path
    assert written_raw_path == raw_manifest_path
    assert runtime_manifest_path.exists()
    assert raw_manifest_path.exists()
    assert json.loads(runtime_manifest_path.read_text(encoding='utf-8')) == json.loads(
        raw_manifest_path.read_text(encoding='utf-8')
    )
    assert manifest['requested_years'] == [2025, 2024]
    assert manifest['successful_years'] == [2025]
    assert manifest['runtime']['season_files'][0]['rows'] == 1
    assert manifest['runtime']['combined_file']['rows'] == 1
    assert manifest['cloud_raw']['season_files'][0]['rows'] == 1
