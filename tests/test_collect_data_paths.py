from pathlib import Path

from ffbayes.data_pipeline.collect_data import mirror_raw_dataset


def test_mirror_raw_dataset_copies_to_destination(tmp_path):
    source_file = tmp_path / 'source' / '2025season.csv'
    destination_dir = tmp_path / 'cloud' / 'data' / 'raw' / 'season_datasets'

    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text('season,data\n2025,ok\n', encoding='utf-8')

    destination_file = mirror_raw_dataset(source_file, destination_dir)

    assert destination_file == destination_dir / source_file.name
    assert destination_file.exists()
    assert destination_file.read_text(encoding='utf-8') == source_file.read_text(encoding='utf-8')
