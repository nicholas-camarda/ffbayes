from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

import ffbayes.data_pipeline.collect_data as collect_data
from ffbayes.data_pipeline.collect_data import (
    build_collection_manifest,
    collect_nfl_data,
    write_collection_manifest,
)
from ffbayes.data_pipeline.collect_data import (
    main as collect_main,
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


class _DummyProgressMonitor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start_timer(self):
        return None

    @contextmanager
    def monitor(self, *args, **kwargs):
        yield self


def _make_valid_merged_frame(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'player_id': f'p{year}',
                'player_display_name': 'Alpha Player',
                'position': 'RB',
                'recent_team': 'NYG',
                'season': year,
                'week': 1,
                'fantasy_points': 12.5,
                'fantasy_points_ppr': 15.0,
                'gameday': f'{year}-09-01',
                'home_team': 'NYG',
                'away_team': 'DAL',
            }
        ]
    )


def test_collect_nfl_data_raises_when_latest_season_missing(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(collect_data, 'CURRENT_YEAR', 2026)
    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(collect_data, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(collect_data, 'ProgressMonitor', _DummyProgressMonitor)
    monkeypatch.setattr(
        collect_data,
        'check_data_availability',
        lambda year: (year != 2025, 'missing latest season' if year == 2025 else 1),
    )
    monkeypatch.setattr(
        collect_data,
        'create_dataset',
        lambda year: _make_valid_merged_frame(year),
    )

    with pytest.raises(RuntimeError, match='Latest expected season 2025 is missing'):
        collect_nfl_data([2021, 2022, 2023, 2024, 2025], allow_stale_latest=False)

    assert not (raw_dir / 'freshness_manifest.json').exists()


def test_collect_nfl_data_allows_stale_season_and_writes_degraded_manifest(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(collect_data, 'CURRENT_YEAR', 2026)
    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(collect_data, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(collect_data, 'ProgressMonitor', _DummyProgressMonitor)
    monkeypatch.setattr(
        collect_data,
        'check_data_availability',
        lambda year: (year != 2025, 'missing latest season' if year == 2025 else 1),
    )
    monkeypatch.setattr(
        collect_data,
        'create_dataset',
        lambda year: _make_valid_merged_frame(year),
    )

    successful_years = collect_nfl_data(
        [2021, 2022, 2023, 2024, 2025], allow_stale_latest=True
    )

    freshness_manifest_path = raw_dir / 'freshness_manifest.json'
    manifest = json.loads(freshness_manifest_path.read_text(encoding='utf-8'))

    assert successful_years == [2021, 2022, 2023, 2024]
    assert freshness_manifest_path.exists()
    assert manifest['analysis_window']['freshness_status'] == 'degraded'
    assert manifest['analysis_window']['latest_expected_year'] == 2025
    assert manifest['analysis_window']['found_years'] == [2021, 2022, 2023, 2024]
    assert any('Missing latest expected season' in warning for warning in manifest['warnings'])


def test_collect_main_combines_only_real_season_files(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    combined_dir = tmp_path / 'combined_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    combine_calls: list[tuple[list[str], list[int]]] = []

    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    class DummyInterface:
        def __init__(self):
            self.logger = DummyLogger()
            self.handle_errors_results = []

        def setup_argument_parser(self):
            return self

        def add_data_arguments(self, parser):
            return parser

        def setup_logging(self, args):
            self.logger = DummyLogger()
            return self.logger

        def parse_years(self, text):
            return [int(part) for part in text.split(',') if part]

        def handle_errors(self, func, *args, **kwargs):
            result = func(*args, **kwargs)
            self.handle_errors_results.append((func.__name__, result))
            return result

        def log_completion(self, message):
            return None

    def fake_create_standardized_interface(*args, **kwargs):
        return dummy_interface

    def fake_combine(directory_path, output_directory_path, years_to_process):
        combine_calls.append(
            (
                sorted(file_path.name for file_path in directory_path.glob('*season.csv')),
                list(years_to_process),
            )
        )
        output_directory_path.mkdir(parents=True, exist_ok=True)
        (output_directory_path / 'combined_data.csv').write_text(
            'Season,Name\n2024,Alpha Player\n', encoding='utf-8'
        )
        return pd.DataFrame({'Season': [2024], 'Name': ['Alpha Player']})

    dummy_interface = DummyInterface()

    monkeypatch.setattr(collect_data, 'CURRENT_YEAR', 2026)
    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(collect_data, 'COMBINED_DATASETS_DIR', combined_dir)
    monkeypatch.setattr(collect_data, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(collect_data, 'ProgressMonitor', _DummyProgressMonitor)
    monkeypatch.setattr(
        collect_data,
        'check_data_availability',
        lambda year: (year != 2025, 'missing latest season' if year == 2025 else 1),
    )
    monkeypatch.setattr(
        collect_data,
        'create_dataset',
        lambda year: _make_valid_merged_frame(year),
    )
    monkeypatch.setattr(collect_data, 'combine_datasets', fake_combine)
    monkeypatch.setattr(
        'ffbayes.utils.script_interface.create_standardized_interface',
        fake_create_standardized_interface,
    )

    result = collect_main(
        args=SimpleNamespace(
            years='2024,2025',
            quick_test=False,
            force_refresh=False,
            allow_stale_season=True,
            verbose=False,
            quiet=False,
            log_level='INFO',
        )
    )

    assert result == [2024]
    assert combine_calls == [(['2024season.csv'], [2024])]
    assert (season_dir / '2024season.csv').exists()
    assert not (season_dir / '2025season.csv').exists()
    assert (combined_dir / 'combined_data.csv').exists()


def test_collect_nfl_data_refreshes_existing_files_with_backend_only_monkeypatch(
    tmp_path, monkeypatch
):
    season_dir = tmp_path / 'season_datasets'
    combined_dir = tmp_path / 'combined_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    stale_file = season_dir / '2025season.csv'
    stale_file.write_text(
        'G#,Date,Tm,Away,Opp,FantPt,FantPtPPR,Name,PlayerID,Position,Season,'
        'GameInjuryStatus,PracticeInjuryStatus,is_home\n'
        '1,2025-09-01,NYG,DAL,DAL,8,9,Stale Player,p-stale,RB,2025,,,1\n',
        encoding='utf-8',
    )

    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    class DummyInterface:
        def setup_argument_parser(self):
            return self

        def add_data_arguments(self, parser):
            return parser

        def setup_logging(self, args):
            return DummyLogger()

        def handle_errors(self, func, *args, **kwargs):
            return func(*args, **kwargs)

        def log_completion(self, message):
            return None

    fake_backend = SimpleNamespace(
        PLAYER_STATS_REQUIRED_COLUMNS=(
            'player_id',
            'player_display_name',
            'position',
            'recent_team',
            'season',
            'week',
            'season_type',
            'fantasy_points',
            'fantasy_points_ppr',
        ),
        NFLVerseBackendError=Exception,
        load_weekly_player_stats=lambda seasons: pd.DataFrame(
            [
                {
                    'player_id': 'p1',
                    'player_display_name': 'Alpha Player',
                    'position': 'RB',
                    'recent_team': 'NYG',
                    'season': 2025,
                    'week': 1,
                    'season_type': 'REG',
                    'fantasy_points': 12.5,
                    'fantasy_points_ppr': 15.0,
                    'game_injury_report_status': 'Questionable',
                    'practice_injury_report_status': 'Limited',
                }
            ]
        ),
        load_player_stats=lambda seasons, summary_level: pd.DataFrame(
            []
        ),
        load_schedules=lambda seasons: pd.DataFrame(
            [
                {
                    'game_id': '2025_01_NYG_DAL',
                    'week': 1,
                    'season': 2025,
                    'gameday': '2025-09-01',
                    'home_team': 'NYG',
                    'away_team': 'DAL',
                    'home_score': 24,
                    'away_score': 17,
                    'game_type': 'REG',
                }
            ]
        ),
        load_weekly_defense_stats=lambda seasons: pd.DataFrame(),
    )

    monkeypatch.setattr(collect_data, 'CURRENT_YEAR', 2026)
    monkeypatch.setattr(collect_data, 'SEASON_DATASETS_DIR', season_dir)
    monkeypatch.setattr(collect_data, 'COMBINED_DATASETS_DIR', combined_dir)
    monkeypatch.setattr(collect_data, 'RAW_DATA_DIR', raw_dir)
    monkeypatch.setattr(collect_data, 'ProgressMonitor', _DummyProgressMonitor)
    monkeypatch.setattr(collect_data, 'nflverse_backend', fake_backend)
    monkeypatch.setattr(
        'ffbayes.utils.script_interface.create_standardized_interface',
        lambda *args, **kwargs: DummyInterface(),
    )

    successful_years = collect_data.collect_nfl_data([2025], allow_stale_latest=False)

    season_csv = stale_file.read_text(encoding='utf-8')

    assert successful_years == [2025]
    assert 'Stale Player' in season_csv
    assert 'Alpha Player' not in season_csv
