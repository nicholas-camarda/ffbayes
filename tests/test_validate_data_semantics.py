from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

import ffbayes.data_pipeline.validate_data as validate_data


class _DummyProgressMonitor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start_timer(self):
        return None

    @contextmanager
    def monitor(self, *args, **kwargs):
        yield self


def test_validation_normalizes_expected_positions_and_dates(tmp_path, monkeypatch):
    season_dir = tmp_path / 'season_datasets'
    raw_dir = tmp_path / 'raw'
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                'G#': 1,
                'Date': '2025-09-01',
                'Tm': 'NYG',
                'Away': 'DAL',
                'Opp': 'DAL',
                'FantPt': 14.2,
                'FantPtPPR': 17.1,
                'Name': 'Alpha Player',
                'PlayerID': 'p1',
                'Position': 'DEF',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 2,
                'Date': '2025-09-08',
                'Tm': 'NYG',
                'Away': 'PHI',
                'Opp': 'PHI',
                'FantPt': 9.8,
                'FantPtPPR': 11.2,
                'Name': 'Beta Player',
                'PlayerID': 'p2',
                'Position': 'SAF',
                'Season': 2025,
                'is_home': 0,
            },
            {
                'G#': 3,
                'Date': '2025-09-15',
                'Tm': 'NYG',
                'Away': 'WAS',
                'Opp': 'WAS',
                'FantPt': 6.5,
                'FantPtPPR': 7.1,
                'Name': 'Gamma Player',
                'PlayerID': 'p3',
                'Position': 'LS',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 4,
                'Date': '2025-09-22',
                'Tm': 'NYG',
                'Away': 'MIA',
                'Opp': 'MIA',
                'FantPt': 3.0,
                'FantPtPPR': 3.8,
                'Name': 'Delta Player',
                'PlayerID': 'p4',
                'Position': 'DL',
                'Season': 2025,
                'is_home': 0,
            },
            {
                'G#': 5,
                'Date': '2025-09-29',
                'Tm': 'NYG',
                'Away': 'BUF',
                'Opp': 'BUF',
                'FantPt': 1.5,
                'FantPtPPR': 2.1,
                'Name': 'Epsilon Player',
                'PlayerID': 'p5',
                'Position': 'LB',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 6,
                'Date': '2025-10-06',
                'Tm': 'NYG',
                'Away': 'DAL',
                'Opp': 'DAL',
                'FantPt': 100.0,
                'FantPtPPR': 105.0,
                'Name': 'Alpha Player',
                'PlayerID': 'p1',
                'Position': 'DEF',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 7,
                'Date': '2025-10-13',
                'Tm': 'NYG',
                'Away': 'PHI',
                'Opp': 'PHI',
                'FantPt': 99.0,
                'FantPtPPR': 102.0,
                'Name': 'Beta Player',
                'PlayerID': 'p2',
                'Position': 'SAF',
                'Season': 2025,
                'is_home': 0,
            },
            {
                'G#': 8,
                'Date': '2025-10-20',
                'Tm': 'NYG',
                'Away': 'WAS',
                'Opp': 'WAS',
                'FantPt': 98.0,
                'FantPtPPR': 101.0,
                'Name': 'Gamma Player',
                'PlayerID': 'p3',
                'Position': 'LS',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 9,
                'Date': '2025-10-27',
                'Tm': 'NYG',
                'Away': 'MIA',
                'Opp': 'MIA',
                'FantPt': 97.0,
                'FantPtPPR': 100.0,
                'Name': 'Delta Player',
                'PlayerID': 'p4',
                'Position': 'DL',
                'Season': 2025,
                'is_home': 0,
            },
            {
                'G#': 10,
                'Date': '2025-11-03',
                'Tm': 'NYG',
                'Away': 'BUF',
                'Opp': 'BUF',
                'FantPt': 96.0,
                'FantPtPPR': 99.0,
                'Name': 'Epsilon Player',
                'PlayerID': 'p5',
                'Position': 'LB',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 11,
                'Date': '2025-11-10',
                'Tm': 'NYG',
                'Away': 'DAL',
                'Opp': 'DAL',
                'FantPt': 95.0,
                'FantPtPPR': 98.0,
                'Name': 'Alpha Player',
                'PlayerID': 'p1',
                'Position': 'DEF',
                'Season': 2025,
                'is_home': 1,
            },
            {
                'G#': 12,
                'Date': '2025-11-17',
                'Tm': 'NYG',
                'Away': 'PHI',
                'Opp': 'PHI',
                'FantPt': 94.0,
                'FantPtPPR': 97.0,
                'Name': 'Beta Player',
                'PlayerID': 'p2',
                'Position': 'SAF',
                'Season': 2025,
                'is_home': 0,
            },
        ]
    )
    season_file = season_dir / '2025season.csv'
    frame.to_csv(season_file, index=False)

    monkeypatch.setattr(validate_data, 'SEASON_DATASETS_DIR', season_dir, raising=False)
    monkeypatch.setattr(validate_data, 'RAW_DATA_DIR', raw_dir, raising=False)
    monkeypatch.setattr(validate_data, 'ProgressMonitor', _DummyProgressMonitor, raising=False)

    results = validate_data.validate_data_quality()

    assert results['blockers'] == []
    assert not any(
        'Invalid positions' in issue for issue in results['data_consistency']
    )
    assert not any(
        'fantasy point outliers' in issue for issue in results['outlier_detection']
    )
