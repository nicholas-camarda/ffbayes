from __future__ import annotations

import pandas as pd

import ffbayes.draft_strategy.traditional_vor_draft as vor_draft
from ffbayes.draft_strategy import draft_decision_strategy as dds
from ffbayes.draft_strategy import draft_decision_system as dds_system


def test_main_reuses_existing_vor_snapshot(tmp_path, monkeypatch):
    input_csv = tmp_path / 'cached_vor.csv'
    output_dir = tmp_path / 'runtime' / 'snake_draft'
    organized_dir = tmp_path / 'runtime' / 'organized'
    output_dir.mkdir(parents=True, exist_ok=True)
    organized_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                'PLAYER': 'Alpha Player',
                'POS': 'RB',
                'AVG': 4.0,
                'FPTS': 240.0,
                'VOR': 55.0,
                'VALUERANK': 1,
            }
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        vor_draft,
        'load_config',
        lambda: {
            'ppr': 0.5,
            'top_rank': 120,
            'output_dir': str(output_dir),
            'organized_output_dir': str(organized_dir),
        },
    )
    monkeypatch.setattr(
        vor_draft,
        '_resolve_existing_vor_csv',
        lambda current_year, config: input_csv,
    )
    monkeypatch.setattr(
        vor_draft,
        'make_adp_df',
        lambda config: (_ for _ in ()).throw(
            AssertionError('ADP scraping should not run for cached VOR snapshots')
        ),
    )
    monkeypatch.setattr(
        vor_draft,
        'make_projection_df',
        lambda config: (_ for _ in ()).throw(
            AssertionError(
                'projection scraping should not run for cached VOR snapshots'
            )
        ),
    )

    result = vor_draft.main()

    assert result['reused_existing_snapshot'] is True
    assert (output_dir / 'snake-draft_ppr-0.5_vor_top-120_2026.csv').exists()
    assert (output_dir / 'DRAFTING STRATEGY -- snake-draft_ppr-0.5_vor_top-120_2026.xlsx').exists()
    assert (organized_dir / 'snake-draft_ppr-0.5_vor_top-120_2026.csv').exists()


def test_history_builder_handles_duplicate_canonical_aliases():
    history = pd.DataFrame(
        [
            {
                'Name': 'Alpha Player',
                'player_name': 'Alpha Player',
                'Position': 'RB',
                'position': 'RB',
                'Season': 2025,
                'G#': 1,
                'Tm': 'NYG',
                'FantPt': 12.0,
            }
        ]
    )

    features = dds._build_history_features(history)

    assert len(features) == 1
    assert features.loc[0, 'player_name'] == 'Alpha Player'
    assert features.loc[0, 'position'] == 'RB'


def test_normalize_player_frame_handles_duplicate_projection_aliases():
    frame = pd.DataFrame(
        [
            {
                'Name': 'Alpha Player',
                'player_name': 'Alpha Player',
                'Pos': 'RB',
                'position': 'RB',
                'FPTS': 240.0,
                'proj_points_mean': 245.0,
                'AVG': 4.0,
            }
        ]
    )

    normalized = dds_system.normalize_player_frame(frame)

    assert len(normalized) == 1
    assert normalized.loc[0, 'player_name'] == 'Alpha Player'
    assert normalized.loc[0, 'position'] == 'RB'
    assert normalized.loc[0, 'proj_points_mean'] == 245.0


def test_summarize_freshness_manifest_supports_collection_source_manifest_shape():
    manifest = {
        'requested_years': [2021, 2022, 2023, 2024, 2025],
        'successful_years': [2021, 2022, 2023, 2024, 2025],
        'source_manifest': {
            'freshness_status': 'fresh',
            'override_used': False,
            'latest_expected_year': 2025,
            'latest_found_year': 2025,
            'found_years': [2021, 2022, 2023, 2024, 2025],
            'warnings': [],
            'is_fresh': True,
        },
    }

    summary = dds._summarize_freshness_manifest(manifest, 'collection_inputs')

    assert summary is not None
    assert summary['status'] == 'fresh'
    assert summary['override_used'] is False
    assert summary['latest_expected_year'] == 2025
    assert summary['latest_found_year'] == 2025
    assert summary['found_years'] == [2021, 2022, 2023, 2024, 2025]
