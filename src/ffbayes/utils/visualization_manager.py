"""Cloud publication helpers for runtime artifacts and stable datasets."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _copy_file(source_path: Path, destination_path: Path) -> str:
    """Copy one file, creating parent directories as needed."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _copy_matching_files(
    source_dir: Path, destination_dir: Path, patterns: Iterable[str]
) -> list[str]:
    """Copy allow-listed files while preserving relative paths."""
    copied_files: list[str] = []
    if not source_dir.exists():
        return copied_files

    seen: set[Path] = set()
    for pattern in patterns:
        for file_path in sorted(source_dir.glob(pattern)):
            if not file_path.is_file() or file_path in seen:
                continue
            seen.add(file_path)
            relative_path = file_path.relative_to(source_dir)
            copied_files.append(_copy_file(file_path, destination_dir / relative_path))

    return copied_files


def _reset_publish_dir(path: Path) -> None:
    """Clear a publisher-owned output directory before repopulating it."""
    if path.exists():
        shutil.rmtree(path)


def _manifest_target_name(file_path: Path, current_year: int) -> str:
    return f'{file_path.stem}_{current_year}{file_path.suffix}'


def _sync_stable_data(current_year: int) -> dict[str, list[str]]:
    """Publish stable datasets into the top-level cloud data tree."""
    from ffbayes.utils.analysis_windows import get_analysis_years
    from ffbayes.utils.path_constants import (
        CLOUD_PROCESSED_DATA_DIR,
        CLOUD_RAW_DATA_DIR,
        PROCESSED_INPUTS_DIR,
        RAW_INPUTS_DIR,
    )

    copied_raw: list[str] = []
    copied_processed: list[str] = []
    analysis_years = get_analysis_years(current_year)
    year_range = f'{analysis_years[0]}-{analysis_years[-1]}'

    raw_manifest_dir = CLOUD_RAW_DATA_DIR / 'manifests'
    raw_season_dir = CLOUD_RAW_DATA_DIR / 'season_datasets'
    processed_combined_dir = CLOUD_PROCESSED_DATA_DIR / 'combined_datasets'
    processed_snake_dir = CLOUD_PROCESSED_DATA_DIR / 'snake_draft_datasets'
    processed_unified_dir = CLOUD_PROCESSED_DATA_DIR / 'unified_dataset'

    for target_dir in (CLOUD_RAW_DATA_DIR, CLOUD_PROCESSED_DATA_DIR):
        _reset_publish_dir(target_dir)

    for season_year in analysis_years:
        file_path = RAW_INPUTS_DIR / 'season_datasets' / f'{season_year}season.csv'
        if file_path.exists():
            copied_raw.append(_copy_file(file_path, raw_season_dir / file_path.name))

    for file_path in sorted(RAW_INPUTS_DIR.glob('*.json')):
        copied_raw.append(
            _copy_file(
                file_path,
                raw_manifest_dir / _manifest_target_name(file_path, current_year),
            )
        )

    for file_path in sorted(
        (RAW_INPUTS_DIR / 'manifests').glob(f'*_{current_year}.json')
    ):
        copied_raw.append(_copy_file(file_path, raw_manifest_dir / file_path.name))

    combined_file = (
        PROCESSED_INPUTS_DIR / 'combined_datasets' / f'{year_range}season_modern.csv'
    )
    if combined_file.exists():
        copied_processed.append(
            _copy_file(combined_file, processed_combined_dir / combined_file.name)
        )

    for file_path in sorted((PROCESSED_INPUTS_DIR / 'snake_draft_datasets').glob('*')):
        if file_path.is_file() and str(current_year) in file_path.name:
            copied_processed.append(
                _copy_file(file_path, processed_snake_dir / file_path.name)
            )

    unified_dir = PROCESSED_INPUTS_DIR / 'unified_dataset'
    unified_targets = {
        unified_dir / 'unified_dataset.csv': processed_unified_dir
        / f'unified_dataset_{current_year}.csv',
        unified_dir / 'unified_dataset.json': processed_unified_dir
        / f'unified_dataset_{current_year}.json',
    }
    for source_path, destination_path in unified_targets.items():
        if source_path.exists():
            copied_processed.append(_copy_file(source_path, destination_path))

    return {'raw_data_files': copied_raw, 'processed_data_files': copied_processed}


def _publish_analysis_snapshot(current_year: int) -> dict[str, Any]:
    """Publish a flat dated analysis snapshot."""
    from ffbayes.utils.path_constants import (
        get_cloud_analysis_snapshot_dir,
        get_dashboard_html_path,
        get_dashboard_payload_path,
        get_draft_strategy_dir,
        get_pre_draft_diagnostics_dir,
        get_vor_strategy_dir,
    )

    snapshot_dir = get_cloud_analysis_snapshot_dir()
    dashboard_dir = snapshot_dir / 'dashboard'
    draft_strategy_dir = snapshot_dir / 'draft_strategy'
    vor_dir = snapshot_dir / 'vor_strategy'
    diagnostics_dir = snapshot_dir / 'diagnostics'

    for target_dir in (dashboard_dir, draft_strategy_dir, vor_dir, diagnostics_dir):
        if target_dir.exists():
            shutil.rmtree(target_dir)

    copied_dashboard: list[str] = []
    copied_analysis: list[str] = []

    dashboard_html = get_dashboard_html_path(current_year)
    if dashboard_html.exists():
        copied_dashboard.append(
            _copy_file(dashboard_html, dashboard_dir / 'index.html')
        )

    dashboard_payload = get_dashboard_payload_path(current_year)
    if dashboard_payload.exists():
        copied_dashboard.append(
            _copy_file(dashboard_payload, dashboard_dir / 'dashboard_payload.json')
        )

    copied_analysis.extend(
        _copy_matching_files(
            get_draft_strategy_dir(current_year),
            draft_strategy_dir,
            (
                f'draft_board_{current_year}.xlsx',
                f'draft_board_{current_year}.json',
                'draft_decision_backtest_????-????.json',
                'historical_strategy_backtest_????-????.json',
                f'model_outputs/current_year_model_comparison_{current_year}.json',
                f'model_outputs/player_forecast/player_forecast_{current_year}.json',
                f'model_outputs/player_forecast/player_forecast_diagnostics_{current_year}.json',
                'model_outputs/player_forecast/player_forecast_validation_????-????.json',
            ),
        )
    )
    copied_analysis.extend(
        _copy_matching_files(
            get_vor_strategy_dir(current_year),
            vor_dir,
            (f'*_{current_year}.csv', f'*_{current_year}.xlsx'),
        )
    )
    copied_analysis.extend(
        _copy_matching_files(
            get_pre_draft_diagnostics_dir(current_year),
            diagnostics_dir,
            ('validation/player_forecast_validation_summary_????-????.json',),
        )
    )

    manifest = {
        'schema_version': 'cloud_publish_manifest_v3',
        'selection_policy': 'supported_pre_draft_publish_selection_v1',
        'excluded_artifact_families': [
            'hybrid_mc_bayesian',
            'legacy_runs',
            'retrospective_outputs',
            'sampled_bayes_diagnostics',
        ],
        'published_at': datetime.now().isoformat(timespec='seconds'),
        'season_year': int(current_year),
        'snapshot_dir': str(snapshot_dir),
        'dashboard_files': copied_dashboard,
        'analysis_files': copied_analysis,
    }
    manifest_path = snapshot_dir / 'publish_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    return {
        'snapshot_dir': snapshot_dir,
        'manifest_path': manifest_path,
        'dashboard_files': copied_dashboard,
        'analysis_files': copied_analysis,
        'copied_snapshot_files': copied_dashboard
        + copied_analysis
        + [str(manifest_path)],
    }


def manage_visualizations(
    current_year: int | None = None, phase: str | None = None
) -> dict[str, Any]:
    """Publish stable data and a flat dated analysis snapshot."""
    if current_year is None:
        current_year = datetime.now().year

    resolved_phase = (phase or 'pre_draft').lower()
    if resolved_phase != 'pre_draft':
        raise ValueError('Only pre_draft publication is supported')

    print(f'🖼️  Publishing {resolved_phase} artifacts for {current_year}...')

    data_results = _sync_stable_data(current_year)
    snapshot_results = _publish_analysis_snapshot(current_year)

    copied_files = (
        data_results['raw_data_files']
        + data_results['processed_data_files']
        + snapshot_results['copied_snapshot_files']
    )

    results = {
        'copied_files': copied_files,
        'synced_data_files': data_results['raw_data_files']
        + data_results['processed_data_files'],
        'raw_data_files': data_results['raw_data_files'],
        'processed_data_files': data_results['processed_data_files'],
        'published_snapshot_files': snapshot_results['copied_snapshot_files'],
        'published_dashboard_files': snapshot_results['dashboard_files'],
        'published_analysis_files': snapshot_results['analysis_files'],
        'snapshot_dir': str(snapshot_results['snapshot_dir']),
        'manifest_path': str(snapshot_results['manifest_path']),
        'readme_updated': False,
        'removed_old_files': 0,
        'year': current_year,
        'phase': resolved_phase,
    }

    print('✅ Cloud publication complete:')
    print(f'   📁 Synced {len(results["synced_data_files"])} stable data files')
    print(f'   📁 Published {len(results["published_snapshot_files"])} snapshot files')
    print(f'   🗂️  Snapshot: {results["snapshot_dir"]}')

    return results


if __name__ == '__main__':
    print(manage_visualizations())
