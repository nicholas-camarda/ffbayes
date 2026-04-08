#!/usr/bin/env python3
"""Outcome-grounded retrospective analysis for finalized draft artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ffbayes.utils.path_constants import (
    get_draft_retrospective_html_path,
    get_draft_retrospective_json_path,
    get_finalized_drafts_dir,
    get_draft_strategy_dir,
    get_unified_dataset_csv_path,
)

FINALIZED_SCHEMA_VERSION = 'finalized_draft_v1'
RETROSPECTIVE_SCHEMA_VERSION = 'draft_retrospective_v1'
FINALIZED_BUNDLE_PATTERN = re.compile(
    r'^ffbayes_finalized_(?:draft|summary)_(?P<year>\d{4})_'
)


def _safe_lower(value: Any) -> str:
    return str(value or '').strip().lower()


def _coerce_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_season_year(
    payload: dict[str, Any], source_path: Path, fallback_year: int | None = None
) -> int:
    for candidate in (
        payload.get('season_year'),
        str(payload.get('source_payload_generated_at') or '')[:4],
        str(payload.get('exported_at') or '')[:4],
    ):
        year = _coerce_int(candidate)
        if year:
            return year

    digits = ''.join(ch for ch in source_path.stem if ch.isdigit())
    if len(digits) >= 4:
        year = _coerce_int(digits[:4])
        if year:
            return year

    if fallback_year is not None:
        return int(fallback_year)
    raise ValueError(f'Could not infer season year from finalized draft {source_path}')


def _determine_scoring_mode(payload: dict[str, Any]) -> str:
    league_settings = payload.get('league_settings') or {}
    preset = _safe_lower(league_settings.get('scoring_preset'))
    scoring_type = _safe_lower(league_settings.get('scoring_type'))
    ppr_value = _coerce_float(league_settings.get('ppr_value'))
    if preset in {'ppr', 'full_ppr'} or scoring_type == 'ppr' or ppr_value == 1.0:
        return 'ppr'
    if preset in {'half_ppr', 'half-ppr'} or ppr_value == 0.5:
        return 'half_ppr'
    return 'standard'


def _outcome_column_name(scoring_mode: str) -> str:
    if scoring_mode == 'ppr':
        return 'actual_points_ppr'
    if scoring_mode == 'half_ppr':
        return 'actual_points_half_ppr'
    return 'actual_points_standard'


def _normalize_outcomes_table(frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _safe_lower(column)
        if normalized in {'season', 'year'}:
            rename_map[column] = 'season_year'
        elif normalized in {'player_name', 'name'}:
            rename_map[column] = 'player_name'
        elif normalized in {'position', 'pos'}:
            rename_map[column] = 'position'
        elif normalized in {'fantasy_points', 'fantpt', 'actual_points'}:
            rename_map[column] = 'actual_points_standard'
        elif normalized in {'fantasy_points_ppr', 'fantptppr', 'actual_points_ppr'}:
            rename_map[column] = 'actual_points_ppr'
        elif normalized in {'rec', 'receptions'}:
            rename_map[column] = 'receptions'

    normalized = frame.rename(columns=rename_map).copy()
    required = {'season_year', 'player_name'}
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(
            f'Outcome dataset at {source_path} is missing required columns: '
            f'{", ".join(sorted(missing))}'
        )

    if 'position' not in normalized.columns:
        normalized['position'] = ''

    normalized['season_year'] = pd.to_numeric(
        normalized['season_year'], errors='coerce'
    ).astype('Int64')
    normalized = normalized[normalized['season_year'].notna()].copy()
    normalized['season_year'] = normalized['season_year'].astype(int)

    normalized['player_name'] = normalized['player_name'].astype(str)
    normalized['position'] = normalized['position'].astype(str).str.upper()
    normalized['player_key'] = normalized['player_name'].map(_safe_lower)

    for column in (
        'actual_points_standard',
        'actual_points_ppr',
        'receptions',
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors='coerce')

    if 'actual_points_standard' not in normalized.columns:
        raise ValueError(
            f'Outcome dataset at {source_path} must provide standard fantasy points.'
        )

    if 'actual_points_ppr' not in normalized.columns:
        if 'receptions' in normalized.columns:
            normalized['actual_points_ppr'] = (
                normalized['actual_points_standard'].fillna(0.0)
                + normalized['receptions'].fillna(0.0)
            )
        else:
            normalized['actual_points_ppr'] = normalized['actual_points_standard']

    normalized['actual_points_half_ppr'] = (
        normalized['actual_points_standard'].fillna(0.0)
        + 0.5
        * (
            normalized['actual_points_ppr'].fillna(0.0)
            - normalized['actual_points_standard'].fillna(0.0)
        )
    )

    grouped = (
        normalized.sort_values(
            ['season_year', 'player_key', 'position', 'actual_points_ppr'],
            ascending=[True, True, True, False],
        )
        .groupby(['season_year', 'player_key', 'position'], as_index=False)
        .agg(
            player_name=('player_name', 'first'),
            actual_points_standard=('actual_points_standard', 'max'),
            actual_points_ppr=('actual_points_ppr', 'max'),
            actual_points_half_ppr=('actual_points_half_ppr', 'max'),
        )
    )
    return grouped


def load_realized_outcomes(outcomes_path: Path | str) -> pd.DataFrame:
    resolved = Path(outcomes_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f'Realized season outcome dataset not found at {resolved}.'
        )
    suffix = resolved.suffix.lower()
    if suffix == '.csv':
        raw = pd.read_csv(resolved)
    elif suffix == '.json':
        raw = pd.read_json(resolved)
    elif suffix in {'.xlsx', '.xls'}:
        raw = pd.read_excel(resolved)
    else:
        raise ValueError(
            f'Unsupported outcome dataset format for {resolved}. '
            'Use CSV, JSON, or Excel.'
        )
    return _normalize_outcomes_table(raw, resolved)


def _validate_finalized_payload(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if payload.get('schema_version') != FINALIZED_SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported finalized draft schema at {source_path}: '
            f'{payload.get("schema_version")!r}'
        )
    required = ['drafted_players', 'summary_metrics']
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f'Finalized draft {source_path} is missing required keys: '
            f'{", ".join(missing)}'
        )
    return payload


def load_finalized_payload(
    source_path: Path | str, fallback_year: int | None = None
) -> tuple[dict[str, Any], int]:
    resolved = Path(source_path)
    if not resolved.exists():
        raise FileNotFoundError(f'Finalized draft not found at {resolved}.')
    payload = json.loads(resolved.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Finalized draft at {resolved} must be a JSON object.')
    payload = _validate_finalized_payload(payload, resolved)
    season_year = _infer_season_year(payload, resolved, fallback_year=fallback_year)
    return payload, season_year


def _infer_finalized_artifact_year(
    source_path: Path | str, fallback_year: int | None = None
) -> int:
    resolved = Path(source_path)
    match = FINALIZED_BUNDLE_PATTERN.match(resolved.name)
    if match:
        return int(match.group('year'))
    if resolved.suffix.lower() == '.json':
        try:
            _, season_year = load_finalized_payload(resolved, fallback_year=fallback_year)
            return season_year
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    if fallback_year is not None:
        return int(fallback_year)
    raise ValueError(
        f'Could not infer season year for finalized artifact {resolved}. '
        'Pass `--year` when importing artifacts with non-standard filenames.'
    )


def import_finalized_artifacts(
    import_paths: list[Path | str],
    *,
    year: int | None = None,
    move_files: bool = False,
) -> dict[str, Any]:
    if not import_paths:
        return {
            'status': 'skipped',
            'imported_paths': [],
            'imported_json_paths': [],
            'season_years': [],
        }

    imported_paths: list[Path] = []
    imported_json_paths: list[Path] = []
    season_years: set[int] = set()

    for source in [Path(path) for path in import_paths]:
        if not source.exists():
            raise FileNotFoundError(f'Finalized artifact not found at {source}.')
        if source.is_dir():
            raise ValueError(
                f'Finalized artifact path must be a file, got directory: {source}'
            )

        season_year = _infer_finalized_artifact_year(source, fallback_year=year)
        destination = get_finalized_drafts_dir(season_year) / source.name
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if move_files:
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(source, destination)
        imported_paths.append(destination)
        if (
            destination.suffix.lower() == '.json'
            and destination.name.startswith('ffbayes_finalized_draft_')
        ):
            imported_json_paths.append(destination)
        season_years.add(season_year)

    return {
        'status': 'moved' if move_files else 'copied',
        'imported_paths': imported_paths,
        'imported_json_paths': imported_json_paths,
        'season_years': sorted(season_years),
    }


def _match_outcomes(
    drafted: pd.DataFrame, season_outcomes: pd.DataFrame, actual_column: str
) -> pd.DataFrame:
    drafted = drafted.copy()
    drafted['player_key'] = drafted['player_name'].map(_safe_lower)
    drafted['position_key'] = drafted['position'].astype(str).str.upper()

    exact = drafted.merge(
        season_outcomes[
            ['player_key', 'position', 'player_name', actual_column]
        ].rename(columns={'position': 'position_key', 'player_name': 'matched_player_name'}),
        on=['player_key', 'position_key'],
        how='left',
    )
    exact = exact.rename(columns={actual_column: 'actual_points'})

    unmatched = exact['actual_points'].isna()
    if unmatched.any():
        by_name = (
            season_outcomes.groupby('player_key', as_index=False)
            .agg(
                matched_player_name=('player_name', 'first'),
                actual_points=(actual_column, 'max'),
                outcome_position_count=('position', 'nunique'),
            )
        )
        fallback = exact.loc[unmatched, ['player_key']].merge(
            by_name, on='player_key', how='left'
        )
        exact.loc[unmatched, 'matched_player_name'] = fallback['matched_player_name'].to_numpy()
        exact.loc[unmatched, 'actual_points'] = fallback['actual_points'].to_numpy()

    exact['matched_outcome'] = exact['actual_points'].notna()
    exact['projection_delta'] = (
        pd.to_numeric(exact.get('actual_points'), errors='coerce')
        - pd.to_numeric(exact.get('proj_points_mean'), errors='coerce')
    )
    exact['beat_projection'] = exact['projection_delta'] >= 0
    return exact


def _bucket_fragility(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return 'unknown'
    if numeric >= 0.66:
        return 'high'
    if numeric >= 0.33:
        return 'medium'
    return 'low'


def _summarize_group(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    if frame.empty or group_column not in frame.columns:
        return []
    grouped = (
        frame.groupby(group_column, dropna=False)
        .agg(
            player_count=('player_name', 'count'),
            matched_outcome_count=('matched_outcome', 'sum'),
            expected_points=('proj_points_mean', 'sum'),
            actual_points=('actual_points', 'sum'),
            mean_delta=('projection_delta', 'mean'),
            hit_rate=('beat_projection', 'mean'),
        )
        .reset_index()
        .sort_values(group_column)
    )
    records = grouped.to_dict(orient='records')
    for record in records:
        record['hit_rate'] = (
            float(record['hit_rate']) if pd.notna(record['hit_rate']) else None
        )
    return records


def _pick_receipt_audit(
    pick_receipts: list[dict[str, Any]],
    season_outcomes: pd.DataFrame,
    actual_column: str,
) -> dict[str, Any]:
    if not pick_receipts:
        return {
            'status': 'unavailable',
            'reason': 'No pick receipts were available in the finalized draft artifact.',
        }

    rows = []
    for receipt in pick_receipts:
        receipt_row = dict(receipt)
        receipt_row['player_key'] = _safe_lower(receipt.get('player_name'))
        receipt_row['top_recommendation_key'] = _safe_lower(
            receipt.get('top_recommendation')
        )
        wait_candidate = receipt.get('top_wait_candidate') or {}
        receipt_row['top_wait_candidate_key'] = _safe_lower(
            wait_candidate.get('player_name')
        )
        rows.append(receipt_row)
    receipts = pd.DataFrame(rows)
    if 'followed_model' not in receipts.columns:
        receipts['followed_model'] = None

    outcomes_by_name = (
        season_outcomes.groupby('player_key', as_index=False)
        .agg(actual_points=(actual_column, 'max'))
        .rename(columns={'actual_points': 'season_actual_points'})
    )
    receipts = receipts.merge(
        outcomes_by_name.rename(columns={'player_key': 'player_key'}),
        on='player_key',
        how='left',
    ).rename(columns={'season_actual_points': 'chosen_actual_points'})
    receipts = receipts.merge(
        outcomes_by_name.rename(
            columns={
                'player_key': 'top_recommendation_key',
                'season_actual_points': 'recommended_actual_points',
            }
        ),
        on='top_recommendation_key',
        how='left',
    )
    receipts = receipts.merge(
        outcomes_by_name.rename(
            columns={
                'player_key': 'top_wait_candidate_key',
                'season_actual_points': 'wait_candidate_actual_points',
            }
        ),
        on='top_wait_candidate_key',
        how='left',
    )

    followed_known = receipts['followed_model'].map(
        lambda value: value if isinstance(value, bool) else None
    )
    known_mask = followed_known.notna()
    known_followed_bool = followed_known[known_mask].astype(bool) if known_mask.any() else pd.Series(dtype=bool)
    status = 'available'
    warnings: list[str] = []
    if not known_mask.any():
        status = 'degraded'
        warnings.append('No pick receipts included followed_model metadata.')
    elif not known_mask.all():
        status = 'degraded'
        warnings.append('Some pick receipts were missing followed_model metadata.')

    pivot_mask = known_mask & (~followed_known.where(known_mask, False).astype(bool))
    follow_count = int(known_followed_bool.sum()) if known_mask.any() else None
    pivot_count = int(pivot_mask.sum()) if known_mask.any() else None
    follow_rate = (
        float(known_followed_bool.sum() / known_mask.sum())
        if known_mask.any()
        else None
    )

    recommendation_delta = None
    if pivot_mask.any():
        pivot_deltas = (
            pd.to_numeric(receipts.loc[pivot_mask, 'chosen_actual_points'], errors='coerce')
            - pd.to_numeric(receipts.loc[pivot_mask, 'recommended_actual_points'], errors='coerce')
        ).dropna()
        if not pivot_deltas.empty:
            recommendation_delta = float(pivot_deltas.mean())

    wait_status = 'unavailable'
    mean_wait_delta = None
    wait_event_count = 0
    if 'top_wait_candidate_key' in receipts.columns and receipts['top_wait_candidate_key'].astype(bool).any():
        wait_deltas = (
            pd.to_numeric(receipts['chosen_actual_points'], errors='coerce')
            - pd.to_numeric(receipts['wait_candidate_actual_points'], errors='coerce')
        ).dropna()
        wait_event_count = int(wait_deltas.shape[0])
        if wait_event_count:
            mean_wait_delta = float(wait_deltas.mean())
            wait_status = 'available'
        else:
            wait_status = 'degraded'
            warnings.append(
                'Wait candidates were captured, but realized outcome matching was incomplete.'
            )

    result = {
        'status': status,
        'warnings': warnings,
        'pick_count': int(len(receipts)),
        'known_follow_decision_count': int(known_mask.sum()),
        'follow_rate': follow_rate,
        'follow_count': follow_count,
        'pivot_count': pivot_count,
        'mean_pivot_actual_delta_vs_recommendation': recommendation_delta,
        'wait_policy_calibration': {
            'status': wait_status,
            'comparison_count': wait_event_count,
            'mean_actual_delta_vs_top_wait_candidate': mean_wait_delta,
        },
    }
    return result


def _season_report(
    payload: dict[str, Any],
    source_path: Path,
    season_year: int,
    outcomes: pd.DataFrame,
    outcomes_path: Path,
) -> dict[str, Any]:
    scoring_mode = _determine_scoring_mode(payload)
    actual_column = _outcome_column_name(scoring_mode)
    season_outcomes = outcomes[outcomes['season_year'] == season_year].copy()
    if season_outcomes.empty:
        raise FileNotFoundError(
            f'No realized season outcomes were available for season {season_year} in '
            f'{outcomes_path}.'
        )

    drafted = pd.DataFrame(payload.get('drafted_players') or [])
    if drafted.empty:
        raise ValueError(f'Finalized draft {source_path} did not contain drafted players.')
    drafted['player_name'] = drafted.get('player_name', pd.Series(dtype=str)).astype(str)
    drafted['position'] = drafted.get('position', pd.Series(dtype=str)).astype(str).str.upper()
    if 'lineup_slot' not in drafted.columns:
        drafted['lineup_slot'] = ''
    drafted['proj_points_mean'] = pd.to_numeric(
        drafted.get('proj_points_mean'), errors='coerce'
    )
    drafted['fragility_bucket'] = drafted.get('fragility_score', pd.Series(dtype=float)).map(
        _bucket_fragility
    )
    drafted['value_indicator'] = drafted.get(
        'value_indicator', pd.Series(['unknown'] * len(drafted))
    ).fillna('unknown')

    matched = _match_outcomes(drafted, season_outcomes, actual_column)
    matched_outcome_count = int(matched['matched_outcome'].sum())
    if matched_outcome_count == 0:
        raise ValueError(
            f'Outcome dataset {outcomes_path} did not match any drafted players for '
            f'season {season_year}.'
        )

    starters_payload = payload.get('starters') or []
    starters = pd.DataFrame(starters_payload)
    if starters.empty:
        lineup_slot = matched.get('lineup_slot', pd.Series('', index=matched.index)).astype(str)
        starters = matched[~lineup_slot.str.contains('BENCH', case=False, na=False)].copy()
    else:
        starters['player_name'] = starters.get('player_name', pd.Series(dtype=str)).astype(str)
        starters = starters.merge(
            matched[['player_name', 'actual_points', 'projection_delta', 'beat_projection']],
            on='player_name',
            how='left',
        )
    starters['proj_points_mean'] = pd.to_numeric(
        starters.get('proj_points_mean'), errors='coerce'
    )

    summary_metrics = payload.get('summary_metrics') or {}
    expected_starter_points = _coerce_float(summary_metrics.get('starter_lineup_mean'))
    if expected_starter_points is None:
        expected_starter_points = float(starters['proj_points_mean'].fillna(0.0).sum())
    expected_full_roster_points = float(matched['proj_points_mean'].fillna(0.0).sum())
    actual_starter_points = float(pd.to_numeric(starters.get('actual_points'), errors='coerce').fillna(0.0).sum())
    actual_full_roster_points = float(pd.to_numeric(matched.get('actual_points'), errors='coerce').fillna(0.0).sum())

    warnings: list[str] = []
    status = 'available'
    missing_outcomes = int((~matched['matched_outcome']).sum())
    if missing_outcomes:
        status = 'degraded'
        warnings.append(
            f'{missing_outcomes} drafted players could not be matched to realized outcomes.'
        )

    outcome_metrics = {
        'status': status,
        'scoring_mode': scoring_mode,
        'matched_outcome_count': matched_outcome_count,
        'missing_outcome_count': missing_outcomes,
        'expected_starter_points': expected_starter_points,
        'actual_starter_points': actual_starter_points,
        'starter_delta': actual_starter_points - expected_starter_points,
        'expected_full_roster_points': expected_full_roster_points,
        'actual_full_roster_points': actual_full_roster_points,
        'full_roster_delta': actual_full_roster_points - expected_full_roster_points,
        'drafted_player_hit_rate': (
            float(matched['beat_projection'].dropna().mean())
            if matched['beat_projection'].dropna().shape[0]
            else None
        ),
        'starter_hit_rate': (
            float(starters['beat_projection'].dropna().mean())
            if starters['beat_projection'].dropna().shape[0]
            else None
        ),
    }

    position_summary = _summarize_group(matched, 'position')
    fragility_summary = _summarize_group(matched, 'fragility_bucket')
    value_indicator_summary = _summarize_group(matched, 'value_indicator')
    audit_context = _pick_receipt_audit(
        list(payload.get('pick_receipts') or []),
        season_outcomes,
        actual_column,
    )
    warnings.extend(audit_context.get('warnings') or [])

    drafted_rows = (
        matched[
            [
                'player_name',
                'position',
                'lineup_slot',
                'proj_points_mean',
                'actual_points',
                'projection_delta',
                'beat_projection',
                'fragility_score',
                'value_indicator',
                'matched_outcome',
            ]
        ]
        .sort_values(['position', 'player_name'])
        .to_dict(orient='records')
    )

    return {
        'season_year': season_year,
        'status': status,
        'warnings': list(dict.fromkeys(warnings)),
        'source_paths': {
            'finalized_draft': str(source_path),
            'realized_outcomes': str(outcomes_path),
        },
        'exported_at': payload.get('exported_at'),
        'source_payload_generated_at': payload.get('source_payload_generated_at'),
        'outcome_metrics': outcome_metrics,
        'position_summary': position_summary,
        'fragility_summary': fragility_summary,
        'value_indicator_summary': value_indicator_summary,
        'audit_context': audit_context,
        'drafted_players': drafted_rows,
    }


def _rollup_reports(season_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not season_reports:
        return {'season_count': 0}

    season_count = len(season_reports)
    starter_deltas = [
        report['outcome_metrics']['starter_delta']
        for report in season_reports
        if report.get('outcome_metrics')
    ]
    full_deltas = [
        report['outcome_metrics']['full_roster_delta']
        for report in season_reports
        if report.get('outcome_metrics')
    ]
    hit_rates = [
        report['outcome_metrics']['drafted_player_hit_rate']
        for report in season_reports
        if report['outcome_metrics'].get('drafted_player_hit_rate') is not None
    ]

    position_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {'player_count': 0.0, 'expected_points': 0.0, 'actual_points': 0.0}
    )
    for report in season_reports:
        for row in report.get('position_summary') or []:
            bucket = position_totals[str(row.get('position') or '')]
            bucket['player_count'] += float(row.get('player_count') or 0.0)
            bucket['expected_points'] += float(row.get('expected_points') or 0.0)
            bucket['actual_points'] += float(row.get('actual_points') or 0.0)

    position_rollup = []
    for position, totals in sorted(position_totals.items()):
        position_rollup.append(
            {
                'position': position,
                'player_count': int(totals['player_count']),
                'expected_points': totals['expected_points'],
                'actual_points': totals['actual_points'],
                'delta': totals['actual_points'] - totals['expected_points'],
            }
        )

    audit_rates = [
        report.get('audit_context', {}).get('follow_rate')
        for report in season_reports
        if report.get('audit_context', {}).get('follow_rate') is not None
    ]
    wait_deltas = [
        report.get('audit_context', {})
        .get('wait_policy_calibration', {})
        .get('mean_actual_delta_vs_top_wait_candidate')
        for report in season_reports
        if report.get('audit_context', {})
        .get('wait_policy_calibration', {})
        .get('mean_actual_delta_vs_top_wait_candidate')
        is not None
    ]

    return {
        'season_count': season_count,
        'season_years': [report['season_year'] for report in season_reports],
        'mean_starter_delta': float(sum(starter_deltas) / len(starter_deltas))
        if starter_deltas
        else None,
        'mean_full_roster_delta': float(sum(full_deltas) / len(full_deltas))
        if full_deltas
        else None,
        'mean_drafted_player_hit_rate': float(sum(hit_rates) / len(hit_rates))
        if hit_rates
        else None,
        'mean_follow_rate': float(sum(audit_rates) / len(audit_rates))
        if audit_rates
        else None,
        'mean_wait_policy_delta': float(sum(wait_deltas) / len(wait_deltas))
        if wait_deltas
        else None,
        'position_rollup': position_rollup,
    }


def build_retrospective_report(
    finalized_paths: list[Path | str],
    outcomes_path: Path | str,
    fallback_year: int | None = None,
) -> dict[str, Any]:
    if not finalized_paths:
        raise ValueError('At least one finalized draft JSON file is required.')

    resolved_finalized = [Path(path) for path in finalized_paths]
    outcomes_resolved = Path(outcomes_path)
    outcomes = load_realized_outcomes(outcomes_resolved)

    season_reports = []
    warnings: list[str] = []
    for path in resolved_finalized:
        payload, season_year = load_finalized_payload(path, fallback_year=fallback_year)
        report = _season_report(
            payload,
            path,
            season_year=season_year,
            outcomes=outcomes,
            outcomes_path=outcomes_resolved,
        )
        season_reports.append(report)
        warnings.extend(report.get('warnings') or [])

    season_reports.sort(key=lambda item: int(item['season_year']))
    overall_status = (
        'available'
        if all(report.get('status') == 'available' for report in season_reports)
        else 'degraded'
    )
    return {
        'schema_version': RETROSPECTIVE_SCHEMA_VERSION,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'status': overall_status,
        'warnings': list(dict.fromkeys(warnings)),
        'provenance': {
            'finalized_drafts': [str(path) for path in resolved_finalized],
            'outcomes_path': str(outcomes_resolved),
            'season_count': len(season_reports),
            'season_years': [report['season_year'] for report in season_reports],
        },
        'season_reports': season_reports,
        'overall_rollup': _rollup_reports(season_reports),
    }


def _report_title(report: dict[str, Any]) -> str:
    seasons = report.get('provenance', {}).get('season_years') or []
    if not seasons:
        return 'Draft retrospective'
    if len(seasons) == 1:
        return f'Draft retrospective: {seasons[0]}'
    return f'Draft retrospective: {min(seasons)}-{max(seasons)}'


def build_retrospective_html(report: dict[str, Any]) -> str:
    title = _report_title(report)
    warnings = report.get('warnings') or []
    season_cards = []
    for season in report.get('season_reports') or []:
        metrics = season.get('outcome_metrics') or {}
        audit = season.get('audit_context') or {}
        season_cards.append(
            f"""
            <section class="card">
              <h2>Season {season.get('season_year')}</h2>
              <p class="status">Status: {season.get('status', 'unknown')}</p>
              <p class="muted">Outcome-grounded evaluation is primary. Follow/pivot metrics below are secondary audit context.</p>
              <div class="grid">
                <div><span class="label">Expected starter points</span><span>{metrics.get('expected_starter_points')}</span></div>
                <div><span class="label">Actual starter points</span><span>{metrics.get('actual_starter_points')}</span></div>
                <div><span class="label">Starter delta</span><span>{metrics.get('starter_delta')}</span></div>
                <div><span class="label">Expected full roster</span><span>{metrics.get('expected_full_roster_points')}</span></div>
                <div><span class="label">Actual full roster</span><span>{metrics.get('actual_full_roster_points')}</span></div>
                <div><span class="label">Full roster delta</span><span>{metrics.get('full_roster_delta')}</span></div>
                <div><span class="label">Drafted player hit rate</span><span>{metrics.get('drafted_player_hit_rate')}</span></div>
              </div>
              <h3>Secondary audit context</h3>
              <div class="grid">
                <div><span class="label">Audit status</span><span>{audit.get('status')}</span></div>
                <div><span class="label">Follow rate</span><span>{audit.get('follow_rate')}</span></div>
                <div><span class="label">Pivot count</span><span>{audit.get('pivot_count')}</span></div>
                <div><span class="label">Mean pivot delta vs recommendation</span><span>{audit.get('mean_pivot_actual_delta_vs_recommendation')}</span></div>
                <div><span class="label">Wait-policy status</span><span>{audit.get('wait_policy_calibration', {}).get('status')}</span></div>
                <div><span class="label">Mean delta vs top wait candidate</span><span>{audit.get('wait_policy_calibration', {}).get('mean_actual_delta_vs_top_wait_candidate')}</span></div>
              </div>
              <h3>Position summary</h3>
              <table>
                <thead><tr><th>Position</th><th>Count</th><th>Expected</th><th>Actual</th><th>Mean delta</th></tr></thead>
                <tbody>
                  {''.join(
                      f"<tr><td>{row.get('position')}</td><td>{row.get('player_count')}</td><td>{row.get('expected_points')}</td><td>{row.get('actual_points')}</td><td>{row.get('mean_delta')}</td></tr>"
                      for row in (season.get('position_summary') or [])
                  )}
                </tbody>
              </table>
            </section>
            """
        )

    rollup = report.get('overall_rollup') or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #0b1220; color: #e5eefc; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 80px; }}
    .hero {{ margin-bottom: 24px; }}
    .muted {{ color: #9fb0cb; }}
    .warning-list {{ padding-left: 20px; color: #ffd089; }}
    .card {{ background: #121c31; border: 1px solid #22314f; border-radius: 16px; padding: 20px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .grid div {{ background: rgba(255,255,255,0.03); border-radius: 12px; padding: 12px; }}
    .label {{ display: block; font-size: 12px; color: #9fb0cb; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #22314f; }}
    th {{ color: #9fb0cb; font-weight: 600; }}
    .status {{ color: #9fe1b9; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{title}</h1>
      <p class="muted">Outcome-grounded retrospective evaluation for finalized FFBayes draft artifacts.</p>
      <p class="muted">Generated {report.get('generated_at')} from {len(report.get('season_reports') or [])} finalized draft file(s).</p>
    </section>
    <section class="card">
      <h2>Overall rollup</h2>
      <div class="grid">
        <div><span class="label">Season count</span><span>{rollup.get('season_count')}</span></div>
        <div><span class="label">Mean starter delta</span><span>{rollup.get('mean_starter_delta')}</span></div>
        <div><span class="label">Mean full roster delta</span><span>{rollup.get('mean_full_roster_delta')}</span></div>
        <div><span class="label">Mean drafted-player hit rate</span><span>{rollup.get('mean_drafted_player_hit_rate')}</span></div>
        <div><span class="label">Mean follow rate</span><span>{rollup.get('mean_follow_rate')}</span></div>
        <div><span class="label">Mean wait-policy delta</span><span>{rollup.get('mean_wait_policy_delta')}</span></div>
      </div>
      {'<ul class="warning-list">' + ''.join(f'<li>{warning}</li>' for warning in warnings) + '</ul>' if warnings else '<p class="muted">No warnings.</p>'}
    </section>
    {''.join(season_cards)}
  </main>
</body>
</html>
"""


def _discover_finalized_json_paths(year: int | None = None) -> list[Path]:
    if year is None:
        return []
    canonical_dir = get_finalized_drafts_dir(year)
    canonical_matches = sorted(
        canonical_dir.glob(f'ffbayes_finalized_draft_{year}_*.json')
    )
    if canonical_matches:
        return canonical_matches
    legacy_dir = get_draft_strategy_dir(year)
    return sorted(legacy_dir.glob(f'ffbayes_finalized_draft_{year}_*.json'))


def write_retrospective_artifacts(
    report: dict[str, Any],
    year: int,
    output_json: Path | str | None = None,
    output_html: Path | str | None = None,
    write_html: bool = True,
) -> dict[str, Path | None]:
    json_path = Path(output_json) if output_json is not None else get_draft_retrospective_json_path(year)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, default=str, indent=2), encoding='utf-8')

    html_path: Path | None = None
    if write_html:
        html_path = Path(output_html) if output_html is not None else get_draft_retrospective_html_path(year)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(build_retrospective_html(report), encoding='utf-8')

    return {
        'json_path': json_path,
        'html_path': html_path,
    }


def run_draft_retrospective(
    finalized_json: list[Path | str] | None = None,
    import_finalized: list[Path | str] | None = None,
    outcomes_path: Path | str | None = None,
    year: int | None = None,
    output_json: Path | str | None = None,
    output_html: Path | str | None = None,
    write_html: bool = True,
    ingest_only: bool = False,
    move_imported: bool = False,
) -> dict[str, Any]:
    imported = import_finalized_artifacts(
        import_paths=import_finalized or [],
        year=year,
        move_files=move_imported,
    )
    if ingest_only:
        return {
            'status': 'imported',
            'report': None,
            'json_path': None,
            'html_path': None,
            'imported': imported,
        }

    finalized_paths = [Path(path) for path in finalized_json or []]
    if not finalized_paths and imported['imported_json_paths']:
        finalized_paths = list(imported['imported_json_paths'])
    if not finalized_paths:
        finalized_paths = _discover_finalized_json_paths(year)
    if not finalized_paths:
        raise FileNotFoundError(
            'No finalized draft JSON files were provided or discovered. '
            'Pass `--finalized-json`, import browser-downloaded finalized artifacts '
            'with `--import-finalized`, or place finalized draft JSON files in the '
            'canonical `finalized_drafts/` runtime directory.'
        )

    resolved_outcomes = Path(outcomes_path) if outcomes_path is not None else get_unified_dataset_csv_path()
    report = build_retrospective_report(
        finalized_paths=finalized_paths,
        outcomes_path=resolved_outcomes,
        fallback_year=year,
    )
    output_year = year or max(
        int(season_year)
        for season_year in report.get('provenance', {}).get('season_years', [])
    )
    written = write_retrospective_artifacts(
        report,
        year=output_year,
        output_json=output_json,
        output_html=output_html,
        write_html=write_html,
    )
    return {
        'status': report.get('status'),
        'report': report,
        'imported': imported,
        **written,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate an outcome-grounded retrospective from finalized draft artifacts'
    )
    parser.add_argument(
        '--import-finalized',
        nargs='+',
        type=Path,
        help=(
            'Import browser-downloaded finalized artifacts into the canonical '
            '`finalized_drafts/` runtime folder before analysis'
        ),
    )
    parser.add_argument(
        '--move-imported',
        action='store_true',
        help='Move imported finalized artifacts instead of copying them',
    )
    parser.add_argument(
        '--ingest-only',
        action='store_true',
        help='Import finalized artifacts into the canonical runtime folder and exit',
    )
    parser.add_argument(
        '--finalized-json',
        nargs='+',
        type=Path,
        help='One or more finalized draft JSON files to analyze',
    )
    parser.add_argument(
        '--outcomes-path',
        type=Path,
        help='Override the realized season outcome dataset (defaults to the unified dataset CSV)',
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Season year for output/discovery and fallback import inference',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        help='Override the retrospective JSON output path',
    )
    parser.add_argument(
        '--output-html',
        type=Path,
        help='Override the retrospective HTML output path',
    )
    parser.add_argument(
        '--skip-html',
        action='store_true',
        help='Write only the canonical JSON artifact',
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_draft_retrospective(
            import_finalized=args.import_finalized,
            finalized_json=args.finalized_json,
            outcomes_path=args.outcomes_path,
            year=args.year,
            output_json=args.output_json,
            output_html=args.output_html,
            write_html=not args.skip_html,
            ingest_only=args.ingest_only,
            move_imported=args.move_imported,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')

    imported = result.get('imported') or {}
    imported_paths = imported.get('imported_paths') or []
    if imported_paths:
        print(
            f'Imported {len(imported_paths)} finalized artifact(s) into '
            'the canonical runtime folder:'
        )
        for path in imported_paths:
            print(f'   - {path}')

    if args.ingest_only:
        return 0

    report = result['report']
    print(f'✅ Draft retrospective status: {result["status"]}')
    print(f'   seasons: {", ".join(str(year) for year in report["provenance"]["season_years"])}')
    print(f'   outcomes: {report["provenance"]["outcomes_path"]}')
    print(f'   json: {result["json_path"]}')
    if result.get('html_path') is not None:
        print(f'   html: {result["html_path"]}')
    if report.get('warnings'):
        print('   warnings:')
        for warning in report['warnings']:
            print(f'   - {warning}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
