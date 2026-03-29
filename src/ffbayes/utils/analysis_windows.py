#!/usr/bin/env python3
"""Canonical analysis-window and freshness helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ANALYSIS_WINDOW_SIZE = 5


def get_analysis_years(reference_year: int | None = None, window_size: int = DEFAULT_ANALYSIS_WINDOW_SIZE) -> list[int]:
    """Return the canonical analysis-year window.

    The default window is the last five completed seasons relative to the
    projection year. On March 29, 2026 this resolves to 2021-2025.
    """
    if reference_year is None:
        reference_year = datetime.now().year
    window_size = max(1, int(window_size))
    return list(range(reference_year - window_size, reference_year))


@dataclass(frozen=True)
class AnalysisWindow:
    """Resolved analysis window plus freshness status."""

    reference_year: int
    window_size: int
    expected_years: tuple[int, ...]
    found_years: tuple[int, ...]
    missing_years: tuple[int, ...]
    allow_stale: bool
    freshness_status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def latest_expected_year(self) -> int | None:
        return self.expected_years[-1] if self.expected_years else None

    @property
    def latest_found_year(self) -> int | None:
        return self.found_years[-1] if self.found_years else None

    @property
    def is_fresh(self) -> bool:
        return self.freshness_status == 'fresh'

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['latest_expected_year'] = self.latest_expected_year
        payload['latest_found_year'] = self.latest_found_year
        payload['is_fresh'] = self.is_fresh
        return payload


def resolve_analysis_window(
    available_years: Iterable[int] | None = None,
    reference_year: int | None = None,
    window_size: int = DEFAULT_ANALYSIS_WINDOW_SIZE,
    allow_stale: bool = False,
) -> AnalysisWindow:
    """Resolve expected vs. found analysis years and enforce freshness policy."""
    if reference_year is None:
        reference_year = datetime.now().year

    expected_years = tuple(get_analysis_years(reference_year, window_size))
    found_years = tuple(sorted({int(year) for year in (available_years or []) if year is not None}))
    missing_years = tuple(year for year in expected_years if year not in found_years)

    warnings: list[str] = []
    freshness_status = 'fresh'

    latest_expected_year = expected_years[-1] if expected_years else None
    if latest_expected_year is not None and latest_expected_year not in found_years:
        freshness_status = 'stale'
        warnings.append(f'Missing latest expected season: {latest_expected_year}')
        if not allow_stale:
            raise RuntimeError(
                f'Latest expected season {latest_expected_year} is missing. '
                f'Expected analysis window: {list(expected_years)}'
            )
    elif missing_years:
        freshness_status = 'degraded'
        warnings.append(f'Missing analysis seasons: {list(missing_years)}')

    if allow_stale and freshness_status == 'stale':
        freshness_status = 'degraded'
        warnings.append('Pipeline explicitly allowed to continue without the latest season')

    return AnalysisWindow(
        reference_year=reference_year,
        window_size=window_size,
        expected_years=expected_years,
        found_years=found_years,
        missing_years=missing_years,
        allow_stale=allow_stale,
        freshness_status=freshness_status,
        warnings=tuple(warnings),
    )


def build_freshness_manifest(
    window: AnalysisWindow,
    source_name: str,
    source_path: Path | str | None = None,
    source_updated_at: str | None = None,
    found_files: Sequence[Path | str] | None = None,
) -> dict:
    """Build a freshness manifest for a source tree."""
    found_files = list(found_files or [])
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'source_name': source_name,
        'source_path': str(source_path) if source_path is not None else None,
        'source_updated_at': source_updated_at,
        'analysis_window': window.to_dict(),
        'expected_years': list(window.expected_years),
        'found_years': list(window.found_years),
        'missing_years': list(window.missing_years),
        'warnings': list(window.warnings),
        'found_files': [str(file_path) for file_path in found_files],
    }


def write_freshness_manifest(manifest: dict, output_path: Path | str) -> Path:
    """Write a freshness manifest to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output_path
