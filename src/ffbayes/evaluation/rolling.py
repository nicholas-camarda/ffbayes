"""Temporal contracts for rolling or prospective evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


class TemporalLeakageError(ValueError):
    """Raised when a feature or source was published after an evaluation cutoff."""


def _timestamp(value: Any, field: str) -> datetime:
    if value is None or value == '':
        raise TemporalLeakageError(f'{field} is required for as-of evaluation')
    text = str(value).replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TemporalLeakageError(f'{field} is not an ISO timestamp') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class SnapshotManifest:
    """Versioned source snapshot metadata used in an evaluation manifest."""

    source: str
    season: int
    as_of: str
    published_at: str
    fetched_at: str
    sha256: str
    schema_version: str
    row_count: int

    def validate(self, cutoff: str | datetime) -> None:
        cutoff_value = (
            cutoff
            if isinstance(cutoff, datetime)
            else _timestamp(cutoff, 'evaluation cutoff')
        )
        if cutoff_value.tzinfo is None:
            cutoff_value = cutoff_value.replace(tzinfo=timezone.utc)
        for field in ('published_at', 'as_of', 'fetched_at'):
            if _timestamp(getattr(self, field), field) > cutoff_value.astimezone(
                timezone.utc
            ):
                raise TemporalLeakageError(
                    f'{self.source} {field} occurs after evaluation cutoff'
                )
        if not re.fullmatch(r'[0-9a-f]{64}', self.sha256) or self.row_count < 1:
            raise TemporalLeakageError(f'{self.source} manifest is incomplete')


def validate_as_of_inputs(
    records: Iterable[Mapping[str, Any]],
    cutoff: str | datetime,
    *,
    timestamp_fields: tuple[str, ...] = (
        'as_of',
        'published_at',
        'available_at',
        'projection_as_of',
        'adp_as_of',
        'fetched_at',
    ),
) -> None:
    """Reject future-dated feature rows before any fit or draft simulation."""
    cutoff_value = (
        cutoff
        if isinstance(cutoff, datetime)
        else _timestamp(cutoff, 'evaluation cutoff')
    )
    if cutoff_value.tzinfo is None:
        cutoff_value = cutoff_value.replace(tzinfo=timezone.utc)
    cutoff_value = cutoff_value.astimezone(timezone.utc)
    for index, record in enumerate(records):
        for required in ('projection_as_of', 'adp_as_of'):
            if not record.get(required):
                raise TemporalLeakageError(f'input row {index} requires {required}')
        for field in timestamp_fields:
            if field not in record or record[field] in (None, ''):
                continue
            if _timestamp(record[field], f'{field}[{index}]') > cutoff_value:
                raise TemporalLeakageError(
                    f'input row {index} has {field} after evaluation cutoff'
                )


def evaluation_manifest(
    *,
    cutoff: str | None,
    season: int,
    source_manifests: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any],
    seed: int,
    version: str,
) -> dict[str, Any]:
    """Create a stable manifest that binds data, rules, and random seed."""
    source_values = [dict(value) for value in source_manifests]
    for source in source_values:
        required_source_fields = {
            'source',
            'season',
            'as_of',
            'published_at',
            'fetched_at',
            'sha256',
            'schema_version',
            'row_count',
        }
        missing = sorted(required_source_fields.difference(source))
        if missing:
            raise TemporalLeakageError(
                f'source manifest is missing verified fields: {missing}'
            )
        SnapshotManifest(
            source=str(source.get('source', 'unknown')),
            season=int(source['season']),
            as_of=str(source['as_of']),
            published_at=str(source['published_at']),
            fetched_at=str(source['fetched_at']),
            sha256=str(source['sha256']),
            schema_version=str(source['schema_version']),
            row_count=int(source['row_count']),
        ).validate(cutoff or str(source['as_of']))
        if int(source['season']) != int(season):
            raise TemporalLeakageError(
                f'{source["source"]} season does not match evaluation season {season}'
            )
    if cutoff and not source_values:
        raise TemporalLeakageError(
            'verified as-of evaluation requires at least one source manifest'
        )
    row_values = [dict(row) for row in rows]
    if cutoff:
        validate_as_of_inputs(row_values, cutoff)
    canonical = json.dumps(
        {'profile': dict(profile), 'rows': row_values, 'sources': source_values},
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
        default=str,
    )
    return {
        'version': version,
        'season': season,
        'cutoff': cutoff,
        'temporal_status': 'declared_timestamps_validated'
        if cutoff
        else 'synthetic_unverified',
        'seed': int(seed),
        'input_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'source_manifests': source_values,
        # Keep the result reproducible.  Wall-clock generation time would make
        # equivalent input/output manifests differ without adding evidence.
        'generated_at': cutoff,
    }
