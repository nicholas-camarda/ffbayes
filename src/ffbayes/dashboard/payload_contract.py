"""Versioned dashboard payload contract and validation."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import jsonschema

DASHBOARD_SCHEMA_VERSION = 1

# Pre-versioning contract; keep in sync with refresh_dashboard expectations.
LEGACY_REQUIRED_PAYLOAD_KEYS = (
    'generated_at',
    'league_settings',
    'decision_table',
    'decision_evidence',
)


class DashboardPayloadError(ValueError):
    """A dashboard payload failed contract validation."""


def load_dashboard_schema() -> dict[str, Any]:
    """Load the packaged dashboard payload JSON Schema."""
    schema_text = (
        resources.files('ffbayes.dashboard')
        .joinpath('schemas/dashboard_payload.schema.json')
        .read_text(encoding='utf-8')
    )
    return json.loads(schema_text)


def stamp_schema_version(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp the current schema version onto a payload in place."""
    payload['dashboard_schema_version'] = DASHBOARD_SCHEMA_VERSION
    return payload


def _validate_legacy_payload(payload: dict[str, Any], source: str) -> None:
    missing = [
        key
        for key in LEGACY_REQUIRED_PAYLOAD_KEYS
        if key not in payload or payload.get(key) is None
    ]
    if missing:
        raise DashboardPayloadError(
            f'Dashboard payload from {source} is missing required keys: '
            f'{", ".join(missing)}'
        )
    if not isinstance(payload.get('decision_table'), list):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has a non-list `decision_table`.'
        )
    if not isinstance(payload.get('league_settings'), dict):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has an invalid `league_settings` object.'
        )


def validate_dashboard_payload(
    payload: Any, source: str = '<in-memory>'
) -> dict[str, Any]:
    """Validate a dashboard payload against the versioned contract.

    Payloads without ``dashboard_schema_version`` are treated as legacy and
    validated only against the pre-existing required-key contract so old
    artifacts keep loading. Versioned payloads must match the JSON Schema
    exactly (fail closed).
    """
    if not isinstance(payload, dict):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} must be a JSON object.'
        )
    version = payload.get('dashboard_schema_version')
    if version is None:
        _validate_legacy_payload(payload, source)
        return payload
    if version != DASHBOARD_SCHEMA_VERSION:
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has unsupported schema version '
            f'{version!r}; this build supports version {DASHBOARD_SCHEMA_VERSION}.'
        )
    validator = jsonschema.Draft202012Validator(load_dashboard_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        details = '; '.join(
            f'{"/".join(str(part) for part in err.absolute_path) or "<root>"}: '
            f'{err.message}'
            for err in errors[:10]
        )
        raise DashboardPayloadError(
            f'Dashboard payload from {source} failed schema validation: {details}'
        )
    return payload
