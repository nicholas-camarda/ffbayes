#!/usr/bin/env python3
"""Stage the live draft dashboard for GitHub Pages."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ffbayes.utils.json_serialization import dumps_strict_json, to_strict_jsonable
from ffbayes.utils.path_constants import (
    get_dashboard_html_path,
    get_dashboard_payload_path,
    get_pages_site_dir,
)

PAYLOAD_ASSIGNMENT_PREFIX = 'window.FFBAYES_DASHBOARD = '
PAYLOAD_ASSIGNMENT_SUFFIX = ';\n\n    (() => {'


def _normalized_json_text(payload: dict[str, Any]) -> str:
    normalized = to_strict_jsonable(payload)
    if isinstance(normalized, dict) and isinstance(
        normalized.get('publish_provenance'), dict
    ):
        normalized['publish_provenance']['published_at'] = '<normalized>'
    if isinstance(normalized, dict) and normalized.get('published_at') is not None:
        normalized['published_at'] = '<normalized>'
    return dumps_strict_json(normalized, sort_keys=True, indent=2)


def _build_publish_provenance(
    payload: dict[str, Any], year: int, source_html: Path, source_payload: Path | None
) -> dict[str, Any]:
    analysis_provenance = payload.get('analysis_provenance') or {}
    analysis_freshness = (
        analysis_provenance.get('overall_freshness')
        or payload.get('decision_evidence', {}).get('freshness')
        or {'status': 'unknown', 'override_used': False, 'warnings': []}
    )
    warnings = list(
        dict.fromkeys(
            str(item) for item in analysis_freshness.get('warnings', []) if item
        )
    )
    return {
        'schema_version': 'publish_provenance_v1',
        'season_year': int(year),
        'published_at': datetime.now().isoformat(timespec='seconds'),
        'dashboard_generated_at': payload.get('generated_at'),
        'source_html': source_html.name,
        'source_payload': source_payload.name if source_payload is not None else None,
        'analysis_freshness': analysis_freshness,
        'override_used': bool(analysis_freshness.get('override_used', False)),
        'warnings': warnings,
        'surface_sync': {
            'status': 'synchronized',
            'detail': 'The staged HTML and staged payload were written together during dashboard staging.',
            'html_bootstrap': 'inline_payload_embedded',
        },
    }


def _inject_dashboard_payload_into_html(html_text: str, payload: dict[str, Any]) -> str:
    start = html_text.find(PAYLOAD_ASSIGNMENT_PREFIX)
    if start == -1:
        return html_text
    end = html_text.find(PAYLOAD_ASSIGNMENT_SUFFIX, start)
    if end == -1:
        return html_text
    payload_json = dumps_strict_json(payload)
    return (
        html_text[:start] + PAYLOAD_ASSIGNMENT_PREFIX + payload_json + html_text[end:]
    )


def stage_pages_site(
    year: int | None = None,
    source_html: Path | str | None = None,
    source_payload: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Copy the canonical dashboard artifacts into the Pages site tree."""
    resolved_year = year or datetime.now().year
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else get_pages_site_dir()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_html = (
        Path(source_html)
        if source_html is not None
        else get_dashboard_html_path(resolved_year)
    )
    if not resolved_html.exists():
        raise FileNotFoundError(
            f'Dashboard HTML not found at {resolved_html}. '
            'Run `ffbayes draft-strategy` first.'
        )

    index_path = resolved_output_dir / 'index.html'
    existing_index_text = (
        index_path.read_text(encoding='utf-8') if index_path.exists() else None
    )
    payload_target = resolved_output_dir / 'dashboard_payload.json'
    provenance_target = resolved_output_dir / 'publish_provenance.json'
    resolved_payload = (
        Path(source_payload)
        if source_payload is not None
        else get_dashboard_payload_path(resolved_year)
    )
    payload_data: dict[str, Any] | None = None
    if resolved_payload.exists():
        payload_data = json.loads(resolved_payload.read_text(encoding='utf-8'))
        payload_data['publish_provenance'] = _build_publish_provenance(
            payload_data, resolved_year, resolved_html, resolved_payload
        )
        staged_payload_text = dumps_strict_json(payload_data, indent=2)
        staged_provenance_text = dumps_strict_json(
            payload_data['publish_provenance'], indent=2
        )
    elif payload_target.exists():
        payload_target.unlink()
        if provenance_target.exists():
            provenance_target.unlink()
        staged_payload_text = None
        staged_provenance_text = None
    else:
        staged_payload_text = None
        staged_provenance_text = None

    staged_index_text = resolved_html.read_text(encoding='utf-8')

    if payload_data is not None:
        staged_index_text = _inject_dashboard_payload_into_html(
            staged_index_text, payload_data
        )

    stale_paths: list[Path] = []
    if existing_index_text is not None and existing_index_text != staged_index_text:
        stale_paths.append(index_path)
    index_path.write_text(staged_index_text, encoding='utf-8')

    if staged_payload_text is not None:
        assert payload_data is not None
        assert staged_provenance_text is not None
        if payload_target.exists():
            existing_payload = payload_target.read_text(encoding='utf-8')
            if _normalized_json_text(
                json.loads(existing_payload)
            ) != _normalized_json_text(payload_data):
                stale_paths.append(payload_target)
        payload_target.write_text(staged_payload_text, encoding='utf-8')
        if provenance_target.exists():
            existing_provenance = json.loads(
                provenance_target.read_text(encoding='utf-8')
            )
            if _normalized_json_text(existing_provenance) != _normalized_json_text(
                payload_data['publish_provenance']
            ):
                stale_paths.append(provenance_target)
        provenance_target.write_text(staged_provenance_text, encoding='utf-8')

    nojekyll_path = resolved_output_dir / '.nojekyll'
    nojekyll_path.write_text('\n', encoding='utf-8')

    return {
        'status': 'staged',
        'site_dir': resolved_output_dir,
        'index_path': index_path,
        'payload_path': payload_target,
        'provenance_path': provenance_target,
        'nojekyll_path': nojekyll_path,
        'stale_paths': list(dict.fromkeys(stale_paths)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Pages staging helper."""
    parser = argparse.ArgumentParser(
        description=(
            'Stage the live draft dashboard for GitHub Pages without rerendering it first'
        )
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to stage'
    )
    parser.add_argument(
        '--source-html', type=Path, help='Override the source dashboard HTML file'
    )
    parser.add_argument(
        '--source-payload',
        type=Path,
        help='Override the source dashboard payload JSON file',
    )
    parser.add_argument(
        '--output-dir', type=Path, help='Override the GitHub Pages site directory'
    )
    return parser


def main() -> int:
    """Entry point for staging the static Pages dashboard."""
    parser = build_parser()
    args = parser.parse_args()
    result = stage_pages_site(
        year=args.year,
        source_html=args.source_html,
        source_payload=args.source_payload,
        output_dir=args.output_dir,
    )
    print(f'✅ Staged GitHub Pages site at {result["site_dir"]}')
    print(f'   index: {result["index_path"]}')
    if result['payload_path'].exists():
        print(f'   payload: {result["payload_path"]}')
    if result['provenance_path'].exists():
        print(f'   provenance: {result["provenance_path"]}')
    if result['stale_paths']:
        print('   replaced stale paths:')
        for path in result['stale_paths']:
            print(f'   - {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
