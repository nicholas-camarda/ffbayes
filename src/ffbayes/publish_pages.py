#!/usr/bin/env python3
"""Stage the live draft dashboard for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ffbayes.utils.path_constants import (
    get_dashboard_html_path,
    get_dashboard_payload_path,
    get_pages_site_dir,
)

PAYLOAD_ASSIGNMENT_PREFIX = 'window.FFBAYES_DASHBOARD = '
PAYLOAD_ASSIGNMENT_SUFFIX = ';\n\n    (() => {'


def _build_publish_provenance(
    payload: dict,
    year: int,
    source_html: Path,
    source_payload: Path | None,
) -> dict:
    analysis_provenance = payload.get('analysis_provenance') or {}
    analysis_freshness = (
        analysis_provenance.get('overall_freshness')
        or payload.get('decision_evidence', {}).get('freshness')
        or {'status': 'unknown', 'override_used': False, 'warnings': []}
    )
    warnings = list(dict.fromkeys(str(item) for item in analysis_freshness.get('warnings', []) if item))
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
    }


def _inject_dashboard_payload_into_html(html_text: str, payload: dict) -> str:
    start = html_text.find(PAYLOAD_ASSIGNMENT_PREFIX)
    if start == -1:
        return html_text
    end = html_text.find(PAYLOAD_ASSIGNMENT_SUFFIX, start)
    if end == -1:
        return html_text
    payload_json = json.dumps(payload, default=str)
    return (
        html_text[:start]
        + PAYLOAD_ASSIGNMENT_PREFIX
        + payload_json
        + html_text[end:]
    )


def stage_pages_site(
    year: int | None = None,
    source_html: Path | str | None = None,
    source_payload: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
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
    if resolved_html.resolve() != index_path.resolve():
        shutil.copy2(resolved_html, index_path)

    payload_target = resolved_output_dir / 'dashboard_payload.json'
    provenance_target = resolved_output_dir / 'publish_provenance.json'
    resolved_payload = (
        Path(source_payload)
        if source_payload is not None
        else get_dashboard_payload_path(resolved_year)
    )
    payload_data: dict | None = None
    if resolved_payload.exists():
        payload_data = json.loads(resolved_payload.read_text(encoding='utf-8'))
        payload_data['publish_provenance'] = _build_publish_provenance(
            payload_data,
            resolved_year,
            resolved_html,
            resolved_payload,
        )
        payload_target.write_text(
            json.dumps(payload_data, default=str, indent=2),
            encoding='utf-8',
        )
        provenance_target.write_text(
            json.dumps(payload_data['publish_provenance'], default=str, indent=2),
            encoding='utf-8',
        )
    elif payload_target.exists():
        payload_target.unlink()
        if provenance_target.exists():
            provenance_target.unlink()

    if payload_data is not None:
        html_text = index_path.read_text(encoding='utf-8')
        index_path.write_text(
            _inject_dashboard_payload_into_html(html_text, payload_data),
            encoding='utf-8',
        )

    nojekyll_path = resolved_output_dir / '.nojekyll'
    nojekyll_path.write_text('\n', encoding='utf-8')

    return {
        'site_dir': resolved_output_dir,
        'index_path': index_path,
        'payload_path': payload_target,
        'provenance_path': provenance_target,
        'nojekyll_path': nojekyll_path,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Pages staging helper."""
    parser = argparse.ArgumentParser(
        description='Stage the live draft dashboard for GitHub Pages'
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to stage'
    )
    parser.add_argument(
        '--source-html',
        type=Path,
        help='Override the source dashboard HTML file',
    )
    parser.add_argument(
        '--source-payload',
        type=Path,
        help='Override the source dashboard payload JSON file',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override the GitHub Pages site directory',
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
