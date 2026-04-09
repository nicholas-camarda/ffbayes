#!/usr/bin/env python3
"""Refresh the dashboard HTML and restage the GitHub Pages copy in one step."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ffbayes.refresh_dashboard import refresh_runtime_dashboard


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def stage_dashboard(
    year: int | None = None,
    payload_path: Path | str | None = None,
    output_html: Path | str | None = None,
) -> dict[str, Any]:
    """Refresh the current dashboard and stage the Pages copy."""
    return refresh_runtime_dashboard(
        year=year,
        payload_path=payload_path,
        output_html=output_html,
        stage_pages=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the one-step staging workflow."""
    parser = argparse.ArgumentParser(
        description='Refresh dashboard HTML and restage the GitHub Pages copy'
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to stage'
    )
    parser.add_argument(
        '--payload-path',
        type=Path,
        help='Override the source dashboard payload JSON file',
    )
    parser.add_argument(
        '--output-html',
        type=Path,
        help='Override the regenerated dashboard HTML path',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit machine-readable JSON output',
    )
    return parser


def main() -> int:
    """Entry point for the one-step dashboard staging workflow."""
    parser = build_parser()
    args = parser.parse_args()
    result = stage_dashboard(
        year=args.year,
        payload_path=args.payload_path,
        output_html=args.output_html,
    )
    if args.json:
        print(json.dumps(_to_jsonable(result), indent=2))
        return 0

    print(f'✅ Refreshed dashboard HTML at {result["html_path"]}')
    print(f'   source payload: {result["source_payload_path"]}')
    if 'runtime_dashboard_index' in result:
        print(f'   runtime dashboard: {result["runtime_dashboard_index"]}')
    if 'repo_dashboard_index' in result:
        print(f'   repo dashboard: {result["repo_dashboard_index"]}')
    if 'staged_index_path' in result:
        print(f'   staged site: {result["staged_index_path"]}')
    if 'staged_payload_path' in result:
        print(f'   staged payload: {result["staged_payload_path"]}')
    if 'staged_provenance_path' in result:
        print(f'   staged provenance: {result["staged_provenance_path"]}')
    if result.get('stale_paths'):
        print(f'   stale paths replaced: {", ".join(result["stale_paths"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
