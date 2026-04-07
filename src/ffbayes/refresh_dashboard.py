#!/usr/bin/env python3
"""Regenerate dashboard HTML from an existing dashboard payload."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    LeagueSettings,
    _stage_runtime_dashboard_shortcuts,
    export_dashboard_html,
)
from ffbayes.publish_pages import stage_pages_site
from ffbayes.utils.path_constants import (
    get_dashboard_html_path,
    get_dashboard_payload_path,
)


def refresh_runtime_dashboard(
    year: int | None = None,
    payload_path: Path | str | None = None,
    output_html: Path | str | None = None,
    stage_pages: bool = False,
) -> dict[str, Path]:
    """Rebuild dashboard HTML from the current runtime payload only."""
    resolved_year = year or datetime.now().year
    resolved_payload = (
        Path(payload_path)
        if payload_path is not None
        else get_dashboard_payload_path(resolved_year)
    )
    if not resolved_payload.exists():
        raise FileNotFoundError(
            f'Dashboard payload not found at {resolved_payload}. '
            'Run `ffbayes draft-strategy` first.'
        )

    payload = json.loads(resolved_payload.read_text(encoding='utf-8'))
    league_settings = LeagueSettings.from_mapping(
        {'league_settings': payload.get('league_settings', {})}
    )
    resolved_output_html = (
        Path(output_html)
        if output_html is not None
        else get_dashboard_html_path(resolved_year)
    )

    export_dashboard_html(
        pd.DataFrame(payload.get('decision_table', [])),
        pd.DataFrame(payload.get('recommendation_summary', [])),
        resolved_output_html,
        league_settings,
        backtest=payload.get('backtest', {}),
        source_freshness=pd.DataFrame(payload.get('source_freshness', [])),
        dashboard_payload=payload,
    )

    result: dict[str, Path] = {
        'source_payload_path': resolved_payload,
        'html_path': resolved_output_html,
    }
    result.update(
        _stage_runtime_dashboard_shortcuts(
            resolved_output_html, resolved_payload, resolved_year
        )
    )

    if stage_pages:
        staged = stage_pages_site(
            year=resolved_year,
            source_html=resolved_output_html,
            source_payload=resolved_payload,
        )
        result.update(staged)
        result['staged_payload_path'] = staged['payload_path']
        result['staged_index_path'] = staged['index_path']
        result['staged_provenance_path'] = staged['provenance_path']

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description='Regenerate dashboard HTML from the existing runtime payload'
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to refresh'
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
        '--stage-pages',
        action='store_true',
        help='Also restage the GitHub Pages site from the refreshed HTML and payload',
    )
    return parser


def main() -> int:
    """Entry point for lightweight dashboard refreshes."""
    parser = build_parser()
    args = parser.parse_args()
    result = refresh_runtime_dashboard(
        year=args.year,
        payload_path=args.payload_path,
        output_html=args.output_html,
        stage_pages=args.stage_pages,
    )
    print(f'✅ Refreshed dashboard HTML at {result["html_path"]}')
    print(f'   source payload: {result["source_payload_path"]}')
    if 'runtime_dashboard_index' in result:
        print(f'   runtime dashboard: {result["runtime_dashboard_index"]}')
    if 'repo_dashboard_index' in result:
        print(f'   repo dashboard: {result["repo_dashboard_index"]}')
    if 'index_path' in result:
        print(f'   staged pages: {result["staged_index_path"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
