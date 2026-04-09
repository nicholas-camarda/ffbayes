#!/usr/bin/env python3
"""Regenerate dashboard HTML from an existing dashboard payload."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

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

REQUIRED_PAYLOAD_KEYS = ('generated_at', 'league_settings', 'decision_table')


def _validate_dashboard_payload(
    payload: dict[str, Any], payload_path: Path
) -> dict[str, Any]:
    missing = [
        key for key in REQUIRED_PAYLOAD_KEYS if key not in payload or payload.get(key) is None
    ]
    if missing:
        raise ValueError(
            f'Dashboard payload at {payload_path} is missing required keys: '
            f'{", ".join(missing)}'
        )
    if not isinstance(payload.get('decision_table'), list):
        raise ValueError(
            f'Dashboard payload at {payload_path} has a non-list `decision_table`.'
        )
    if not isinstance(payload.get('league_settings'), dict):
        raise ValueError(
            f'Dashboard payload at {payload_path} has an invalid `league_settings` object.'
        )
    return payload


def load_dashboard_payload(payload_path: Path | str) -> dict[str, Any]:
    """Load and validate a dashboard payload."""
    resolved_payload = Path(payload_path)
    if not resolved_payload.exists():
        raise FileNotFoundError(
            f'Dashboard payload not found at {resolved_payload}. '
            'Run `ffbayes draft-strategy` first.'
        )
    payload = json.loads(resolved_payload.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(
            f'Dashboard payload at {resolved_payload} must be a JSON object.'
        )
    return _validate_dashboard_payload(payload, resolved_payload)


def _generated_label_from_payload(payload: dict[str, Any]) -> str:
    generated_at = payload.get('generated_at')
    if generated_at:
        try:
            return datetime.fromisoformat(str(generated_at)).strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def _normalized_json_text(payload: Any) -> str:
    return json.dumps(
        json.loads(json.dumps(payload, default=str)),
        sort_keys=True,
        indent=2,
    )


def _infer_surface_kind(output_html: Path, source_payload: Path) -> str:
    from ffbayes.utils.path_constants import get_project_root, get_runtime_root

    try:
        project_root = get_project_root().resolve()
    except Exception:
        project_root = None
    try:
        runtime_root = get_runtime_root().resolve()
    except Exception:
        runtime_root = None

    resolved_html = output_html.resolve()
    resolved_payload = source_payload.resolve()
    if resolved_html.parent == resolved_payload.parent:
        return 'canonical_runtime'
    if project_root is not None and resolved_html.parent == (project_root / 'dashboard'):
        return 'repo_shortcut'
    if runtime_root is not None and resolved_html.parent == (runtime_root / 'dashboard'):
        return 'runtime_shortcut'
    if project_root is not None and resolved_html.parent == (project_root / 'site'):
        return 'staged_site'
    return 'derived_surface'


def _paired_payload_path(
    output_html: Path,
    year: int,
    authoritative_payload: Path,
) -> Path:
    if output_html.parent == authoritative_payload.parent:
        return authoritative_payload
    if output_html.name == 'index.html':
        return output_html.with_name('dashboard_payload.json')
    if output_html.name == f'draft_board_{year}.html':
        return output_html.with_name(f'dashboard_payload_{year}.json')
    return output_html.with_name('dashboard_payload.json')


def _render_dashboard_html_text(
    payload: dict[str, Any],
    output_html: Path,
    year: int,
) -> str:
    league_settings = LeagueSettings.from_mapping(
        {'league_settings': payload.get('league_settings', {})}
    )
    with tempfile.TemporaryDirectory(prefix='ffbayes-refresh-dashboard-') as tmpdir:
        temp_path = Path(tmpdir) / output_html.name
        export_dashboard_html(
            pd.DataFrame(payload.get('decision_table', [])),
            pd.DataFrame(payload.get('recommendation_summary', [])),
            temp_path,
            league_settings,
            backtest=payload.get('backtest', {}),
            source_freshness=pd.DataFrame(payload.get('source_freshness', [])),
            dashboard_payload=payload,
            generated_label=_generated_label_from_payload(payload),
        )
        return temp_path.read_text(encoding='utf-8')


def check_dashboard_freshness(
    year: int | None = None,
    payload_path: Path | str | None = None,
    output_html: Path | str | None = None,
) -> dict[str, Any]:
    """Check whether a dashboard HTML target matches regeneration from payload."""
    resolved_year = year or datetime.now().year
    resolved_payload = (
        Path(payload_path)
        if payload_path is not None
        else get_dashboard_payload_path(resolved_year)
    )
    payload = load_dashboard_payload(resolved_payload)
    resolved_output_html = (
        Path(output_html)
        if output_html is not None
        else get_dashboard_html_path(resolved_year)
    )

    result: dict[str, Any] = {
        'status': 'fresh',
        'checked_paths': [str(resolved_output_html)],
        'stale_paths': [],
        'source_payload_path': str(resolved_payload),
        'target_html_path': str(resolved_output_html),
        'target_payload_path': None,
        'generated_at': payload.get('generated_at'),
        'surface_kind': _infer_surface_kind(resolved_output_html, resolved_payload),
        'authoritative_paths': {
            'payload': str(resolved_payload),
            'target_html': str(resolved_output_html),
        },
        'mutated': False,
    }

    if not resolved_output_html.exists():
        result['status'] = 'missing_target'
        result['stale_paths'] = [str(resolved_output_html)]
        return result

    expected_html = _render_dashboard_html_text(
        payload,
        resolved_output_html,
        resolved_year,
    )
    current_html = resolved_output_html.read_text(encoding='utf-8')
    if current_html != expected_html:
        result['status'] = 'stale'
        result['stale_paths'] = [str(resolved_output_html)]

    target_payload_path = _paired_payload_path(
        resolved_output_html,
        resolved_year,
        resolved_payload,
    )
    result['target_payload_path'] = str(target_payload_path)
    if target_payload_path.resolve() != resolved_payload.resolve():
        result['checked_paths'].append(str(target_payload_path))
        if not target_payload_path.exists():
            result['status'] = 'stale'
            result['stale_paths'] = list(
                dict.fromkeys(result['stale_paths'] + [str(target_payload_path)])
            )
            return result
        target_payload = json.loads(target_payload_path.read_text(encoding='utf-8'))
        if _normalized_json_text(target_payload) != _normalized_json_text(payload):
            result['status'] = 'stale'
            result['stale_paths'] = list(
                dict.fromkeys(result['stale_paths'] + [str(target_payload_path)])
            )
    return result


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def refresh_runtime_dashboard(
    year: int | None = None,
    payload_path: Path | str | None = None,
    output_html: Path | str | None = None,
    stage_pages: bool = False,
) -> dict[str, Any]:
    """Rebuild dashboard HTML from the current runtime payload only."""
    resolved_year = year or datetime.now().year
    resolved_payload = (
        Path(payload_path)
        if payload_path is not None
        else get_dashboard_payload_path(resolved_year)
    )
    payload = load_dashboard_payload(resolved_payload)
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
        generated_label=_generated_label_from_payload(payload),
    )

    result: dict[str, Any] = {
        'status': 'refreshed',
        'source_payload_path': resolved_payload,
        'html_path': resolved_output_html,
        'checked_paths': [str(resolved_output_html)],
        'stale_paths': [],
        'mutated': True,
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
        result['status'] = 'refreshed'
        result['staged_payload_path'] = staged['payload_path']
        result['staged_index_path'] = staged['index_path']
        result['staged_provenance_path'] = staged['provenance_path']
        result['checked_paths'].extend(
            [str(staged['index_path']), str(staged['payload_path'])]
        )
        result['stale_paths'] = [str(path) for path in staged.get('stale_paths', [])]

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Regenerate dashboard HTML from the existing runtime payload. '
            'Use `ffbayes stage-dashboard` when you also want to restage `site/`.'
        )
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
        help=(
            'Also restage the GitHub Pages site from the refreshed HTML and payload. '
            'For the normal one-step operator path, prefer `ffbayes stage-dashboard`.'
        ),
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check whether the target HTML matches regeneration from the current payload without mutating files',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit machine-readable JSON output',
    )
    return parser


def main() -> int:
    """Entry point for lightweight dashboard refreshes."""
    parser = build_parser()
    args = parser.parse_args()
    if args.check and args.stage_pages:
        parser.error('--check cannot be combined with --stage-pages')

    try:
        if args.check:
            result = check_dashboard_freshness(
                year=args.year,
                payload_path=args.payload_path,
                output_html=args.output_html,
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                symbol = '✅' if result['status'] == 'fresh' else '⚠️'
                print(
                    f'{symbol} Dashboard check status: {result["status"]}\n'
                    f'   surface: {result["surface_kind"]}\n'
                    f'   source payload: {result["source_payload_path"]}\n'
                    f'   target html: {result["target_html_path"]}'
                )
                if result.get('target_payload_path'):
                    print(f'   target payload: {result["target_payload_path"]}')
                if result['stale_paths']:
                    print('   stale paths:')
                    for path in result['stale_paths']:
                        print(f'   - {path}')
            return 0 if result['status'] == 'fresh' else 1

        result = refresh_runtime_dashboard(
            year=args.year,
            payload_path=args.payload_path,
            output_html=args.output_html,
            stage_pages=args.stage_pages,
        )
        if args.json:
            print(json.dumps(_to_jsonable(result), indent=2))
        else:
            print(f'✅ Refreshed dashboard HTML at {result["html_path"]}')
            print(f'   source payload: {result["source_payload_path"]}')
            print(f'   checked paths: {", ".join(result["checked_paths"])}')
            if result['stale_paths']:
                print(f'   stale paths replaced: {", ".join(result["stale_paths"])}')
            if 'runtime_dashboard_index' in result:
                print(f'   runtime dashboard: {result["runtime_dashboard_index"]}')
            if 'repo_dashboard_index' in result:
                print(f'   repo dashboard: {result["repo_dashboard_index"]}')
            if 'staged_index_path' in result:
                print(f'   staged pages: {result["staged_index_path"]}')
        return 0
    except (FileNotFoundError, ValueError) as exc:
        error_payload = {'status': 'error', 'error': str(exc), 'mutated': False}
        if args.json:
            print(json.dumps(error_payload, indent=2))
        else:
            print(f'Error: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
