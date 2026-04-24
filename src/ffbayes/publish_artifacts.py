#!/usr/bin/env python3
"""Explicit publication entrypoint for public FFBayes artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime

from ffbayes.stage_dashboard import stage_dashboard
from ffbayes.utils.visualization_manager import manage_visualizations


def main() -> int:
    """Stage GitHub Pages and mirror selected runtime artifacts into cloud."""
    parser = argparse.ArgumentParser(
        description='Publish selected FFBayes runtime artifacts to public surfaces'
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to publish'
    )
    args = parser.parse_args()

    pages = stage_dashboard(year=args.year)
    results = manage_visualizations(args.year, phase='pre_draft')
    print(
        '✅ Published public surfaces for '
        f'{args.year}\n'
        f'   GitHub Pages: {pages.get("staged_index_path")}\n'
        f'{len(results["synced_data_files"])} stable data files and '
        f'{len(results["published_snapshot_files"])} analysis snapshot files '
        f'to cloud storage\n'
        f'   Cloud snapshot: {results["snapshot_dir"]}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
