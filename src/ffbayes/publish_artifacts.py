#!/usr/bin/env python3
"""Explicit publication entrypoint for runtime draft artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime

from ffbayes.utils.visualization_manager import manage_visualizations


def main() -> int:
    """Mirror selected runtime artifacts into the cloud workspace."""
    parser = argparse.ArgumentParser(
        description='Publish selected FFBayes runtime artifacts to cloud storage'
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year, help='Season year to publish'
    )
    args = parser.parse_args()

    results = manage_visualizations(args.year, phase='pre_draft')
    print(
        '✅ Published '
        f'{len(results["synced_data_files"])} stable data files and '
        f'{len(results["published_snapshot_files"])} analysis snapshot files '
        f'for {args.year} (pre_draft)\n'
        f'   Snapshot: {results["snapshot_dir"]}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
