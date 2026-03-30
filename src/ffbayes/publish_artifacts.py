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
    parser.add_argument(
        '--phase',
        choices=['pre_draft'],
        default='pre_draft',
        help='Which supported phase artifacts to publish',
    )
    args = parser.parse_args()

    results = manage_visualizations(args.year, phase=args.phase)
    print(
        f'✅ Published {len(results["copied_files"])} files for {args.year} ({args.phase})'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
