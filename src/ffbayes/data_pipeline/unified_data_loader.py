#!/usr/bin/env python3
"""Helpers for reading the canonical unified dataset.

The supported workflow writes the unified dataset under
`inputs/processed/unified_dataset/` as both JSON and CSV. These helpers read
that canonical output or a caller-specified override path and provide simple
selection helpers for player, position, and season slices.
"""


from pathlib import Path

import pandas as pd


def load_unified_dataset(data_directory=None):
    """Load the canonical unified dataset or an explicit override path.

    Args:
        data_directory: Optional runtime root, explicit unified dataset
            directory, or direct file path. When omitted, the canonical
            `inputs/processed/unified_dataset/` location is used.
    """
    from ffbayes.utils.path_constants import (
        get_unified_dataset_path,
    )

    if data_directory in (None, ''):
        json_path = get_unified_dataset_path()
        csv_path = json_path.with_suffix('.csv')
    else:
        candidate = Path(data_directory).expanduser()
        if candidate.is_dir():
            json_path = candidate / 'unified_dataset' / 'unified_dataset.json'
            csv_path = candidate / 'unified_dataset' / 'unified_dataset.csv'
        else:
            json_path = candidate
            csv_path = candidate.with_suffix('.csv')

    dataset_path = csv_path if csv_path.exists() else json_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Unified dataset not found at {dataset_path}. "
            "Run create_unified_dataset.py first."
        )

    if dataset_path.suffix.lower() == '.csv':
        data = pd.read_csv(dataset_path)
    else:
        data = pd.read_json(dataset_path)
    return data


def get_player_data(player_name, data=None):
    """Return all unified-dataset rows for one player."""
    if data is None:
        data = load_unified_dataset()

    player_data = data[data['Name'] == player_name]

    if len(player_data) == 0:
        raise ValueError(f"Player '{player_name}' not found in unified dataset")

    return player_data


def get_position_data(position, data=None):
    """Return all unified-dataset rows for one fantasy position."""
    if data is None:
        data = load_unified_dataset()

    position_data = data[data['Position'] == position]

    if len(position_data) == 0:
        raise ValueError(f"Position '{position}' not found in unified dataset")

    return position_data


def get_season_data(season, data=None):
    """Return all unified-dataset rows for one historical season."""
    if data is None:
        data = load_unified_dataset()

    season_data = data[data['Season'] == season]

    if len(season_data) == 0:
        raise ValueError(f"Season {season} not found in unified dataset")

    return season_data
