from __future__ import annotations

import pandas as pd

from ffbayes.data_pipeline.preprocess_analysis_data import (
    POSITION_ID_MAP,
    _encode_team_codes,
)


def test_encode_team_codes_handles_unknown_values():
    team_names = pd.Index(['NYG', 'DAL'])
    values = pd.Series(['NYG', 'DAL', None, 'MIA'])

    encoded = _encode_team_codes(values, team_names)

    assert list(encoded) == [0, 1, -1, -1]


def test_position_id_map_is_stable_for_fantasy_positions():
    assert POSITION_ID_MAP == {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'DST': 4, 'K': 5}
