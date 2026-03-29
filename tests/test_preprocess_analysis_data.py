from __future__ import annotations

import pandas as pd

from ffbayes.data_pipeline.preprocess_analysis_data import _encode_team_codes


def test_encode_team_codes_handles_unknown_values():
    team_names = pd.Index(['NYG', 'DAL'])
    values = pd.Series(['NYG', 'DAL', None, 'MIA'])

    encoded = _encode_team_codes(values, team_names)

    assert list(encoded) == [0, 1, -1, -1]
