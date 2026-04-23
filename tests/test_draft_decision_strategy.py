from __future__ import annotations

import pandas as pd

from ffbayes.draft_strategy.draft_decision_strategy import (
    _build_current_posterior_snapshot_from_unified,
)


def test_build_current_posterior_snapshot_from_unified_tolerates_missing_rookie_columns():
    history = pd.DataFrame(
        [
            {
                'Season': 2024,
                'G#': 1,
                'Name': 'Veteran WR',
                'Position': 'WR',
                'FantPt': 14.0,
                'FantPtPPR': 14.0,
                'Tm': 'CIN',
            },
            {
                'Season': 2025,
                'G#': 1,
                'Name': 'Veteran WR',
                'Position': 'WR',
                'FantPt': 16.0,
                'FantPtPPR': 16.0,
                'Tm': 'CIN',
            },
            {
                'Season': 2025,
                'G#': 1,
                'Name': 'Stable RB',
                'Position': 'RB',
                'FantPt': 15.0,
                'FantPtPPR': 15.0,
                'Tm': 'DET',
            },
        ]
    )

    projections = _build_current_posterior_snapshot_from_unified(history, 2026)

    assert not projections.empty
    assert {'Veteran WR', 'Stable RB'} == set(projections['player_name'])
    assert 'rookie_draft_round' in projections.columns
    assert 'rookie_draft_pick' in projections.columns
    assert 'rookie_combine_score' in projections.columns
