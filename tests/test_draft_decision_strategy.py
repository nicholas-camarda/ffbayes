from __future__ import annotations

import pandas as pd
import pytest

from ffbayes.draft_strategy.draft_decision_strategy import (
    _build_current_posterior_snapshot_from_unified,
)


def test_build_current_posterior_snapshot_from_unified_requires_rookie_context():
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

    with pytest.raises(ValueError, match='missing required rookie context columns'):
        _build_current_posterior_snapshot_from_unified(history, 2026)


def test_build_current_posterior_snapshot_from_unified_carries_rookie_context():
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
                'rookie_draft_round': None,
                'rookie_draft_pick': None,
                'rookie_combine_score': None,
            },
            {
                'Season': 2025,
                'G#': 1,
                'Name': 'Veteran WR',
                'Position': 'WR',
                'FantPt': 16.0,
                'FantPtPPR': 16.0,
                'Tm': 'CIN',
                'rookie_draft_round': None,
                'rookie_draft_pick': None,
                'rookie_combine_score': None,
            },
            {
                'Season': 2025,
                'G#': 1,
                'Name': 'Rookie RB',
                'Position': 'RB',
                'FantPt': 15.0,
                'FantPtPPR': 15.0,
                'Tm': 'DET',
                'rookie_draft_round': 1,
                'rookie_draft_pick': 12,
                'rookie_combine_score': 0.72,
            },
        ]
    )

    projections = _build_current_posterior_snapshot_from_unified(history, 2026)

    assert not projections.empty
    assert {'Veteran WR', 'Rookie RB'} == set(projections['player_name'])
    assert 'rookie_draft_round' in projections.columns
    assert 'rookie_draft_pick' in projections.columns
    assert 'rookie_combine_score' in projections.columns
