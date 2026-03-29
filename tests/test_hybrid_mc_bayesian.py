from __future__ import annotations

import numpy as np
import pandas as pd

from ffbayes.analysis.hybrid_mc_bayesian import HybridMCBayesianModel


def test_add_bayesian_uncertainty_layers_handles_missing_vor_rank(monkeypatch):
    model = HybridMCBayesianModel()
    model.data = pd.DataFrame(
        [
            {
                'Name': 'Alpha Player',
                'FantPt': 12.0,
                'Position': 'RB',
                'Season': 2025,
                'vor_global_rank': np.nan,
            }
        ]
    )
    model.monte_carlo_results = {
        'Alpha Player': {
            'monte_carlo': {'mean': 11.0, 'std': 2.0},
            'position': 'RB',
        }
    }

    monkeypatch.setattr(model, '_train_uncertainty_model', lambda: None)
    monkeypatch.setattr(model, '_predict_uncertainty', lambda features: 0.25)

    results = model.add_bayesian_uncertainty_layers()

    assert results['Alpha Player']['vor_validation']['global_rank'] == 121
    assert results['Alpha Player']['vor_validation']['rank_tier'] == 'Low'
