"""
Draft strategy module.

This package now exposes the draft decision system as the primary interface,
while keeping lightweight compatibility wrappers for older imports.
"""

from .draft_decision_strategy import (
    BayesianDraftStrategy,
    DraftConfig,
    TeamConstructionOptimizer,
    TierBasedStrategy,
    UncertaintyAwareSelector,
    main as draft_decision_main,
)

from .traditional_vor_draft import main as traditional_vor_main

__all__ = [
    'BayesianDraftStrategy',
    'TierBasedStrategy',
    'TeamConstructionOptimizer',
    'UncertaintyAwareSelector',
    'DraftConfig',
    'draft_decision_main',
    'traditional_vor_main',
]

