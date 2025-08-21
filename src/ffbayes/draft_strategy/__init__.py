"""
Draft Strategy Module

This module provides advanced draft strategy functionality for fantasy football,
including tier-based Bayesian approaches and traditional VOR-based strategies.
"""

# Traditional VOR-based draft strategy
from .traditional_vor_draft import main as traditional_vor_main

# Use lazy imports to avoid the RuntimeWarning when running the module as a script
# This prevents the module from being imported into sys.modules before execution

def _import_classes():
    """Lazy import of classes to avoid import conflicts."""
    from .bayesian_draft_strategy import (
        BayesianDraftStrategy,
        DraftConfig,
        TeamConstructionOptimizer,
        TierBasedStrategy,
        UncertaintyAwareSelector,
    )
    return (
        BayesianDraftStrategy,
        DraftConfig,
        TeamConstructionOptimizer,
        TierBasedStrategy,
        UncertaintyAwareSelector,
    )

# Define __all__ for explicit exports
__all__ = [
    'BayesianDraftStrategy',
    'TierBasedStrategy', 
    'TeamConstructionOptimizer',
    'UncertaintyAwareSelector',
    'DraftConfig',
    'traditional_vor_main'
]

# Lazy import implementation
def __getattr__(name):
    """Lazy import implementation for Python 3.7+."""
    if name in __all__:
        classes = _import_classes()
        class_map = {
            'BayesianDraftStrategy': classes[0],
            'DraftConfig': classes[1],
            'TeamConstructionOptimizer': classes[2],
            'TierBasedStrategy': classes[3],
            'UncertaintyAwareSelector': classes[4],
        }
        return class_map[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
