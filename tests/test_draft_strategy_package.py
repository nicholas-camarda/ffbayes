from __future__ import annotations

import importlib
import sys


def test_draft_strategy_package_import_does_not_preload_entrypoint_modules():
    sys.modules.pop('ffbayes.draft_strategy', None)
    sys.modules.pop('ffbayes.draft_strategy.draft_decision_strategy', None)
    sys.modules.pop('ffbayes.draft_strategy.traditional_vor_draft', None)

    package = importlib.import_module('ffbayes.draft_strategy')

    assert package.__all__ == ()
    assert 'ffbayes.draft_strategy.draft_decision_strategy' not in sys.modules
    assert 'ffbayes.draft_strategy.traditional_vor_draft' not in sys.modules
