"""Rollback-only legacy dashboard HTML renderer.

The default dashboard path uses :mod:`ffbayes.dashboard.frontend_renderer`
(React + Vite single-file template). Import from this module only when
``FFBAYES_DASHBOARD_RENDERER=legacy`` or when testing legacy parity.

Implementation still lives in
:func:`ffbayes.draft_strategy.draft_decision_system.export_dashboard_html`
until removal after one stable draft day on the React dashboard.
See ``docs/DASHBOARD_FRONTEND_CUTOVER.md``.
"""

from __future__ import annotations

from ffbayes.draft_strategy.draft_decision_system import export_dashboard_html

__all__ = ['export_dashboard_html']
