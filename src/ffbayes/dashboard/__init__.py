"""Dashboard payload contract, frontend rendering, and legacy rollback.

Modules:

- :mod:`payload_contract` — JSON Schema validation before payload writes
- :mod:`frontend_renderer` — default React+Vite single-file HTML renderer
- :mod:`legacy_renderer` — rollback import surface for the Python HTML renderer

See ``docs/DASHBOARD_FRONTEND_ARCHITECTURE.md`` and
``docs/DASHBOARD_FRONTEND_CUTOVER.md``.
"""
