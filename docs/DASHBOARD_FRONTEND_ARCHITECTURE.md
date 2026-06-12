# Dashboard Frontend Architecture

Audience: contributors changing dashboard UI, payload contract, or renderer wiring.

Scope: how the React frontend, Python payload pipeline, and HTML artifact paths
fit together after cutover.

Trust boundary: Python builds and validates the payload; the browser only
renders. Runtime `seasons/<year>/draft_strategy/` artifacts are authoritative.

## What This Is

Technical architecture for the draft war room dashboard: payload contract,
frontend build toolchain, default renderer, and legacy rollback surface.

## When To Use It

Use this guide when:

- changing `dashboard_frontend/` or the committed HTML template
- wiring new payload sections into the UI
- debugging renderer selection or Pages payload injection
- planning removal of the legacy Python HTML renderer

## What To Inspect

- `dashboard_frontend/` — React source and Vitest tests
- `src/ffbayes/dashboard/` — contract, `frontend_renderer`, `legacy_renderer`
- `src/ffbayes/dashboard/assets/dashboard_template.html` — committed build artifact
- `src/ffbayes/draft_strategy/draft_decision_system.py` — payload builder and legacy HTML (rollback only)

## Interpretation Boundaries

- This doc describes implementation layout, not draft-day operator steps (see
  `DASHBOARD_OPERATOR_GUIDE.md`).
- Legacy `export_dashboard_html()` remains for rollback; it is not the default path.

## Commands And Paths

- Rebuild template: `cd dashboard_frontend && npm ci && npm run build:template`
- Default HTML generation: `ffbayes stage-dashboard --year <year>` (no env var)
- Rollback: `FFBAYES_DASHBOARD_RENDERER=legacy ffbayes stage-dashboard --year <year>`

## Status

Cutover complete: **default renderer is `frontend`**. Roll back with
`FFBAYES_DASHBOARD_RENDERER=legacy`. Operator checklist:
`docs/DASHBOARD_FRONTEND_CUTOVER.md`.

## Layout (post-cutover)

**Payload (Python, source of truth).** `build_dashboard_payload()` in
`draft_decision_system.py` assembles the authoritative JSON. Payloads are
validated fail-closed via `ffbayes.dashboard.payload_contract` (JSON Schema at
`src/ffbayes/dashboard/schemas/dashboard_payload.schema.json`, version field
`dashboard_schema_version: 1`). Legacy payloads without a version field still
validate against the pre-existing required-key contract.

**HTML (default: React + Vite template).** `dashboard_frontend/` builds to a
single self-contained file committed at
`src/ffbayes/dashboard/assets/dashboard_template.html`. At artifact-write time,
`ffbayes.dashboard.frontend_renderer.render_dashboard_html()` injects the
payload into `<script type="application/json" id="ffbayes-dashboard-payload">`
(replacing the `__PAYLOAD_JSON__` placeholder). Node is required only when
frontend source changes, not during normal pipeline runs.

**HTML (rollback: legacy Python renderer).** `export_dashboard_html()` remains
in `draft_decision_system.py` (~3,000-line inline HTML/JS string). Import
surface for rollback paths: `ffbayes.dashboard.legacy_renderer`. Scheduled for
removal after one stable draft day on the React dashboard.

**Artifact write path.** `save_draft_decision_artifacts()` writes workbook,
payload JSON, and HTML (frontend by default). `_stage_runtime_dashboard_shortcuts()`
copies `index.html` and `dashboard_payload.json` into `<runtime>/dashboard/` and
`repo/dashboard/`.

**Pages staging.** `publish_pages.py` re-injects payload during
`stage_pages_site()`: prefers the frontend JSON script tag; falls back to legacy
`window.FFBAYES_DASHBOARD = …` markers for older HTML.

**Refresh.** `refresh_dashboard.py` reloads a payload, validates it, and
re-renders HTML through the active renderer.

**CLI.** Unchanged entry points: `ffbayes pre-draft`, `ffbayes draft-strategy`,
`ffbayes stage-dashboard`, `ffbayes refresh-dashboard`, `ffbayes draft-retrospective`.

**Tests.** Python: contract, renderer switch, refresh, publish, pipeline.
Frontend: Vitest in `dashboard_frontend/`. E2E parity:
`tests/dashboard_smoke.mjs` (Playwright) against frontend-rendered `site/`.

## Maintainability problems
1. Monolithic module: payload logic, HTML, CSS, and JS in one 7k-line file.
2. Untyped payload: `dict[str, Any]` end-to-end; contract enforced only at refresh time.
3. UI logic untestable in isolation: only greps of generated strings + full browser smoke.
4. Python/JS duplication of business presentation rules (e.g. scoring presets).
5. Every UI change requires editing a Python raw string with no syntax tooling.

## Framework comparison

| Option | Pages/static deploy | Draft-day local use | Offline | Reproducible | Typed contract | UI extensibility | Component tests | E2E tests | Long-term maintainability |
|---|---|---|---|---|---|---|---|---|---|
| Current Python-rendered HTML | yes | yes | yes | yes | no | poor (raw strings) | no | yes (smoke) | poor |
| Dash | needs server | needs server | no | yes | partial | medium | medium | medium | medium; violates no-server constraint |
| Streamlit | needs server | needs server | no | yes | no | medium | poor | medium | medium; violates no-server constraint |
| Plain JS (extracted) | yes | yes | yes | yes | no | medium | medium | yes | medium |
| **TS + React + Vite (single-file build)** | yes | yes | yes | yes (lockfile + pinned deps) | yes (schema → generated types) | high | high (Vitest) | yes (existing Playwright smoke) | high |

Decision: TypeScript + React + Vite with `vite-plugin-singlefile`. Dash/Streamlit are
disqualified by the no-server/offline constraints. Plain JS extraction loses the typed
contract and component testing. React+Vite compiled to a single HTML file keeps the
exact runtime model the repo already has.

## Architecture
- Python remains source of truth: data, models, recommendations, payload generation,
  workbook, retrospective.
- New: versioned payload contract (`dashboard_schema_version: 1`), JSON Schema at
  `src/ffbayes/dashboard/schemas/dashboard_payload.schema.json`, validated fail-closed
  before every payload write; legacy payloads (no version field) validated against the
  pre-existing required-key contract only.
- New: `dashboard_frontend/` (React+Vite) builds to ONE self-contained HTML file,
  committed as `src/ffbayes/dashboard/assets/dashboard_template.html` (packaged via
  setuptools package-data). Rationale for committing a built artifact: pipeline runtime
  must never require Node; template only changes when frontend source changes; diffs are
  reviewed via source + deterministic rebuild.
- Python `ffbayes.dashboard.frontend_renderer` injects payload into the template using
  the same `__PAYLOAD_JSON__` placeholder mechanism as the legacy renderer.
- Renderer selection: `FFBAYES_DASHBOARD_RENDERER=legacy|frontend`, default `frontend`.
  Missing template fails with an actionable message; set `legacy` to roll back.

## Migration plan
Slices A–D as in `docs/superpowers/plans/2026-06-12-dashboard-frontend-refactor.md`.
Default renderer is `frontend` as of cutover (see `docs/DASHBOARD_FRONTEND_CUTOVER.md`).

## Testing plan
- Python: `tests/test_dashboard_payload_schema.py` (contract), existing suites extended
  for renderer switch; fixtures in `tests/fixtures/`.
- Frontend: Vitest + Testing Library per panel; payload fixtures shared from
  `tests/fixtures/`.
- Parity/E2E: existing `tests/dashboard_smoke.mjs` run against frontend-rendered
  output via `FFBAYES_SMOKE_SITE_DIR`. Parity gate = full smoke suite passes.

## Compatibility plan
- All five CLI workflows unchanged: `ffbayes pre-draft`, `ffbayes pre-draft --stage-pages`, `ffbayes draft-strategy`, `ffbayes stage-dashboard --year <year>`, `ffbayes draft-retrospective`. `dashboard/index.html` shortcuts unchanged.
- `site/` staging unchanged; `publish_pages.py` injects into the frontend JSON
  script tag first, then legacy `window.FFBAYES_DASHBOARD` markers if needed.
- Legacy payloads load without `dashboard_schema_version`.

## Rollback / fallback plan
- `FFBAYES_DASHBOARD_RENDERER=legacy` → legacy Python renderer (rollback path).
- Legacy renderer code untouched; deleting it is only allowed after the default flip
  has survived a real draft.
