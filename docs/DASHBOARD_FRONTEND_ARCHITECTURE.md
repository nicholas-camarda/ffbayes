# Dashboard Frontend Architecture

## Clean-start confirmation
- Working tree verified clean before branching.
- Branch: `refactor/ffbayes-dashboard-frontend`
- Baseline: `master` @ `89c61a3e7bb992508ddfb6a5aacd91a168a2741d`

## Current-state audit

The interactive draft dashboard is produced entirely inside `src/ffbayes/draft_strategy/draft_decision_system.py` (7,103 lines). Payload construction, HTML rendering, workbook export, and shortcut staging all live in this single module.

**Payload generation.** `build_dashboard_payload()` (`draft_decision_system.py:3094–3250`) assembles a JSON-serializable `dict[str, Any]` from decision tables, recommendations, tier cliffs, roster scenarios, scoring presets, war-room visuals, and supporting metadata. The return type is untyped beyond `dict[str, Any]`; there is no schema version field today.

**Artifact write path.** `save_draft_decision_artifacts()` (`draft_decision_system.py:6876–7103`) writes the full artifact bundle: Excel workbook via `export_workbook()`, payload JSON at lines 6945–6947, and HTML via `export_dashboard_html()` at lines 6949–6957. After writing, `_stage_runtime_dashboard_shortcuts()` (`draft_decision_system.py:6780–6873`) copies `index.html` and `dashboard_payload.json` into convenience shortcuts under `<runtime_root>/dashboard/` and `<repo>/dashboard/` using `shutil.copy2`.

**HTML renderer.** `export_dashboard_html()` (`draft_decision_system.py:3613–6657`) embeds a ~3,000-line raw Python string (`html = r"""` at line 3650 through closing `"""` at line 6651). Client-side logic is vanilla JavaScript inside a `<script>` block from lines 4568–6647 (~2,080 lines). The payload is injected at build time by replacing the `__PAYLOAD_JSON__` placeholder (line 4569); `__GENERATED_LABEL__` is substituted at lines 6652–6654.

**Payload injection for publish.** `src/ffbayes/publish_pages.py` re-injects payload into staged HTML using string markers: `PAYLOAD_ASSIGNMENT_PREFIX = 'window.FFBAYES_DASHBOARD = '` and `PAYLOAD_ASSIGNMENT_SUFFIX = ';\n\n    (() => {'` (lines 22–23). `_inject_dashboard_payload_into_html()` (lines 94–103) finds these markers and splices in fresh JSON during `stage_pages_site()` (lines 107–222).

**Artifact authority model.** Canonical season artifacts resolve under `seasons/<year>/draft_strategy/` via `get_draft_strategy_dir()` → `get_pre_draft_artifacts_dir()` → `get_run_root()` (`path_constants.py:100–115`, `118–127`, `164–168`). Payload and HTML paths are `get_dashboard_payload_path()` and `get_dashboard_html_path()` (`path_constants.py:187–202`), yielding `dashboard_payload_<year>.json` and `draft_board_<year>.html`. Derived shortcuts land at `<runtime>/dashboard/` and `<repo>/dashboard/` (`draft_decision_system.py:6785–6791`, `6825–6861`). Published output is staged to `<repo>/site/` via `get_pages_site_dir()` (`path_constants.py:408–412`).

**Refresh and validation.** `src/ffbayes/refresh_dashboard.py` loads an existing payload and re-renders HTML. Required-key validation is hand-rolled: `REQUIRED_PAYLOAD_KEYS` and `_validate_dashboard_payload()` (lines 33–91) check four top-level keys plus nested `decision_evidence` freshness rules. `load_dashboard_payload()` (lines 94–107) reads JSON and calls the validator. No JSON Schema or generated types exist today.

**CLI wiring.** Five workflows remain entry-point driven via `src/ffbayes/cli.py`. Pages staging hooks through `run_pipeline_split.py:379–387` (`--stage-pages` → `stage_dashboard()`), and `src/ffbayes/stage_dashboard.py` wraps `refresh_runtime_dashboard(..., stage_pages=True)`.

**Tests today.** Python coverage includes string/content asserts in `tests/test_draft_decision_system.py` (payload shape, scoring presets, war-room visuals), `tests/test_refresh_dashboard.py` (rebuild + staging), `tests/test_publish_pages.py` (injection markers), `tests/test_run_pipeline_split.py`, and `tests/test_documentation_contracts.py`. End-to-end browser coverage is `tests/dashboard_smoke.mjs` (686-line Playwright suite) invoked from `tests/test_dashboard_smoke.py`; there are no isolated frontend unit or component tests.

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
- Renderer selection: `FFBAYES_DASHBOARD_RENDERER=legacy|frontend`, default `legacy`.
  Selecting `frontend` without a built template fails with an actionable message.

## Migration plan
Slices A–D as in `docs/superpowers/plans/2026-06-12-dashboard-frontend-refactor.md`.
Default-flip to `frontend` is a follow-up PR after a draft-day dry run.

## Testing plan
- Python: `tests/test_dashboard_payload_schema.py` (contract), existing suites extended
  for renderer switch; fixtures in `tests/fixtures/`.
- Frontend: Vitest + Testing Library per panel; payload fixtures shared from
  `tests/fixtures/`.
- Parity/E2E: existing `tests/dashboard_smoke.mjs` run against frontend-rendered
  output via `FFBAYES_SMOKE_SITE_DIR`. Parity gate = full smoke suite passes.

## Compatibility plan
- All five CLI workflows unchanged: `ffbayes pre-draft`, `ffbayes pre-draft --stage-pages`, `ffbayes draft-strategy`, `ffbayes stage-dashboard --year <year>`, `ffbayes draft-retrospective`. `dashboard/index.html` shortcuts unchanged.
- `site/` staging unchanged; `publish_pages.py` injection gains a second suffix marker
  (`; /*FFBAYES_PAYLOAD_END*/`) tried before the legacy suffix.
- Legacy payloads load without `dashboard_schema_version`.

## Rollback / fallback plan
- `FFBAYES_DASHBOARD_RENDERER` unset/`legacy` → byte-identical legacy behavior.
- Legacy renderer code untouched; deleting it is only allowed after the default flip
  has survived a real draft.
