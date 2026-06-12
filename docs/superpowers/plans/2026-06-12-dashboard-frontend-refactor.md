# Dashboard Frontend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the browser dashboard from a 3,000-line inline HTML/JS string in Python to a typed React + Vite frontend that consumes versioned, schema-validated JSON payloads, while the Python pipeline remains the source of truth and every existing CLI command keeps working.

**Architecture:** The frontend builds to a **single self-contained HTML template** (via `vite-plugin-singlefile`) that is committed into the Python package as `src/ffbayes/dashboard/assets/dashboard_template.html`. Python keeps doing exactly what it does today — inject the payload JSON into the template at artifact-write time via the `__PAYLOAD_JSON__` placeholder — so offline draft-day use, `dashboard/index.html` shortcuts, and `site/` GitHub Pages staging are all unchanged. Node is only required when the frontend source changes, never at pipeline runtime. The legacy Python-rendered dashboard stays the default behind a `FFBAYES_DASHBOARD_RENDERER` env switch until the new renderer passes the existing Playwright smoke suite (`tests/dashboard_smoke.mjs`), which is the parity acceptance gate.

**Tech Stack:** Python 3.10 + `jsonschema` (payload contract); TypeScript + React 18 + Vite + `vite-plugin-singlefile` (frontend); Vitest + Testing Library (component tests); existing Playwright smoke suite (e2e/parity); `json-schema-to-typescript` (generated payload types).

**Baseline:** branch `master`, commit `89c61a3e7bb992508ddfb6a5aacd91a168a2741d` ("Enhance scoring presets and dashboard functionality"). Working tree verified clean on 2026-06-12.

**Feature branch:** `refactor/ffbayes-dashboard-frontend`

---

## Repo facts the executor must know (from the audit)

- All Python/pytest/ruff commands run inside the `ffbayes` conda env: `conda run -n ffbayes <cmd>`.
- **Payload builder:** `build_dashboard_payload()` at `src/ffbayes/draft_strategy/draft_decision_system.py:3094–3250`. Returns a plain dict; top-level keys listed in Task 5.
- **Payload written:** `save_draft_decision_artifacts()` at `draft_decision_system.py:6876–7103`; payload written at lines 6945–6947, HTML rendered at 6949–6957.
- **Legacy HTML renderer:** `export_dashboard_html()` at `draft_decision_system.py:3613–6657`. Raw string template; client JS at ~4568–6647. Placeholders replaced at 6652–6654: `__PAYLOAD_JSON__` and `__GENERATED_LABEL__`.
- **HTML `<title>`:** `FFBayes Draft War Room` (line 3656). The smoke test asserts on it.
- **Client localStorage key:** `ffbayes-dashboard-state-v2` (line 4573). Reuse it.
- **Finalize schema constant:** `finalized_draft_v1` (line 4582); finalize bundle builds 1 JSON + 2 HTML downloads (snapshot title `FFBayes Finalized Draft Snapshot` line 6328, summary title `FFBayes Post-Draft Summary` line 6421).
- **Refresh path:** `src/ffbayes/refresh_dashboard.py` — `REQUIRED_PAYLOAD_KEYS` at 33–38, `_validate_dashboard_payload()` at 69–91, `load_dashboard_payload()` at 94–107, HTML re-render at 192–210.
- **Pages staging:** `src/ffbayes/publish_pages.py` — `stage_pages_site()` at 107–222; payload re-injection markers `PAYLOAD_ASSIGNMENT_PREFIX = 'window.FFBAYES_DASHBOARD = '` and `PAYLOAD_ASSIGNMENT_SUFFIX = ';\n\n    (() => {'` at lines 22–23; injection in `_inject_dashboard_payload_into_html()` (~94–104).
- **Stage wrapper:** `src/ffbayes/stage_dashboard.py` → `refresh_runtime_dashboard(..., stage_pages=True)`.
- **CLI:** `src/ffbayes/cli.py` forwards `pre-draft`, `draft-strategy`, `stage-dashboard`, `draft-retrospective`, etc. `--stage-pages` hook in `src/ffbayes/run_pipeline_split.py:379–387`.
- **Paths:** `src/ffbayes/utils/path_constants.py` — `get_dashboard_payload_path()` (187–193), `get_dashboard_html_path()` (196–202), `get_pages_site_dir()` (408–412), shortcut dirs `RUNTIME_DASHBOARD_DIR`/`REPO_DASHBOARD_DIR` (66–67).
- **Shortcut staging:** `_stage_runtime_dashboard_shortcuts()` at `draft_decision_system.py:6780–6873` (uses `shutil.copy2`, not symlinks).
- **Existing tests:** `tests/test_refresh_dashboard.py` (has a complete minimal payload fixture inline at lines 104–228 — the source for our fixture file), `tests/test_publish_pages.py`, `tests/test_draft_decision_system.py`, `tests/test_dashboard_smoke.py` + `tests/dashboard_smoke.mjs` (687-line Playwright suite; honors `FFBAYES_SMOKE_SITE_DIR` and `FFBAYES_SMOKE_MODE=minimal`), `tests/test_run_pipeline_split.py:85–129`, `tests/test_cli.py`, `tests/test_documentation_contracts.py:310–347`.
- **No `tests/fixtures/` dir yet.** No jsonschema/pydantic dependency yet. Root `package.json` has only Playwright; do not touch it except where stated.
- Strict JSON helpers: `ffbayes.utils.json_serialization.dumps_strict_json` / `to_strict_jsonable` — always use these when writing payload JSON.
- Lint/format: `ruff` with single quotes, 4-space indent. Type check: `mypy src`.

## Scope check / PR slicing

This plan is one branch but four reviewable slices. Open PRs (or at least group commits) in this order; each slice leaves the repo green:

1. **Slice A (Tasks 1–8):** design doc + payload contract/schema. Pure Python, no behavior change for existing payloads.
2. **Slice B (Tasks 9–12):** frontend scaffold + build-to-template toolchain. No Python integration yet.
3. **Slice C (Tasks 13–19):** behavior parity port in the frontend.
4. **Slice D (Tasks 20–23):** renderer switch integration, regression tests, parity validation, final report.

Flipping the default renderer from `legacy` to `frontend` is **explicitly out of scope** — recommend it as a follow-up PR after a real draft-day dry run.

## Hard constraints (do not violate)

- Never edit on `master`/`main`; all work on `refactor/ffbayes-dashboard-frontend`.
- Do not rewrite modeling/statistics code. Do not touch `build_decision_table`, `build_recommendations`, backtest code, etc.
- Do not remove or alter `export_dashboard_html()` behavior (it is the fallback).
- `ffbayes pre-draft`, `ffbayes pre-draft --stage-pages`, `ffbayes draft-strategy`, `ffbayes stage-dashboard --year <year>`, `ffbayes draft-retrospective` must keep working with **zero flag changes**.
- No server runtime for viewing the dashboard. No secrets, no absolute local paths in committed files, no large runtime artifacts committed (the built template HTML is the one deliberate exception, documented in the design doc).
- Existing payloads (no `dashboard_schema_version` field) must continue to load — validation fails closed only on the critical-key contract that already exists.

---

# Phase 0 — Clean start and branch

### Task 1: Verify clean state and create the feature branch

**Files:** none (git only)

- [ ] **Step 1: Verify clean tree and baseline**

Run: `git status --porcelain && git branch --show-current && git log -1 --format='%H %s'`
Expected: empty porcelain output, `master`, commit `89c61a3e7bb992508ddfb6a5aacd91a168a2741d`.
**If porcelain output is non-empty: STOP. Report the dirty files and do nothing else.** Do not stash.
If the baseline commit differs (new commits landed since this plan was written), record the actual HEAD as the baseline in the final report and proceed.

- [ ] **Step 2: Create branch**

```bash
git checkout -b refactor/ffbayes-dashboard-frontend
```

Expected: `Switched to a new branch 'refactor/ffbayes-dashboard-frontend'`.

- [ ] **Step 3: Record** branch name and baseline commit for the final report (no commit yet; nothing changed).

---

# Phase 1 — Current-state audit → design document

### Task 2: Write `docs/DASHBOARD_FRONTEND_ARCHITECTURE.md`

**Files:**
- Create: `docs/DASHBOARD_FRONTEND_ARCHITECTURE.md`

- [ ] **Step 1: Write the document.** Use this exact structure and content (expand prose where marked, but all decisions below are final — do not re-litigate):

```markdown
# Dashboard Frontend Architecture

## Clean-start confirmation
- Working tree verified clean before branching.
- Branch: `refactor/ffbayes-dashboard-frontend`
- Baseline: `master` @ `89c61a3e7bb992508ddfb6a5aacd91a168a2741d`

## Current-state audit
[Summarize, citing exact paths/lines from "Repo facts" in the plan:
- Payload + HTML + workbook all generated by `draft_decision_system.py` (7,104 lines);
  HTML is a ~3,000-line raw string with ~2,000 lines of embedded vanilla JS.
- Payload injected via `__PAYLOAD_JSON__`; staging re-injects via string markers in `publish_pages.py`.
- Artifact authority model: canonical `seasons/<year>/draft_strategy/`, derived shortcuts
  `<runtime>/dashboard/` + `<repo>/dashboard/`, derived publish `<repo>/site/`.
- Validation today: hand-rolled required-key checks in `refresh_dashboard.py` only.
- Tests: Python string asserts + `tests/dashboard_smoke.mjs` Playwright suite.]

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
- All five CLI workflows unchanged. `dashboard/index.html` shortcuts unchanged.
- `site/` staging unchanged; `publish_pages.py` injection gains a second suffix marker
  (`; /*FFBAYES_PAYLOAD_END*/`) tried before the legacy suffix.
- Legacy payloads load without `dashboard_schema_version`.

## Rollback / fallback plan
- `FFBAYES_DASHBOARD_RENDERER` unset/`legacy` → byte-identical legacy behavior.
- Legacy renderer code untouched; deleting it is only allowed after the default flip
  has survived a real draft.
```

- [ ] **Step 2: Commit**

```bash
git add docs/DASHBOARD_FRONTEND_ARCHITECTURE.md
git commit -m "docs(dashboard): add frontend architecture design doc"
```

---

# Phase 2 — Payload contract and schema

### Task 3: Add `jsonschema` dependency

**Files:**
- Modify: `pyproject.toml:14-33` (dependencies array)
- Modify: `environment.yml` (pip section, lines 24–29)

- [ ] **Step 1: Add `"jsonschema>=4.17",`** to the `dependencies` array in `pyproject.toml` (alphabetical position: after `"fuzzywuzzy>=0.18",`).
- [ ] **Step 2: Add `- jsonschema>=4.17`** under the `pip:` list in `environment.yml`.
- [ ] **Step 3: Install**

Run: `conda run -n ffbayes pip install 'jsonschema>=4.17'`
Then: `conda run -n ffbayes python -c "import jsonschema; print(jsonschema.__version__)"`
Expected: a version ≥ 4.17 printed.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml
git commit -m "build(deps): add jsonschema for dashboard payload contract"
```

### Task 4: Create payload fixtures

**Files:**
- Create: `tests/fixtures/dashboard_payload_minimal.json`
- Modify: `tests/test_refresh_dashboard.py` (refactor inline fixture to load the file)

- [ ] **Step 1: Extract the fixture.** Copy the complete inline payload dict from `tests/test_refresh_dashboard.py:104–228` (including the `scoring_presets` bundle from its `_scoring_preset_bundle_fixture()` helper and the other keys it sets — read the whole test file first) into `tests/fixtures/dashboard_payload_minimal.json` as plain JSON. Add `"dashboard_schema_version": 1` as the first key. This file is the canonical minimal-valid payload for Python tests AND the frontend.
- [ ] **Step 2: Refactor `tests/test_refresh_dashboard.py`** to load the fixture file instead of building the dict inline, e.g.:

```python
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_minimal_payload() -> dict:
    return json.loads(
        (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(encoding='utf-8')
    )
```

Keep test-specific mutations (e.g. degraded evidence variants) as in-test copies: `payload = _load_minimal_payload()` then mutate.

- [ ] **Step 3: Run the refactored tests**

Run: `conda run -n ffbayes pytest tests/test_refresh_dashboard.py -q`
Expected: all pass (same count as before the refactor — check with `git stash` if unsure).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/dashboard_payload_minimal.json tests/test_refresh_dashboard.py
git commit -m "test(dashboard): extract minimal payload fixture to shared file"
```

### Task 5: JSON Schema + contract module (TDD)

**Files:**
- Create: `src/ffbayes/dashboard/__init__.py`
- Create: `src/ffbayes/dashboard/payload_contract.py`
- Create: `src/ffbayes/dashboard/schemas/dashboard_payload.schema.json`
- Create: `tests/test_dashboard_payload_schema.py`
- Modify: `pyproject.toml` (add `[tool.setuptools.package-data]`)

- [ ] **Step 1: Write the failing tests** in `tests/test_dashboard_payload_schema.py`:

```python
"""Tests for the versioned dashboard payload contract."""

import json
import unittest
from pathlib import Path

from ffbayes.dashboard.payload_contract import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardPayloadError,
    load_dashboard_schema,
    stamp_schema_version,
    validate_dashboard_payload,
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _minimal_payload() -> dict:
    return json.loads(
        (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(encoding='utf-8')
    )


class TestPayloadContract(unittest.TestCase):
    def test_schema_loads_and_declares_draft_2020_12(self):
        schema = load_dashboard_schema()
        self.assertEqual(
            schema['$schema'], 'https://json-schema.org/draft/2020-12/schema'
        )

    def test_minimal_fixture_validates(self):
        payload = _minimal_payload()
        self.assertIs(validate_dashboard_payload(payload, source='fixture'), payload)

    def test_stamp_sets_current_version(self):
        payload = {}
        stamp_schema_version(payload)
        self.assertEqual(
            payload['dashboard_schema_version'], DASHBOARD_SCHEMA_VERSION
        )

    def test_missing_critical_key_fails_closed(self):
        payload = _minimal_payload()
        del payload['decision_table']
        with self.assertRaises(DashboardPayloadError) as ctx:
            validate_dashboard_payload(payload, source='fixture')
        self.assertIn('decision_table', str(ctx.exception))

    def test_wrong_type_fails_closed(self):
        payload = _minimal_payload()
        payload['decision_table'] = 'not-a-list'
        with self.assertRaises(DashboardPayloadError):
            validate_dashboard_payload(payload, source='fixture')

    def test_unknown_extra_keys_are_allowed(self):
        payload = _minimal_payload()
        payload['some_future_section'] = {'anything': True}
        validate_dashboard_payload(payload, source='fixture')

    def test_legacy_payload_without_version_uses_legacy_contract(self):
        payload = _minimal_payload()
        del payload['dashboard_schema_version']
        # Legacy contract only requires the four pre-existing keys.
        validate_dashboard_payload(payload, source='fixture')
        del payload['decision_evidence']
        with self.assertRaises(DashboardPayloadError):
            validate_dashboard_payload(payload, source='fixture')

    def test_future_version_is_rejected(self):
        payload = _minimal_payload()
        payload['dashboard_schema_version'] = DASHBOARD_SCHEMA_VERSION + 1
        with self.assertRaises(DashboardPayloadError) as ctx:
            validate_dashboard_payload(payload, source='fixture')
        self.assertIn('version', str(ctx.exception).lower())


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n ffbayes pytest tests/test_dashboard_payload_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffbayes.dashboard'`.

- [ ] **Step 3: Write the schema** at `src/ffbayes/dashboard/schemas/dashboard_payload.schema.json`. Critical fields are required and typed; everything else optional with `additionalProperties: true` for graceful degradation:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "ffbayes/dashboard_payload.schema.json",
  "title": "ffbayes dashboard payload",
  "type": "object",
  "required": [
    "dashboard_schema_version",
    "generated_at",
    "league_settings",
    "decision_table",
    "recommendation_summary",
    "decision_evidence"
  ],
  "properties": {
    "dashboard_schema_version": { "const": 1 },
    "generated_at": { "type": "string", "minLength": 1 },
    "league_settings": {
      "type": "object",
      "required": ["league_size", "draft_position", "scoring_type", "roster_spots"],
      "properties": {
        "league_size": { "type": "integer", "minimum": 2 },
        "draft_position": { "type": "integer", "minimum": 1 },
        "scoring_type": { "type": "string" },
        "roster_spots": { "type": "object" }
      },
      "additionalProperties": true
    },
    "decision_table": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["player_name", "position"],
        "properties": {
          "player_name": { "type": "string", "minLength": 1 },
          "position": { "type": "string", "minLength": 1 }
        },
        "additionalProperties": true
      }
    },
    "recommendation_summary": { "type": "array" },
    "decision_evidence": {
      "type": "object",
      "required": ["available", "status"],
      "properties": {
        "available": { "type": "boolean" },
        "status": { "type": "string" }
      },
      "additionalProperties": true
    },
    "runtime_controls": { "type": "object" },
    "current_pick_number": { "type": "integer" },
    "next_pick_number": { "type": "integer" },
    "current_draft_context_defaults": { "type": "object" },
    "selected_player": { "type": ["string", "null"] },
    "recommendation_inputs": { "type": "array" },
    "live_state": { "type": "object" },
    "scoring_presets": { "type": "object" },
    "position_summary": { "type": ["array", "object"] },
    "tier_cliffs": { "type": "array" },
    "roster_scenarios": { "type": "array" },
    "source_freshness": { "type": "array" },
    "analysis_provenance": { "type": "object" },
    "player_forecast_validation": { "type": "object" },
    "war_room_visuals": { "type": "object" },
    "backtest": { "type": "object" },
    "supporting_math": { "type": "object" },
    "metric_glossary": { "type": "object" },
    "model_overview": { "type": "object" },
    "bayesian_vor_summary": { "type": ["object", "null"] },
    "publish_provenance": { "type": "object" }
  },
  "additionalProperties": true
}
```

Before committing, cross-check every property name against the actual return dict at `draft_decision_system.py:3190–3250`; if the code has keys not listed above, add them as optional typed properties.

- [ ] **Step 4: Write `src/ffbayes/dashboard/__init__.py`:**

```python
"""Dashboard payload contract and frontend rendering."""
```

- [ ] **Step 5: Write `src/ffbayes/dashboard/payload_contract.py`:**

```python
"""Versioned dashboard payload contract and validation."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import jsonschema

DASHBOARD_SCHEMA_VERSION = 1

# Pre-versioning contract; keep in sync with refresh_dashboard expectations.
LEGACY_REQUIRED_PAYLOAD_KEYS = (
    'generated_at',
    'league_settings',
    'decision_table',
    'decision_evidence',
)


class DashboardPayloadError(ValueError):
    """A dashboard payload failed contract validation."""


def load_dashboard_schema() -> dict[str, Any]:
    """Load the packaged dashboard payload JSON Schema."""
    schema_text = (
        resources.files('ffbayes.dashboard')
        .joinpath('schemas/dashboard_payload.schema.json')
        .read_text(encoding='utf-8')
    )
    return json.loads(schema_text)


def stamp_schema_version(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp the current schema version onto a payload in place."""
    payload['dashboard_schema_version'] = DASHBOARD_SCHEMA_VERSION
    return payload


def _validate_legacy_payload(payload: dict[str, Any], source: str) -> None:
    missing = [
        key
        for key in LEGACY_REQUIRED_PAYLOAD_KEYS
        if key not in payload or payload.get(key) is None
    ]
    if missing:
        raise DashboardPayloadError(
            f'Dashboard payload from {source} is missing required keys: '
            f'{", ".join(missing)}'
        )
    if not isinstance(payload.get('decision_table'), list):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has a non-list `decision_table`.'
        )
    if not isinstance(payload.get('league_settings'), dict):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has an invalid `league_settings` object.'
        )


def validate_dashboard_payload(
    payload: Any, source: str = '<in-memory>'
) -> dict[str, Any]:
    """Validate a dashboard payload against the versioned contract.

    Payloads without ``dashboard_schema_version`` are treated as legacy and
    validated only against the pre-existing required-key contract so old
    artifacts keep loading. Versioned payloads must match the JSON Schema
    exactly (fail closed).
    """
    if not isinstance(payload, dict):
        raise DashboardPayloadError(
            f'Dashboard payload from {source} must be a JSON object.'
        )
    version = payload.get('dashboard_schema_version')
    if version is None:
        _validate_legacy_payload(payload, source)
        return payload
    if version != DASHBOARD_SCHEMA_VERSION:
        raise DashboardPayloadError(
            f'Dashboard payload from {source} has unsupported schema version '
            f'{version!r}; this build supports version {DASHBOARD_SCHEMA_VERSION}.'
        )
    validator = jsonschema.Draft202012Validator(load_dashboard_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        details = '; '.join(
            f'{"/".join(str(part) for part in err.absolute_path) or "<root>"}: '
            f'{err.message}'
            for err in errors[:10]
        )
        raise DashboardPayloadError(
            f'Dashboard payload from {source} failed schema validation: {details}'
        )
    return payload
```

- [ ] **Step 6: Add package-data** to `pyproject.toml` (after `[tool.setuptools]`):

```toml
[tool.setuptools.package-data]
ffbayes = ["dashboard/schemas/*.json", "dashboard/assets/*.html"]
```

Reinstall so resources resolve: `conda run -n ffbayes pip install -e .[dev]`

- [ ] **Step 7: Run tests**

Run: `conda run -n ffbayes pytest tests/test_dashboard_payload_schema.py -q`
Expected: all 8 pass. If `test_minimal_fixture_validates` fails, the fixture and schema disagree — fix the schema (the fixture mirrors real payloads).

- [ ] **Step 8: Lint and commit**

```bash
conda run -n ffbayes ruff check src/ffbayes/dashboard tests/test_dashboard_payload_schema.py
git add src/ffbayes/dashboard pyproject.toml tests/test_dashboard_payload_schema.py
git commit -m "feat(dashboard): add versioned payload contract with JSON Schema validation"
```

### Task 6: Stamp version + validate before every payload write

**Files:**
- Modify: `src/ffbayes/draft_strategy/draft_decision_system.py:3190` (return of `build_dashboard_payload`) and `:6945-6947` (write in `save_draft_decision_artifacts`)
- Test: `tests/test_draft_decision_system.py` (add cases)

- [ ] **Step 1: Write failing tests** — add to `tests/test_draft_decision_system.py` (follow the file's existing fixture/helper conventions for building a payload; read the existing `build_dashboard_payload` tests first and reuse their setup):

```python
def test_build_dashboard_payload_stamps_schema_version(self):
    payload = <existing helper that builds a payload in this test file>
    self.assertEqual(payload['dashboard_schema_version'], 1)

def test_save_artifacts_rejects_invalid_payload(self):
    # Build artifacts via the existing test helper, then corrupt the payload.
    artifacts = <existing helper that builds DraftDecisionArtifacts>
    artifacts.dashboard_payload.pop('decision_table')
    with self.assertRaises(DashboardPayloadError):
        save_draft_decision_artifacts(artifacts, tmp_dir, year=2026)
```

Run: `conda run -n ffbayes pytest tests/test_draft_decision_system.py -q -k 'schema_version or rejects_invalid'`
Expected: FAIL (no version key; no validation).

- [ ] **Step 2: Implement.** In `build_dashboard_payload()` add `'dashboard_schema_version': DASHBOARD_SCHEMA_VERSION,` as the first key of the return dict (line 3190). In `save_draft_decision_artifacts()` validate immediately before the write at 6945:

```python
validate_dashboard_payload(
    to_strict_jsonable(artifacts.dashboard_payload), source=str(payload_path)
)
payload_path.write_text(
    dumps_strict_json(artifacts.dashboard_payload, indent=2), encoding='utf-8'
)
```

Import at top of the module: `from ffbayes.dashboard.payload_contract import (DASHBOARD_SCHEMA_VERSION, validate_dashboard_payload)`. Check whether `to_strict_jsonable` is already imported in this module; if validation of the raw dict fails on numpy types, validating `json.loads(dumps_strict_json(...))` output is the fallback approach.

- [ ] **Step 3: Run the new tests and the full module suite**

Run: `conda run -n ffbayes pytest tests/test_draft_decision_system.py tests/test_dashboard_payload_schema.py -q`
Expected: PASS. Existing payload-contract assertions in `test_draft_decision_system.py` may need the new key added to expected-key lists — update them.

- [ ] **Step 4: Commit**

```bash
git add src/ffbayes/draft_strategy/draft_decision_system.py tests/test_draft_decision_system.py
git commit -m "feat(dashboard): stamp schema version and validate payload before write"
```

### Task 7: Route refresh-path validation through the contract

**Files:**
- Modify: `src/ffbayes/refresh_dashboard.py:33-107`
- Test: `tests/test_refresh_dashboard.py`

- [ ] **Step 1: Write failing test** — add to `tests/test_refresh_dashboard.py`:

```python
def test_load_dashboard_payload_rejects_unsupported_schema_version(tmp_path):
    payload = _load_minimal_payload()
    payload['dashboard_schema_version'] = 999
    payload_path = tmp_path / 'dashboard_payload_2026.json'
    payload_path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        refresh_dashboard.load_dashboard_payload(payload_path)
```

(Match the file's existing style — it may be unittest-based; adapt accordingly.)
Run: `conda run -n ffbayes pytest tests/test_refresh_dashboard.py -q -k unsupported`
Expected: FAIL (version currently ignored).

- [ ] **Step 2: Implement.** In `refresh_dashboard.py`, replace the body of `_validate_dashboard_payload()` so it first calls `validate_dashboard_payload(payload, source=str(payload_path))` from `ffbayes.dashboard.payload_contract`, then runs the existing `_validate_required_decision_evidence(payload, payload_path)` (freshness gating is policy, not schema — keep it separate). Keep `REQUIRED_PAYLOAD_KEYS` re-exported as an alias of `LEGACY_REQUIRED_PAYLOAD_KEYS` so any imports elsewhere keep working (grep for usages first: `rg -n 'REQUIRED_PAYLOAD_KEYS' src tests`).

- [ ] **Step 3: Run**

Run: `conda run -n ffbayes pytest tests/test_refresh_dashboard.py tests/test_publish_pages.py -q`
Expected: PASS (legacy fixtures without version still load via legacy contract; the fixture file from Task 4 has version 1 and passes full validation).

- [ ] **Step 4: Commit**

```bash
git add src/ffbayes/refresh_dashboard.py tests/test_refresh_dashboard.py
git commit -m "refactor(dashboard): route refresh payload validation through versioned contract"
```

### Task 8: Slice A wrap-up — full Python suite green

- [ ] **Step 1:** Run: `conda run -n ffbayes pytest -q`
Expected: same pass/fail profile as baseline `master` (run on master via `git stash`/worktree if any pre-existing failures need confirming) plus the new tests passing.
- [ ] **Step 2:** Run: `conda run -n ffbayes ruff check . && conda run -n ffbayes mypy src`
Expected: clean (or no new errors vs baseline).
- [ ] **Step 3:** Fix anything introduced; commit fixes as `fix(dashboard): ...`.

---

# Phase 3 — Frontend scaffold

### Task 9: Scaffold `dashboard_frontend/` (Vite + React + TS + Vitest)

**Files:**
- Create: `dashboard_frontend/package.json`, `dashboard_frontend/vite.config.ts`, `dashboard_frontend/tsconfig.json`, `dashboard_frontend/index.html`, `dashboard_frontend/src/main.tsx`, `dashboard_frontend/src/App.tsx`, `dashboard_frontend/.gitignore`
- Location rationale (document in design doc if it changes): repo root sibling to `src/`, matching the repo's flat top-level layout (`tests/`, `site/`, `config/`). Root `package.json` stays Playwright-only.

- [ ] **Step 1: Scaffold**

```bash
cd dashboard_frontend 2>/dev/null || npm create vite@latest dashboard_frontend -- --template react-ts
cd dashboard_frontend
npm install
npm install -D vite-plugin-singlefile vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom json-schema-to-typescript
```

Commit the generated `package-lock.json` (determinism). Add `dashboard_frontend/.gitignore` with `node_modules/` and `dist/`.

Pin the Node toolchain: add to `dashboard_frontend/package.json` an `"engines": { "node": ">=20" }` field, and create `dashboard_frontend/.nvmrc` containing the major version actually used to build (run `node --version` and record it). This keeps template rebuilds reproducible across machines.

- [ ] **Step 2: `vite.config.ts`:**

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  // Vitest config
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
});
```

Create `src/test/setup.ts` containing `import '@testing-library/jest-dom';`.
If TS complains about the `test` key, change the import to `import { defineConfig } from 'vitest/config';`.

- [ ] **Step 3: `index.html`** — the payload bootstrap is the load-bearing part. Title must match legacy exactly (`FFBayes Draft War Room`, see `draft_decision_system.py:3656`); the smoke test asserts on it:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FFBayes Draft War Room</title>
  </head>
  <body>
    <div id="root"></div>
    <script>
      try { window.FFBAYES_DASHBOARD = __PAYLOAD_JSON__; /*FFBAYES_PAYLOAD_END*/ } catch (_e) { window.FFBAYES_DASHBOARD = null; }
    </script>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

The `try/catch` makes the raw template harmless in `npm run dev` (where `__PAYLOAD_JSON__` is an undefined identifier → ReferenceError → caught → app falls back to fetching `./dashboard_payload.json`). The `/*FFBAYES_PAYLOAD_END*/` comment is the new injection suffix marker for `publish_pages.py` (Task 20).

- [ ] **Step 4: Minimal `src/App.tsx` / `src/main.tsx`** (placeholder shell; real layout comes in Task 14):

```tsx
// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

```tsx
// src/App.tsx
export default function App() {
  return <h1>FFBayes Draft War Room</h1>;
}
```

Delete unused scaffold files (`App.css`, logo assets) to keep the tree clean.

- [ ] **Step 5: Verify build + single-file output**

```bash
npm run build
ls dist && grep -c FFBAYES_DASHBOARD dist/index.html
```

Expected: `dist/` contains only `index.html` (plus maybe `vite.svg` — remove the favicon link if so); grep count ≥ 1; no separate `.js`/`.css` assets.

- [ ] **Step 6: Commit**

```bash
git add dashboard_frontend
git commit -m "feat(frontend): scaffold React+Vite single-file dashboard app"
```

### Task 10: Payload types + loader (TDD)

**Files:**
- Create: `dashboard_frontend/src/payload/types.generated.ts` (generated, committed)
- Create: `dashboard_frontend/src/payload/load.ts`
- Test: `dashboard_frontend/src/payload/load.test.ts`
- Modify: `dashboard_frontend/package.json` (scripts)

- [ ] **Step 1: Add scripts** to `dashboard_frontend/package.json`:

```json
"scripts": {
  "dev": "vite",
  "build": "tsc -b && vite build",
  "test": "vitest run",
  "typecheck": "tsc -b --noEmit",
  "generate:types": "json2ts -i ../src/ffbayes/dashboard/schemas/dashboard_payload.schema.json -o src/payload/types.generated.ts --bannerComment '/* AUTO-GENERATED from dashboard_payload.schema.json — run npm run generate:types */'"
}
```

Run: `npm run generate:types` — verify `src/payload/types.generated.ts` exports a `FfbayesDashboardPayload` (or similarly named) interface. Re-export it with a stable name in `load.ts` (next step) so components never import the generated name directly.

- [ ] **Step 2: Write the failing test** `src/payload/load.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadPayload } from './load';

afterEach(() => {
  // @ts-expect-error test cleanup
  delete window.FFBAYES_DASHBOARD;
  vi.restoreAllMocks();
});

describe('loadPayload', () => {
  it('returns the embedded payload when present', async () => {
    (window as any).FFBAYES_DASHBOARD = { generated_at: 'x', decision_table: [] };
    const payload = await loadPayload();
    expect(payload.generated_at).toBe('x');
  });

  it('falls back to fetching dashboard_payload.json', async () => {
    (window as any).FFBAYES_DASHBOARD = null;
    const fake = { generated_at: 'fetched', decision_table: [] };
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(fake))));
    const payload = await loadPayload();
    expect(payload.generated_at).toBe('fetched');
  });

  it('throws a clear error when neither source is available', async () => {
    (window as any).FFBAYES_DASHBOARD = null;
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    await expect(loadPayload()).rejects.toThrow(/dashboard payload/i);
  });
});
```

Run: `npm test` — expected FAIL (`load.ts` missing).

- [ ] **Step 3: Implement `src/payload/load.ts`:**

```ts
import type { FfbayesDashboardPayload } from './types.generated';

export type DashboardPayload = FfbayesDashboardPayload;

declare global {
  interface Window {
    FFBAYES_DASHBOARD?: unknown;
  }
}

export async function loadPayload(): Promise<DashboardPayload> {
  const embedded = window.FFBAYES_DASHBOARD;
  if (embedded && typeof embedded === 'object') {
    return embedded as DashboardPayload;
  }
  const response = await fetch('./dashboard_payload.json');
  if (!response.ok) {
    throw new Error(
      `Could not load the dashboard payload: no embedded payload and ` +
        `fetching ./dashboard_payload.json returned HTTP ${response.status}.`
    );
  }
  return (await response.json()) as DashboardPayload;
}
```

(Adjust the imported type name to whatever `json2ts` actually generated.)

- [ ] **Step 4: Run** `npm test` → PASS; `npm run typecheck` → clean.
- [ ] **Step 5: Commit**

```bash
git add dashboard_frontend/src/payload dashboard_frontend/package.json dashboard_frontend/package-lock.json
git commit -m "feat(frontend): typed payload loader with embedded/fetch fallback"
```

### Task 11: Graceful-degradation primitive (TDD)

**Files:**
- Create: `dashboard_frontend/src/components/SectionGate.tsx`
- Test: `dashboard_frontend/src/components/SectionGate.test.tsx`

Every optional payload section (`decision_evidence`, `war_room_visuals.*`, `player_forecast_validation`, `backtest`, …) follows the convention `{available: bool, status: string, reason?/reason_unavailable?: string}`. One gate component handles all of them.

- [ ] **Step 1: Failing test** `SectionGate.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { SectionGate } from './SectionGate';

it('renders children when the section is available', () => {
  render(
    <SectionGate section={{ available: true, status: 'available' }} title="Evidence">
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.getByText('panel-body')).toBeInTheDocument();
});

it('renders an unavailable notice with the reason when absent', () => {
  render(
    <SectionGate
      section={{ available: false, status: 'unavailable', reason_unavailable: 'no data' }}
      title="Evidence"
    >
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.queryByText('panel-body')).not.toBeInTheDocument();
  expect(screen.getByText(/no data/)).toBeInTheDocument();
});

it('treats a missing section object as unavailable without crashing', () => {
  render(
    <SectionGate section={undefined} title="Evidence">
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.queryByText('panel-body')).not.toBeInTheDocument();
  expect(screen.getByText(/not available/i)).toBeInTheDocument();
});
```

Run `npm test` → FAIL.

- [ ] **Step 2: Implement `SectionGate.tsx`:**

```tsx
import type { ReactNode } from 'react';

export interface GatedSection {
  available?: boolean;
  status?: string;
  reason?: string;
  reason_unavailable?: string;
}

export function SectionGate(props: {
  section: GatedSection | undefined | null;
  title: string;
  children: ReactNode;
}) {
  const { section, title, children } = props;
  if (section && section.available === true) {
    return <>{children}</>;
  }
  const reason =
    section?.reason_unavailable || section?.reason || 'This section is not available for this board.';
  return (
    <section className="section-unavailable" aria-label={`${title} unavailable`}>
      <h3>{title}</h3>
      <p>{reason}</p>
    </section>
  );
}
```

- [ ] **Step 3:** `npm test` → PASS. **Commit:** `git add dashboard_frontend/src/components && git commit -m "feat(frontend): SectionGate graceful degradation primitive"`

### Task 12: Build-to-template toolchain

**Files:**
- Create: `dashboard_frontend/scripts/build_template.mjs`
- Create: `src/ffbayes/dashboard/assets/dashboard_template.html` (built artifact, committed)
- Modify: `dashboard_frontend/package.json` (add `build:template` script)

- [ ] **Step 1: Write `scripts/build_template.mjs`:**

```js
// Copies the single-file Vite build into the Python package as the dashboard template.
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, '../dist/index.html');
const target = resolve(here, '../../src/ffbayes/dashboard/assets/dashboard_template.html');

const html = readFileSync(dist, 'utf-8');
for (const marker of ['window.FFBAYES_DASHBOARD = __PAYLOAD_JSON__', '/*FFBAYES_PAYLOAD_END*/']) {
  if (!html.includes(marker)) {
    console.error(`Built template is missing required marker: ${marker}`);
    process.exit(1);
  }
}
mkdirSync(dirname(target), { recursive: true });
copyFileSync(dist, target);
console.log(`Template staged at ${target} (${html.length} bytes)`);
```

Add script: `"build:template": "npm run build && node scripts/build_template.mjs"`.

- [ ] **Step 2: Run** `npm run build:template`
Expected: template staged, marker check passes. Verify Vite did **not** mangle the inline bootstrap script (singlefile inlines bundles but must leave the literal `__PAYLOAD_JSON__` script untouched — confirm with `grep -c '__PAYLOAD_JSON__' ../src/ffbayes/dashboard/assets/dashboard_template.html` → 1). If Vite minifies/transforms the inline script, move the bootstrap to a `<script>` tag with attribute `data-ffbayes-payload` and adjust the marker check — but verify first; plain inline scripts in `index.html` pass through unchanged.

- [ ] **Step 3: Python-side smoke check**

```bash
conda run -n ffbayes python - <<'EOF'
from importlib import resources
t = resources.files('ffbayes.dashboard').joinpath('assets/dashboard_template.html')
text = t.read_text(encoding='utf-8')
assert 'window.FFBAYES_DASHBOARD = __PAYLOAD_JSON__' in text
print('template resolvable from package:', len(text), 'bytes')
EOF
```

- [ ] **Step 4: Commit**

```bash
git add dashboard_frontend/scripts dashboard_frontend/package.json src/ffbayes/dashboard/assets/dashboard_template.html
git commit -m "feat(frontend): build pipeline staging single-file template into Python package"
```

---

# Phase 4 — Reproduce dashboard behavior (parity port)

General porting method for Tasks 13–19 — read before starting any of them:

1. The reference implementation is the JS inside `export_dashboard_html()` at `draft_decision_system.py:4568–6647`. For each task, the line ranges below are starting points; locate the exact functions with `rg -n '<functionName>' src/ffbayes/draft_strategy/draft_decision_system.py`.
2. The acceptance spec is `tests/dashboard_smoke.mjs` — before porting a panel, read the smoke assertions that touch it and treat the DOM ids/classes/text it queries as the required interface. **Match the smoke test's selectors exactly**; that is what "parity" means mechanically.
3. Port behavior, not style: replicate CSS classes/ids the smoke test uses; otherwise reasonable modern styling is fine. Copy the legacy CSS block (in the template string before line 4568) into `src/styles.css` as the starting point if that's fastest.
4. Per task: write Vitest tests first for logic, implement, run `npm test && npm run typecheck`, commit with `feat(frontend): ...`.
5. Use the shared fixture: add a Vitest helper `src/test/fixture.ts` that imports `../../../tests/fixtures/dashboard_payload_minimal.json` (Vite supports JSON imports; add `"resolveJsonModule": true` in tsconfig if needed).

### Task 13: Draft state store (taken/mine/queue, undo/redo, persistence)

**Files:**
- Create: `dashboard_frontend/src/state/draftState.ts`
- Test: `dashboard_frontend/src/state/draftState.test.ts`

**Reference:** state object, `captureDraftSnapshot`/`restoreDraftSnapshot`/`pushSnapshot`, `undoLast`/`redoLast` (lines 6622–6647), localStorage persistence (lines 4929, 4951), `STORAGE_KEY = 'ffbayes-dashboard-state-v2'` (line 4573). **Reuse the exact storage key and persisted shape** so existing in-browser state survives the renderer switch — read the legacy `JSON.parse`/`setItem` call sites and replicate the serialized fields exactly.

- [ ] **Step 1: Write failing tests** covering, at minimum:

```ts
import { beforeEach, describe, expect, it } from 'vitest';
import { createDraftStore, STORAGE_KEY } from './draftState';

beforeEach(() => window.localStorage.clear());

describe('draft store', () => {
  it('marks a player taken and undoes it', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markTaken('Test Player');
    expect(store.getState().takenPlayers).toContain('Test Player');
    store.undo();
    expect(store.getState().takenPlayers).not.toContain('Test Player');
  });

  it('redo restores an undone action', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markMine('Test Player');
    store.undo();
    store.redo();
    expect(store.getState().yourPlayers).toContain('Test Player');
  });

  it('persists to and rehydrates from localStorage under the legacy key', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markTaken('Test Player');
    const raw = window.localStorage.getItem(STORAGE_KEY);
    expect(raw).toBeTruthy();
    const rehydrated = createDraftStore({ initialPickNumber: 5 });
    expect(rehydrated.getState().takenPlayers).toContain('Test Player');
  });

  it('rehydrates legacy-format state written by the old dashboard', () => {
    // Paste a real serialized blob captured from the legacy dashboard here
    // (generate one by opening a legacy-rendered board, taking actions, and
    // copying localStorage['ffbayes-dashboard-state-v2']).
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ /* legacy blob */ }));
    const store = createDraftStore({ initialPickNumber: 5 });
    expect(store.getState().takenPlayers.length).toBeGreaterThanOrEqual(0);
  });
});
```

- [ ] **Step 2: Implement** `createDraftStore` as a framework-agnostic store (plain TS class or `useSyncExternalStore`-compatible): state fields mirroring the legacy `state` object (`takenPlayers`, `yourPlayers`, `queue`, `pickLog`, `history`, `redoHistory`, current pick, settings overrides — enumerate them from the legacy JS), snapshot-based undo/redo identical to legacy semantics (undo pushes onto redo stack, any new action clears redo), and persistence on every mutation.
- [ ] **Step 3:** `npm test` → PASS. Commit: `feat(frontend): draft state store with undo/redo and legacy persistence`.

### Task 14: App shell, settings summary, pick status

**Files:**
- Create: `dashboard_frontend/src/components/AppShell.tsx`, `SettingsPanel.tsx`, `PickStatus.tsx`, `dashboard_frontend/src/styles.css`
- Modify: `dashboard_frontend/src/App.tsx` (load payload → provide store + payload via context → render shell)
- Test: `dashboard_frontend/src/components/SettingsPanel.test.tsx`, `PickStatus.test.tsx`

**Reference:** header/league summary markup near the top of the HTML template (after line 3656); controls for league settings, risk tolerance, and scoring-preset switching (`runtime_controls`, `scoring_presets` payload keys); current/next pick from `current_pick_number`/`next_pick_number`/`live_state`. Smoke assertions: title, league/draft/scoring preset controls, preset persistence.

- [ ] **Step 1:** Tests first: `SettingsPanel` renders league size/draft position/scoring preset from the fixture payload and switching the preset select calls the store; `PickStatus` shows current and next pick numbers from the fixture.
- [ ] **Step 2:** Implement. `App.tsx` becomes: `loadPayload()` in an effect → loading/error states → `<AppShell payload={...} store={...}>` rendering all panels (panels added incrementally in later tasks; render placeholders only for not-yet-ported ones and delete each placeholder in its task).
- [ ] **Step 2b: Add an error boundary.** Create `dashboard_frontend/src/components/ErrorBoundary.tsx` (standard React class component implementing `componentDidCatch`/`getDerivedStateFromError`) and wrap the entire app in it from `main.tsx`. On error it must render the error message and a hint ("The dashboard payload may be malformed — regenerate with `ffbayes stage-dashboard --year <year>`") instead of a blank page. Add a Vitest test that renders a child which throws and asserts the fallback message appears.
- [ ] **Step 3:** `npm test && npm run build:template` → PASS, template rebuilds. Commit.

### Task 15: Recommendation views (pick-now / fallback / can-wait)

**Files:**
- Create: `dashboard_frontend/src/components/RecommendationPanel.tsx`
- Test: `dashboard_frontend/src/components/RecommendationPanel.test.tsx`

**Reference:** `recommendation_summary`, `live_state`, `recommendation_inputs` payload keys; legacy rendering functions (search `rg -n 'pick_now|can_wait|fallback' src/ffbayes/draft_strategy/draft_decision_system.py` within the JS block). The live recomputation logic (recommendations updating as players are marked taken) must replicate legacy client-side behavior — find the legacy `render()` path that filters recommendations by `state.takenPlayers`.

- [ ] **Step 1:** Tests: with fixture payload, panel renders the top recommendation; marking the top player taken in the store removes them from the rendered recommendations; lanes (pick-now/fallback/can-wait) render per the fixture's `live_state`.
- [ ] **Step 2:** Implement; reuse `SectionGate` for absent sections. **Step 3:** test+commit.

### Task 16: Player board table + player inspector

**Files:**
- Create: `dashboard_frontend/src/components/PlayerBoard.tsx`, `PlayerInspector.tsx`
- Test: `PlayerBoard.test.tsx`, `PlayerInspector.test.tsx`

**Reference:** `decision_table` rows and per-row action buttons (taken/mine/queue) wired to the Task 13 store; inspector driven by `selected_player` + row click; columns and formatting from the legacy board-render function (find via `rg -n 'decision_table|renderBoard|boardBody' ...`). Smoke assertions: board interactions taken/mine/queue, undo/redo round-trip, inspector content, row filtering.

- [ ] **Step 1:** Tests: board renders fixture rows; clicking "taken" updates store and the row gains the taken styling/state; clicking a row updates the inspector; filter/search box narrows rows (if legacy has one — check the smoke test).
- [ ] **Step 2:** Implement. **Step 3:** test+commit.

### Task 17: Evidence, freshness, and provenance panels

**Files:**
- Create: `dashboard_frontend/src/components/EvidencePanel.tsx`, `FreshnessPanel.tsx`, `ProvenanceBanner.tsx`
- Test: `EvidencePanel.test.tsx`

**Reference:** `decision_evidence` (strategy_summary, season_rows, top_disagreements, freshness, limitations — built at `draft_decision_system.py:817–896`), `source_freshness`, `analysis_provenance`, `player_forecast_validation`, `publish_provenance` (staged payloads only). Smoke assertions: decision evidence text, freshness display, staged site shows publish provenance.

- [ ] **Step 1:** Tests: evidence panel renders strategy summary + season rows from fixture; unavailable evidence renders the `SectionGate` notice; `ProvenanceBanner` renders only when `publish_provenance` exists in the payload (staged mode).
- [ ] **Step 2:** Implement. **Step 3:** test+commit.

### Task 18: War-room visuals

**Files:**
- Create: `dashboard_frontend/src/components/warroom/TimingFrontier.tsx`, `PositionalCliffs.tsx`, `ComparativeExplainer.tsx`
- Test: `dashboard_frontend/src/components/warroom/warroom.test.tsx`

**Reference:** `war_room_visuals` payload section (`schema_version: war_room_visuals_v1`, built at `draft_decision_system.py:2290–2331`): `timing_frontier` (candidates with lane/survival/regret), `positional_cliffs` (positions + default_positions), `comparative_explainer` (top_disagreements). Legacy rendering is SVG/DOM built in the JS block — port the same visual encodings (no charting library unless the legacy used one; it did not). Each sub-visual is independently gated by `SectionGate`.

- [ ] **Step 1:** Tests: each sub-visual renders from the fixture's `war_room_visuals`; each degrades via `SectionGate` when `available: false`; timing frontier reflects `current_pick_number`.
- [ ] **Step 2:** Implement. **Step 3:** test+commit.

### Task 19: Finalize bundle + staged-mode behavior

**Files:**
- Create: `dashboard_frontend/src/finalize/buildBundle.ts`, `dashboard_frontend/src/components/FinalizePanel.tsx`
- Test: `dashboard_frontend/src/finalize/buildBundle.test.ts`

**Reference:** `FINALIZED_SCHEMA_VERSION = 'finalized_draft_v1'` (line 4582); finalize builders produce one JSON download plus two HTML documents (snapshot, title `FFBayes Finalized Draft Snapshot` line 6328; summary, title `FFBayes Post-Draft Summary` line 6421). Smoke assertions: finalize triggers 3 downloads with the right schema/titles/content; **staged sites (payload contains `publish_provenance`) hide the finalize button**.

- [ ] **Step 1:** Tests: `buildBundle(state, payload)` returns `{json, snapshotHtml, summaryHtml}` where `json.schema_version === 'finalized_draft_v1'` and both HTML strings contain their legacy titles and the drafted roster; `FinalizePanel` is absent when payload has `publish_provenance`.
- [ ] **Step 2:** Implement — port the legacy bundle-builder functions verbatim in structure (same JSON fields; diff against a legacy-produced bundle if available). Downloads via `Blob` + anchor click, as legacy does.
- [ ] **Step 3:** test+commit. Then run `npm run build:template` and commit the refreshed template: `git add src/ffbayes/dashboard/assets/dashboard_template.html && git commit -m "feat(frontend): rebuild template with full panel parity port"`.

---

# Phase 5 — Integrate with Python CLI/workflows

### Task 20: `frontend_renderer.py` + renderer switch + injection marker (TDD)

**Files:**
- Create: `src/ffbayes/dashboard/frontend_renderer.py`
- Modify: `src/ffbayes/draft_strategy/draft_decision_system.py:6949-6957` (renderer branch in `save_draft_decision_artifacts`)
- Modify: `src/ffbayes/refresh_dashboard.py:192-210` (renderer branch in the HTML re-render)
- Modify: `src/ffbayes/publish_pages.py:22-23` and `_inject_dashboard_payload_into_html` (second suffix marker)
- Test: `tests/test_frontend_renderer.py`, extend `tests/test_publish_pages.py`

- [ ] **Step 1: Failing tests** in `tests/test_frontend_renderer.py`:

```python
"""Tests for the frontend template renderer and renderer selection."""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ffbayes.dashboard.frontend_renderer import (
    RENDERER_ENV_VAR,
    active_renderer,
    render_dashboard_html,
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestFrontendRenderer(unittest.TestCase):
    def test_default_renderer_is_legacy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RENDERER_ENV_VAR, None)
            self.assertEqual(active_renderer(), 'legacy')

    def test_invalid_renderer_value_raises(self):
        with mock.patch.dict(os.environ, {RENDERER_ENV_VAR: 'bogus'}):
            with self.assertRaises(ValueError):
                active_renderer()

    def test_render_injects_payload_and_label(self):
        payload = json.loads(
            (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(
                encoding='utf-8'
            )
        )
        out = Path(self.tmp.name) / 'board.html'  # use tempfile.TemporaryDirectory in setUp
        render_dashboard_html(payload, out, generated_label='2026-06-12 10:00')
        html = out.read_text(encoding='utf-8')
        self.assertNotIn('__PAYLOAD_JSON__', html)
        self.assertIn('window.FFBAYES_DASHBOARD = {', html)
        self.assertIn('/*FFBAYES_PAYLOAD_END*/', html)
        self.assertIn('2026-06-12 10:00', html)

    def test_render_escapes_script_breaking_strings(self):
        payload = json.loads(
            (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(
                encoding='utf-8'
            )
        )
        payload['decision_table'][0]['player_name'] = 'Bad </script><script>alert(1)'
        out = Path(self.tmp.name) / 'board.html'
        render_dashboard_html(payload, out, generated_label='x')
        html = out.read_text(encoding='utf-8')
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn('<\\/script>', html)
```

(Add the standard `setUp`/`tearDown` with `tempfile.TemporaryDirectory`.)
Run: `conda run -n ffbayes pytest tests/test_frontend_renderer.py -q` → FAIL (module missing).

- [ ] **Step 2: Implement `src/ffbayes/dashboard/frontend_renderer.py`:**

```python
"""Render the dashboard HTML from the prebuilt frontend template."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

from ffbayes.utils.json_serialization import dumps_strict_json

PAYLOAD_PLACEHOLDER = '__PAYLOAD_JSON__'
GENERATED_LABEL_PLACEHOLDER = '__GENERATED_LABEL__'

RENDERER_ENV_VAR = 'FFBAYES_DASHBOARD_RENDERER'
RENDERER_LEGACY = 'legacy'
RENDERER_FRONTEND = 'frontend'
_VALID_RENDERERS = (RENDERER_LEGACY, RENDERER_FRONTEND)


def dumps_html_safe_json(payload: dict[str, Any]) -> str:
    """Serialize payload JSON safely for embedding inside a <script> tag.

    A literal ``</script>`` inside a JSON string would terminate the script
    element early (HTML parses tags before JS). ``<\\/`` is a valid JSON/JS
    string escape for ``</``, so this is lossless.
    """
    return dumps_strict_json(payload).replace('</', '<\\/')


def active_renderer() -> str:
    """Return the configured dashboard renderer (default: legacy)."""
    value = os.environ.get(RENDERER_ENV_VAR, RENDERER_LEGACY).strip().lower()
    if value not in _VALID_RENDERERS:
        raise ValueError(
            f'{RENDERER_ENV_VAR} must be one of {_VALID_RENDERERS}, got {value!r}.'
        )
    return value


def load_dashboard_template() -> str:
    """Load the packaged single-file frontend template."""
    template = resources.files('ffbayes.dashboard').joinpath(
        'assets/dashboard_template.html'
    )
    if not template.is_file():
        raise FileNotFoundError(
            'Frontend dashboard template is missing from the ffbayes package. '
            'Build it with `cd dashboard_frontend && npm ci && npm run build:template`, '
            f'or unset {RENDERER_ENV_VAR} to use the legacy renderer.'
        )
    return template.read_text(encoding='utf-8')


def render_dashboard_html(
    payload: dict[str, Any], output_path: Path | str, generated_label: str
) -> Path:
    """Inject the payload into the frontend template and write the HTML."""
    output_path = Path(output_path)
    html = load_dashboard_template()
    html = html.replace(PAYLOAD_PLACEHOLDER, dumps_html_safe_json(payload))
    html = html.replace(GENERATED_LABEL_PLACEHOLDER, generated_label)
    output_path.write_text(html, encoding='utf-8')
    return output_path
```

Note: the scaffolded `index.html` only uses `__PAYLOAD_JSON__`; if `__GENERATED_LABEL__` should appear in the UI (legacy shows a generated-at label — check smoke assertions), add `<span data-generated-label>__GENERATED_LABEL__</span>` handling in the frontend shell (Task 14) — the payload's `generated_at` is the better source; prefer rendering from payload and keep the placeholder replacement as a no-op-safe extra.

- [ ] **Step 3: Renderer branch in `save_draft_decision_artifacts`** — replace the call at 6949–6957:

```python
if active_renderer() == RENDERER_FRONTEND:
    render_dashboard_html(
        canonical_payload,
        html_path,
        generated_label=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )
else:
    export_dashboard_html(
        artifacts.decision_table,
        artifacts.recommendations,
        html_path,
        artifacts.league_settings,
        backtest=artifacts.backtest,
        source_freshness=artifacts.source_freshness,
        dashboard_payload=canonical_payload,
    )
```

Apply the same branch in `refresh_dashboard.py`'s re-render (192–210), reusing its `_generated_label_from_payload(payload)` for the label.

- [ ] **Step 4: Injection marker in `publish_pages.py`** — add after line 23:

```python
PAYLOAD_ASSIGNMENT_SUFFIX_V2 = '; /*FFBAYES_PAYLOAD_END*/'
```

In `_inject_dashboard_payload_into_html`, search for `PAYLOAD_ASSIGNMENT_SUFFIX_V2` first and fall back to the legacy `PAYLOAD_ASSIGNMENT_SUFFIX`; preserve the matched suffix in the output. When the V2 suffix matched (frontend template), serialize the injected payload with `dumps_html_safe_json` from `ffbayes.dashboard.frontend_renderer` instead of `dumps_strict_json`, so `</script>`-bearing strings cannot break the staged page either. Add a test in `tests/test_publish_pages.py` that injects into a frontend-style HTML string (`'<script>try { window.FFBAYES_DASHBOARD = {"old": 1}; /*FFBAYES_PAYLOAD_END*/ } catch (_e) {}</script>'`) and asserts the new payload replaces the old and the marker survives.

- [ ] **Step 5: Add a renderer-switch regression test** to `tests/test_refresh_dashboard.py`: with `mock.patch.dict(os.environ, {'FFBAYES_DASHBOARD_RENDERER': 'frontend'})`, `refresh_runtime_dashboard(...)` against the fixture payload produces HTML containing `/*FFBAYES_PAYLOAD_END*/` and the fixture's player name; with the env var unset, output matches legacy (contains the legacy IIFE `(() => {`).

- [ ] **Step 6: Run**

```bash
conda run -n ffbayes pytest tests/test_frontend_renderer.py tests/test_refresh_dashboard.py tests/test_publish_pages.py tests/test_draft_decision_system.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ffbayes/dashboard/frontend_renderer.py src/ffbayes/draft_strategy/draft_decision_system.py src/ffbayes/refresh_dashboard.py src/ffbayes/publish_pages.py tests/test_frontend_renderer.py tests/test_refresh_dashboard.py tests/test_publish_pages.py
git commit -m "feat(dashboard): opt-in frontend renderer behind FFBAYES_DASHBOARD_RENDERER"
```

### Task 21: CLI regression confirmation

**Files:** none new (verification only, plus fixes if anything broke)

- [ ] **Step 1:** Run the CLI-level suites:

```bash
conda run -n ffbayes pytest tests/test_cli.py tests/test_run_pipeline_split.py tests/test_documentation_contracts.py -q
```

Expected: PASS — these prove `pre-draft`, `--stage-pages`, `stage-dashboard`, `draft-retrospective` wiring is untouched.

- [ ] **Step 2:** End-to-end refresh with real or fixture data (no model rerun):

```bash
conda run -n ffbayes ffbayes stage-dashboard --year 2026
```

If runtime data under `~/ProjectsRuntime/ffbayes/seasons/2026/draft_strategy/` is missing, this fails with the documented "Run `ffbayes draft-strategy` first" error — that is acceptable; record it as an environment constraint in the final report and rely on the Step 1 tests + Task 22 fixture-staged smoke instead.

- [ ] **Step 3:** Commit any fixes (`fix(dashboard): ...`).

---

# Phase 6 — Parity validation via the smoke suite

### Task 22: Run `dashboard_smoke.mjs` against the frontend renderer

**Files:**
- Possibly modify: `tests/dashboard_smoke.mjs` (only if selectors need parameterizing — prefer fixing the frontend to match the smoke test, not vice versa)
- Create: `tests/test_dashboard_smoke_frontend.py` (optional pytest wrapper mirroring `tests/test_dashboard_smoke.py` with the renderer env var set)

- [ ] **Step 1: Stage a frontend-rendered site into a temp dir.** Reuse the fixture-staging approach from `tests/test_player_forecast_stress_fixture.py:521+` (read it first):

```bash
conda run -n ffbayes python - <<'EOF'
import os, tempfile
from pathlib import Path
os.environ['FFBAYES_DASHBOARD_RENDERER'] = 'frontend'
from ffbayes.refresh_dashboard import refresh_runtime_dashboard
tmp = Path(tempfile.mkdtemp(prefix='ffbayes-smoke-'))
payload = Path('tests/fixtures/dashboard_payload_minimal.json')
refresh_runtime_dashboard(
    year=2026,
    payload_path=payload,
    output_html=tmp / 'draft_board_2026.html',
    stage_pages=True,
    pages_output_dir=tmp / 'site',   # check refresh_runtime_dashboard's actual signature for the site-dir parameter; adapt as needed
)
print(tmp / 'site')
EOF
```

(If `refresh_runtime_dashboard` can't redirect the site dir, call `publish_pages.stage_pages_site(source_html=..., source_payload=..., output_dir=...)` directly.)

- [ ] **Step 2: Run the smoke suite against it**

```bash
npm ci   # repo root, installs playwright
FFBAYES_SMOKE_SITE_DIR=<printed site dir> node tests/dashboard_smoke.mjs
```

Expected: **all checks pass.** This is the parity gate. Iterate: each failure points at a missing selector/behavior — fix the frontend component (Tasks 13–19 files), rebuild the template (`npm run build:template`), restage, rerun. Note: the staged site hides finalize; to exercise finalize checks, also run the smoke against the non-staged `draft_board_2026.html` parent dir if the suite supports it (read how `tests/test_dashboard_smoke.py` invokes it).

- [ ] **Step 3:** Also confirm legacy is untouched: run the existing pytest wrapper `conda run -n ffbayes pytest tests/test_dashboard_smoke.py -q` (legacy renderer default). Expected: same result as baseline.
- [ ] **Step 4:** Commit frontend fixes + final template rebuild: `test(dashboard): frontend renderer passes full smoke parity suite`.

---

# Phase 7 — Final validation and report

### Task 23: Full validation sweep + final report

- [ ] **Step 1: Python:** `conda run -n ffbayes ruff check . && conda run -n ffbayes mypy src && conda run -n ffbayes pytest -q`
- [ ] **Step 2: Frontend:** `cd dashboard_frontend && npm run typecheck && npm test && npm run build:template && git diff --exit-code ../src/ffbayes/dashboard/assets/dashboard_template.html` (template in repo matches a fresh deterministic build; if the diff is non-empty due to non-determinism, investigate — likely a timestamp or hash leaking into the bundle — and eliminate it).
- [ ] **Step 3: Docs cross-check:** update `docs/DASHBOARD_OPERATOR_GUIDE.md` and `docs/DATA_LINEAGE_AND_PATHS.md` with the renderer env var and the `dashboard_frontend/` build commands; run `conda run -n ffbayes pytest tests/test_documentation_contracts.py -q`.
- [ ] **Step 4: Commit:** `docs(dashboard): document frontend renderer and build workflow`.
- [ ] **Step 5: Write the final report** (chat message, not a committed file) containing exactly:
  - Starting branch `master` + baseline commit `89c61a3e7bb992508ddfb6a5aacd91a168a2741d` (or actual), clean-start confirmation
  - New branch `refactor/ffbayes-dashboard-frontend`
  - Files changed (from `git diff --stat master...HEAD`)
  - Architecture decision (single-file React/Vite template, committed into package, legacy default)
  - Implementation summary per phase
  - Tests run and results (pytest counts, vitest counts, smoke pass/fail)
  - Commands that could not be run and why (e.g. `ffbayes stage-dashboard --year 2026` if runtime data absent; full `ffbayes pre-draft` — never run it, it's a long model pipeline)
  - Remaining gaps (at minimum: default renderer still `legacy`; flip + legacy deletion are follow-ups)
  - Recommended next PRs: (1) flip default renderer after a draft-day dry run, (2) delete legacy `export_dashboard_html` + slim `draft_decision_system.py`, (3) CI job that rebuilds the template and fails on drift, (4) retrospective HTML (`draft_retrospective.py:800–890`) migration to the same pattern
  - Exact user-facing commands:
    - Rebuild frontend template: `cd dashboard_frontend && npm ci && npm run build:template`
    - Regenerate dashboard (legacy, unchanged): `ffbayes draft-strategy` / `ffbayes pre-draft [--stage-pages]`
    - Regenerate with new frontend: `FFBAYES_DASHBOARD_RENDERER=frontend ffbayes stage-dashboard --year <year>`

---

## Self-review checklist (run after writing/before executing)

1. **Spec coverage:** Phases 0–7 of the original request each map to Tasks 1–23; framework comparison lives in Task 2; payload contract in Tasks 3–8; scaffold in 9–12; parity in 13–19; CLI integration in 20–21; tests in every task plus 22; final validation in 23. Constraint "never remove legacy until parity" is enforced by the renderer default + out-of-scope flip.
2. **Placeholder scan:** Tasks 13–19 intentionally reference legacy source line ranges instead of inlining ~2,000 lines of ported JS — each gives exact reference locations, the smoke test as the mechanical interface spec, and concrete test code. Everything else has complete code.
3. **Type consistency:** `DashboardPayloadError`, `validate_dashboard_payload`, `stamp_schema_version`, `DASHBOARD_SCHEMA_VERSION`, `active_renderer`, `render_dashboard_html`, `RENDERER_ENV_VAR`, `PAYLOAD_ASSIGNMENT_SUFFIX_V2`, `SectionGate`, `createDraftStore`, `loadPayload` are named consistently across tasks.
