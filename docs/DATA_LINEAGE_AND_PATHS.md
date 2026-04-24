# Data Lineage And Paths

Audience: operators and contributors who need path authority and artifact
lineage.

Scope: runtime roots, repo-local shortcuts, staged Pages files, optional publish
targets, and retrospective artifacts.

Trust boundary: runtime artifacts are authoritative. Repo `dashboard/`, repo
`site/`, and cloud mirrors are derived surfaces with different purposes.

## What This Is

This guide answers two questions: where did this artifact come from, and which
copy should be trusted?

## When To Use It

Use this guide when:

- you need to find a generated artifact
- you want to know whether a path is authoritative or derived
- you need to understand what `pre-draft --stage-pages`, `stage-dashboard`, or `publish` changes
- you are auditing lineage from raw data to final board

## What To Inspect

Primary path policy lives in:

- `src/ffbayes/utils/path_constants.py`

Primary commands that create or move artifacts:

- `ffbayes pre-draft`
- `ffbayes draft-strategy`
- `ffbayes pre-draft --stage-pages`
- `ffbayes stage-dashboard`
- `ffbayes draft-retrospective`
- `ffbayes publish`

Command mutation summary:

| Command | Main mutation | Authority effect |
| --- | --- | --- |
| `ffbayes pre-draft` | rebuilds supported runtime artifacts | updates authoritative runtime outputs |
| `ffbayes pre-draft --stage-pages` | rebuilds runtime artifacts and stages Pages | updates authoritative runtime outputs, then derived publish surface |
| `ffbayes stage-dashboard` | stages dashboard HTML, payload, and provenance | updates derived publish surface from runtime artifacts |
| `ffbayes draft-retrospective` | imports finalized draft artifacts | updates authoritative runtime retrospective inputs |
| `ffbayes publish` | stages Pages and mirrors selected outputs | updates derived publish and cloud mirror surfaces |

## Interpretation Boundaries

- Do not assume a repo path is authoritative just because it is easy to open.
- Do not assume cloud-mirrored data is the working source of truth.
- Do not assume `site/` updates itself without an explicit staging command.

## Path Categories

| Category | Example path | Purpose | Authority |
| --- | --- | --- | --- |
| runtime root | `<runtime-root>` | main local working tree | authoritative base |
| raw runtime data | `inputs/raw/season_datasets/` under runtime root | collected season-level inputs | authoritative runtime input |
| processed runtime data | `inputs/processed/` under runtime root | cleaned and derived inputs | authoritative runtime input |
| unified dataset | `inputs/processed/unified_dataset/unified_dataset.csv` under runtime root | analysis-ready player dataset | authoritative runtime input |
| canonical pre-draft artifacts | `seasons/<year>/` under runtime root | main analysis outputs | authoritative runtime output |
| runtime dashboard shortcut | `<runtime-root>/dashboard/index.html` | local shortcut beside runtime artifacts | derived local shortcut |
| repo dashboard shortcut | `dashboard/index.html` | easy local opening path | derived local shortcut |
| staged Pages copy | `site/index.html` | repo-tracked publishing surface | derived publish surface |
| cloud mirror | `data/` under cloud root | mirrored selected runtime artifacts | derived mirror |
| cloud snapshot | `Analysis/<date>/` under cloud root | dated publish snapshot | derived mirror |

## Lineage From Source Data To Final Board

### 1. Raw Collection

Purpose: collect source season data into the runtime raw-data tree.

Typical area:

```text
<runtime-root>/inputs/raw/season_datasets/
```

### 2. Validation

Purpose: confirm required seasons and expected inputs exist.

Trust surface:

- freshness manifests and warnings later exposed in `analysis_provenance` and `decision_evidence`

### 3. Preprocessing

Purpose: build analysis-ready datasets from collected raw inputs.

Typical area:

```text
<runtime-root>/inputs/processed/
```

### 4. Unified Dataset Construction

Purpose: produce the player-level dataset used by the board and retrospective.

Canonical path:

```text
<runtime-root>/inputs/processed/unified_dataset/unified_dataset.csv
```

### 5. Pre-Draft Artifact Generation

Purpose: create the board workbook, payload, HTML, and required decision backtest.

Canonical area:

```text
<runtime-root>/seasons/<year>/draft_strategy/
```

Typical files:

- `draft_board_<year>.xlsx`
- `dashboard_payload_<year>.json`
- `draft_board_<year>.html`
- `draft_decision_backtest_<year_range>.json`
- `model_outputs/player_forecast/player_forecast_<year>.json`
- `model_outputs/player_forecast/player_forecast_diagnostics_<year>.json`
- `model_outputs/player_forecast/player_forecast_validation_<year_range>.json`
- `../diagnostics/validation/player_forecast_validation_summary_<year_range>.json`

Production dashboard generation requires the decision backtest evidence to be
available and fresh. If that evidence is missing, degraded, or unavailable, the
dashboard build fails instead of publishing a weakened production surface.

### 6. Repo-Local Shortcut Staging

Purpose: make the current board easy to open from the repo root.

Derived paths:

- `<runtime-root>/dashboard/index.html`
- `<runtime-root>/dashboard/dashboard_payload.json`
- `dashboard/index.html`
- `dashboard/dashboard_payload.json`

These are derived convenience copies, not the source of truth.

### 7. GitHub Pages Staging

Purpose: copy the current dashboard into repo-tracked `site/`.

Full pre-draft refresh plus Pages staging:

```bash
ffbayes pre-draft --stage-pages
```

Dashboard-only refresh and Pages staging:

```bash
ffbayes stage-dashboard --year 2026
```

Derived paths:

- `site/index.html`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`

### 8. Finalized Draft Import And Retrospective

Purpose: move browser-downloaded finalized artifacts into the canonical runtime folder, then evaluate them later against realized outcomes.

Canonical finalized import path:

```text
<runtime-root>/seasons/<year>/draft_strategy/finalized_drafts/
```

Files named like `*_test.*` are test artifacts. They are not imported,
autodiscovered, or used as production retrospective inputs.

Canonical retrospective outputs:

- `draft_retrospective_<year>.json`
- `draft_retrospective_<year>.html`

## Environment Overrides

Use overrides only when you intentionally want a non-default layout:

- `FFBAYES_RUNTIME_ROOT`
- `FFBAYES_PROJECT_ROOT`
- `FFBAYES_CLOUD_ROOT`

These should be set before running the CLI so collection, preprocessing, dashboard staging, and retrospective import all agree on the same path base.

## Minimal Provenance Example

```json
{
  "analysis_provenance": {
    "overall_freshness": {
      "status": "fresh",
      "override_used": false
    }
  },
  "publish_provenance": {
    "schema_version": "publish_provenance_v1",
    "source_html": "draft_board_2026.html",
    "source_payload": "dashboard_payload_2026.json"
  }
}
```

What to notice:

- runtime provenance tells you about freshness and source inputs
- publish provenance tells you about the staged Pages copy

## Commands And Paths

Purpose: pair common commands with the surface they mutate.

| Command | Purpose | Main authority level |
| --- | --- | --- |
| `ffbayes pre-draft` | rebuild the supported workflow | authoritative runtime |
| `ffbayes pre-draft --stage-pages` | rebuild the supported workflow and stage `site/` | authoritative runtime, then derived publish surface |
| `ffbayes draft-strategy` | rebuild board artifacts from current processed inputs | authoritative runtime |
| `ffbayes stage-dashboard --year <year>` | regenerate HTML from authoritative payload and stage `site/` | authoritative runtime, then derived publish surface |
| `ffbayes draft-retrospective --import-finalized ... --ingest-only --year <year>` | import finalized artifacts | authoritative runtime |
| `ffbayes publish --year <year>` | stage `site/` and mirror selected runtime artifacts into cloud storage | derived publish surface and derived cloud mirror |
