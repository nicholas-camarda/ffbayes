# Data Lineage And Paths

Audience: operators and contributors who need to know where artifacts come from, where they live, and which path is authoritative.

Scope: the supported `pre-draft` workflow, runtime roots, repo-local shortcuts, staged Pages files, and optional cloud publish targets.

Trust boundary: authoritative runtime artifacts live under the configured runtime root. Repo `dashboard/`, repo `site/`, and cloud mirrors are derived surfaces with different purposes.

## What This Is

This guide maps the path flow from collected source data to the final board, Pages staging, and retrospective artifacts.

## When To Use It

Use this guide when:

- you need to find a generated artifact
- you want to know whether a path is authoritative or derived
- you need to understand what `stage-dashboard`, `publish-pages`, or `publish` changes
- you are auditing lineage from raw data to final board

## What To Inspect

Primary path policy lives in:

- `src/ffbayes/utils/path_constants.py`

Primary commands that create or move artifacts:

- `ffbayes pre-draft`
- `ffbayes draft-strategy`
- `ffbayes stage-dashboard`
- `ffbayes refresh-dashboard`
- `ffbayes publish-pages`
- `ffbayes draft-retrospective`
- `ffbayes publish`

## What Not To Infer

- Do not assume a repo path is authoritative just because it is easy to open.
- Do not assume cloud-mirrored data is the working source of truth.
- Do not assume `site/` updates itself without an explicit staging command.

## Path Categories

| Category | Example path | Purpose | Authority |
| --- | --- | --- | --- |
| runtime root | `~/ProjectsRuntime/ffbayes` | main local working tree | authoritative base |
| raw runtime data | `data/raw/season_datasets/` under runtime root | collected season-level inputs | authoritative runtime input |
| processed runtime data | `data/processed/` under runtime root | cleaned and derived inputs | authoritative runtime input |
| unified dataset | `data/processed/unified_dataset/unified_dataset.csv` under runtime root | analysis-ready player dataset | authoritative runtime input |
| canonical pre-draft artifacts | `runs/<year>/pre_draft/artifacts/` under runtime root | main analysis outputs | authoritative runtime output |
| repo dashboard shortcut | `dashboard/index.html` | easy local opening path | derived local shortcut |
| staged Pages copy | `site/index.html` | repo-tracked publishing surface | derived publish surface |
| cloud mirror | `data/` under cloud root | mirrored selected runtime artifacts | derived mirror |
| cloud snapshot | `Analysis/<date>/` under cloud root | dated publish snapshot | derived mirror |

## Lineage From Source Data To Final Board

### 1. Raw Collection

Purpose: collect source season data into the runtime raw-data tree.

Typical area:

```text
<runtime-root>/data/raw/season_datasets/
```

### 2. Validation

Purpose: confirm required seasons and expected inputs exist.

Trust surface:

- freshness manifests and warnings later exposed in `analysis_provenance` and `decision_evidence`

### 3. Preprocessing

Purpose: build analysis-ready datasets from collected raw inputs.

Typical area:

```text
<runtime-root>/data/processed/
```

### 4. Unified Dataset Construction

Purpose: produce the player-level dataset used by the board and retrospective.

Canonical path:

```text
<runtime-root>/data/processed/unified_dataset/unified_dataset.csv
```

### 5. Pre-Draft Artifact Generation

Purpose: create the board workbook, payload, HTML, and decision backtest.

Canonical area:

```text
<runtime-root>/runs/<year>/pre_draft/artifacts/draft_strategy/
```

Typical files:

- `draft_board_<year>.xlsx`
- `dashboard_payload_<year>.json`
- `draft_board_<year>.html`
- `draft_decision_backtest_<year_range>.json`

### 6. Repo-Local Shortcut Staging

Purpose: make the current board easy to open from the repo root.

Derived paths:

- `dashboard/index.html`
- `dashboard/dashboard_payload.json`

These are derived convenience copies, not the source of truth.

### 7. GitHub Pages Staging

Purpose: copy the current dashboard into repo-tracked `site/`.

Command:

```bash
ffbayes stage-dashboard --year 2026
```

Lower-level compatibility command:

```bash
ffbayes publish-pages --year 2026
```

Derived paths:

- `site/index.html`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`

### 8. Finalized Draft Import And Retrospective

Purpose: move browser-downloaded finalized artifacts into the canonical runtime folder, then evaluate them later against realized outcomes.

Canonical finalized import path:

```text
<runtime-root>/runs/<year>/pre_draft/artifacts/draft_strategy/finalized_drafts/
```

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
| `ffbayes draft-strategy` | rebuild board artifacts from current processed inputs | authoritative runtime |
| `ffbayes stage-dashboard --year <year>` | regenerate HTML from authoritative payload and stage `site/` | authoritative runtime, then derived publish surface |
| `ffbayes refresh-dashboard --year <year>` | regenerate HTML from authoritative payload | authoritative runtime, then derived shortcuts |
| `ffbayes publish-pages --year <year>` | stage `site/` from existing runtime dashboard | derived publish surface |
| `ffbayes draft-retrospective --import-finalized ... --ingest-only --year <year>` | import finalized artifacts | authoritative runtime |
| `ffbayes publish --year <year>` | mirror selected runtime artifacts into cloud storage | derived cloud mirror |
