# Dashboard Operator Guide

Audience: someone running the supported draft-day workflow and using the dashboard as the main local decision surface.

Scope: the current `pre-draft` workflow, the authoritative runtime dashboard, the repo-local shortcut, staged Pages behavior, finalize flow, and the post-draft ingest and retrospective path.

Trust boundary: the authoritative dashboard is the runtime HTML plus runtime payload under the configured runtime root. Repo `dashboard/` and repo `site/` are derived surfaces. Decision evidence is internal holdout evidence, not external proof.

## What This Is

This guide explains how to operate the current dashboard from setup through draft-day use and post-draft follow-up.

## When To Use It

Use this guide when you need to:

- run the supported workflow end to end
- know which dashboard file to open
- understand what each dashboard section is for
- stage GitHub Pages without confusing it with the local working surface
- import finalized draft artifacts and run the retrospective later

## What To Inspect

Primary commands:

- `ffbayes pre-draft`
- `ffbayes draft-strategy`
- `ffbayes stage-dashboard --year <year>`
- `ffbayes refresh-dashboard --year <year>`
- `ffbayes publish-pages --year <year>`
- `ffbayes draft-retrospective --import-finalized ... --ingest-only --year <year>`
- `ffbayes draft-retrospective --year <year>`

Primary surfaces:

- authoritative runtime HTML: `seasons/<year>/draft_strategy/draft_board_<year>.html`
- authoritative runtime payload: `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- derived runtime-root shortcut: `<runtime-root>/dashboard/index.html`
- derived local shortcut: `dashboard/index.html`
- staged Pages copy: `site/index.html`

## What Not To Infer

- Do not treat `site/index.html` as the authoritative local dashboard.
- Do not treat the recommendation lanes as externally validated truth.
- Do not assume a visual panel is a separate model. If present, it is an interpretation layer on top of the same board.

## Surface Authority

| Surface | Purpose | Authority |
| --- | --- | --- |
| `seasons/<year>/draft_strategy/draft_board_<year>.html` | authoritative local HTML | authoritative |
| `seasons/<year>/draft_strategy/dashboard_payload_<year>.json` | authoritative local payload | authoritative |
| `<runtime-root>/dashboard/index.html` | easy local shortcut beside runtime artifacts | derived local shortcut |
| `<runtime-root>/dashboard/dashboard_payload.json` | paired runtime shortcut payload | derived local shortcut |
| `dashboard/index.html` | easy local shortcut from the repo root | derived local shortcut |
| `dashboard/dashboard_payload.json` | paired shortcut payload | derived local shortcut |
| `site/index.html` | staged GitHub Pages copy | derived publish surface |
| `site/dashboard_payload.json` | staged Pages payload | derived publish surface |
| `site/publish_provenance.json` | staged Pages publish metadata | derived publish surface |

## Commands And Paths

### Full Supported Refresh

Purpose: rebuild the supported pre-draft stack from source data through dashboard artifacts.

```bash
ffbayes pre-draft
```

### Board-Only Refresh

Purpose: regenerate the board and dashboard from current processed inputs and current league settings.

```bash
ffbayes draft-strategy
```

### One-Step Public Dashboard Refresh

Purpose: rebuild HTML from the authoritative payload and restage the repo's GitHub Pages copy in one step.

```bash
ffbayes stage-dashboard --year 2026
```

### HTML-Only Refresh

Purpose: rebuild HTML from an authoritative payload without rerunning the broader analysis stack.

```bash
ffbayes refresh-dashboard --year 2026
```

### Derived-Surface Drift Check

Purpose: verify whether a target HTML file still matches regeneration from its authoritative payload.

```bash
ffbayes refresh-dashboard --check --json \
  --payload-path /path/to/dashboard_payload.json \
  --output-html /path/to/index.html
```

### Stage GitHub Pages Directly

Purpose: copy the current validated dashboard into repo-tracked `site/` without rerendering the dashboard first.

```bash
ffbayes publish-pages --year 2026
```

### Import Finalized Draft Artifacts

Purpose: move browser-downloaded finalized artifacts into the canonical runtime folder.

```bash
ffbayes draft-retrospective \
  --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* \
  --ingest-only \
  --year 2026
```

## Before The Draft

1. Run `ffbayes pre-draft` unless you are intentionally doing a narrower refresh.
2. Open `dashboard/index.html`.
3. Confirm the dashboard reflects your league settings.
4. Check `Freshness and provenance` before trusting the board.
5. Keep the workbook handy as a tabular backup.

What to inspect in the payload if you need to audit the run:

```json
{
  "runtime_controls": {
    "risk_tolerance_options": ["low", "medium", "high"],
    "supported_scoring_presets": ["standard", "half_ppr", "ppr"],
    "active_scoring_preset": "half_ppr"
  },
  "analysis_provenance": {
    "overall_freshness": {
      "status": "fresh",
      "override_used": false
    }
  }
}
```

What to notice:

- `runtime_controls` tells you what the dashboard expects locally
- `overall_freshness` tells you whether the underlying inputs were fresh or degraded

## During The Draft

### Recommendation Lanes

Purpose: separate immediate take-now candidates from fallback options and safer wait candidates.

What to inspect:

- the top `pick now` candidate
- the `can wait` lane before passing on a player you like
- `why_flags`, `Availability to next pick`, and `Expected regret`

What not to infer:

- a `pick now` recommendation is not a guarantee the player is the only correct choice
- a `can wait` label is not certainty that the player will survive

### Player Inspector

Purpose: show the selected player's board value, baseline comparison, risk and upside signals, and supporting reasons.

What to inspect:

- `Board value score`
- `Simple VOR proxy`
- `Fragility score`
- `Upside score`
- `starter_delta`
- `why_flags`

Projection detail stays behind the `Projection breakdown` disclosure. That is where the dashboard shows:

- `Season total mean`
- `Rate when active`
- `Expected games`
- `Availability rate`
- `Current team`
- `Team change`

When relevant, the same section also surfaces rookie context such as draft pick, combine signal, and depth-chart rank. Keep that detail in the inspector; the main board should stay lean.

### Decision Evidence

Purpose: show internal backtest evidence and interpretation limits.

Production dashboard generation requires this evidence to be available and fresh.
Missing, unavailable, or degraded evidence is a failed dashboard build, not a
successful dashboard with a warning.

Minimal shape:

```json
{
  "decision_evidence": {
    "status": "available",
    "headline": "Contextual draft score outperforms the simple VOR proxy in backtests.",
    "winner": "draft_score",
    "season_count": 4
  }
}
```

What to inspect:

- evidence status should be `available` on production dashboards
- the winner label and season count
- the compact cohort validation table in the first view
- interpretation limits and freshness status
- any `n/a` or `not estimable` validation entries as a sign that the slice could not support that metric cleanly

What not to infer:

- this is not external validation
- degraded or unavailable evidence belongs in non-production investigation only
- `n/a` in a validation table does not mean a measured zero relationship

The first visible evidence surface is intentionally compact:

- one summary block
- summary metrics
- one compact validation table

Season-level deltas, strategy rows, and disagreement tables live under the nested `Detailed evidence` disclosure.

Sampled-Bayes comparison artifacts are diagnostic history only. The live dashboard uses the empirical-Bayes estimator and does not show sampled comparison rows.

### Freshness And Provenance

Purpose: show whether the board and evidence were built from current expected inputs.

What to inspect:

- freshness status
- override usage
- warnings
- publish provenance if you are looking at a staged Pages copy

## War-Room Visuals When Present

If the current dashboard build includes war-room visuals, use them as quick interpretation aids:

- `Wait vs Pick Frontier`: timing tradeoff between value and next-pick survival
- `Positional Cliffs`: where a position group is about to thin out
- `Contextual vs baseline` explainer: why the contextual board differs from the `Simple VOR proxy`

What not to infer:

- these visuals do not create a separate model
- they do not strengthen the evidence beyond what the underlying `Decision evidence` surface already supports

## Finalize Flow

Purpose: capture the draft result in the format the retrospective expects later.

During the draft:

1. use the dashboard to track taken players and your roster
2. when the draft is over, click Finalize
3. keep the downloaded finalized bundle
4. do not treat the browser download folder as the canonical long-term storage location

Canonical runtime destination after import:

```text
seasons/<year>/draft_strategy/finalized_drafts/
```

The import and autodiscovery paths exclude finalized files named like `*_test.*`.
Those files are test artifacts, not production retrospective inputs.

## After The Draft

### Ingest Only

Purpose: import finalized artifacts and stop.

```bash
ffbayes draft-retrospective \
  --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* \
  --ingest-only \
  --year 2026
```

### Retrospective

Purpose: compare finalized draft artifacts against realized season outcomes.

```bash
ffbayes draft-retrospective --year 2026
```

Runtime-local retrospective artifacts:

- `seasons/<year>/draft_strategy/draft_retrospective_<year>.json`
- `seasons/<year>/draft_strategy/draft_retrospective_<year>.html`

## Publish Pages Only When Needed

Use `ffbayes stage-dashboard --year <year>` when you want the repo's public Pages copy updated and also want the dashboard HTML rerendered first.

Use `ffbayes publish-pages --year <year>` only as a lower-level staging helper when you intentionally want to copy the current runtime dashboard into `site/` without rerendering it.

What to inspect in the staged Pages payload:

```json
{
  "publish_provenance": {
    "schema_version": "publish_provenance_v1",
    "source_html": "draft_board_2026.html",
    "source_payload": "dashboard_payload_2026.json",
    "surface_sync": {
      "status": "synchronized"
    }
  }
}
```

What to notice:

- the staged Pages bundle records where it came from
- the staged site is for publishing, not local source-of-truth drafting
