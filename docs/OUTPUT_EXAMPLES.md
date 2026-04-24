# Output Examples

Audience: readers who need to recognize the output files and understand what the
examples mean.

Scope: supported `ffbayes pre-draft` outputs first, then clearly labeled optional analyses.

Trust boundary: runtime artifacts are authoritative. Repo `dashboard/` and
`site/` are derived. Optional analyses are not default `pre-draft` outputs.

## What This Is

This document shows what the current output files look like and how to read the
important fields.

## When To Use It

Use this document to answer:

- what files does `ffbayes pre-draft` actually produce?
- which files are authoritative versus derived copies?
- what does the dashboard payload contain at a high level?
- which outputs require a separate optional command?

## What To Inspect

Supported default artifacts:

- `seasons/<year>/draft_strategy/draft_board_<year>.xlsx`
- `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- `seasons/<year>/draft_strategy/draft_board_<year>.html`
- `seasons/<year>/draft_strategy/draft_decision_backtest_<year_range>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_<year>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_diagnostics_<year>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_validation_<year_range>.json`
- `seasons/<year>/diagnostics/validation/player_forecast_validation_summary_<year_range>.json`
- `<runtime-root>/dashboard/index.html`
- `dashboard/index.html`

Outputs from explicit non-default commands:

- `seasons/<year>/montecarlo_results/`
- `seasons/<year>/model_evaluation/`
- repo `site/` plus cloud `data/` and `Analysis/<date>/` after `ffbayes publish`

## Interpretation Boundaries

- Do not assume every file under `seasons/<year>/` comes from `ffbayes pre-draft`.
- Do not assume a staged Pages file is the authoritative local draft surface.
- Do not treat example ranges in optional analyses as guaranteed weekly outcomes.

## Supported Pre-Draft Outputs

### Draft Board Workbook

Purpose: authoritative runtime workbook for tabular review and draft-day backup.

Authoritative path:

```text
seasons/<year>/draft_strategy/draft_board_<year>.xlsx
```

What it typically contains:

- board ranking
- by-position views
- recommendation and availability sheets
- roster scenarios
- diagnostics and freshness summaries

### Dashboard Payload

Purpose: authoritative runtime JSON contract for the local dashboard.

Authoritative path:

```text
seasons/<year>/draft_strategy/dashboard_payload_<year>.json
```

Minimal example:

```json
{
  "generated_at": "2026-04-09T18:05:00",
  "runtime_controls": {
    "risk_tolerance_options": ["low", "medium", "high"],
    "supported_scoring_presets": ["standard", "half_ppr", "ppr"],
    "active_scoring_preset": "half_ppr"
  },
  "analysis_provenance": {
    "overall_freshness": {
      "status": "fresh",
      "override_used": false,
      "warnings": []
    }
  },
  "decision_evidence": {
    "status": "available",
    "headline": "Contextual draft score outperforms the simple VOR proxy in backtests.",
    "winner": "draft_score",
    "season_count": 4
  },
  "metric_glossary": {
    "draft_score": {
      "label": "Board value score"
    },
    "replacement_delta": {
      "label": "Simple VOR proxy"
    }
  }
}
```

What to notice:

- `runtime_controls` tells you which local controls the dashboard expects
- `analysis_provenance` and `decision_evidence` carry trust messaging
- `winner` names the internally stronger strategy for the evaluated holdout
  comparison
- `season_count` tells you how many seasons supported that internal comparison
- `metric_glossary` and `model_overview` define canonical names and interpretation language
- `war_room_visuals` may be present in newer dashboard builds, but should be treated as additive
- if a validation metric is unavailable, runtime payloads render it as `n/a` or `not estimable` rather than fabricating `0.00`

### Dashboard HTML

Purpose: authoritative runtime HTML surface paired with the payload above.

Authoritative path:

```text
seasons/<year>/draft_strategy/draft_board_<year>.html
```

Derived local shortcut:

```text
dashboard/index.html
```

What to notice:

- the runtime HTML plus runtime payload are the authoritative local pair
- `dashboard/index.html` is a convenience copy for local use
- `site/index.html` is the staged Pages copy, not the working source of truth

### Decision Backtest

Purpose: internal holdout evidence for the `Decision evidence` surface.

Authoritative path:

```text
seasons/<year>/draft_strategy/draft_decision_backtest_<year_range>.json
```

Minimal example:

```json
{
  "evaluation_scope": {
    "type": "internal_holdout"
  },
  "overall": {
    "by_strategy": [
      {
        "strategy": "draft_score",
        "mean_lineup_points": 201.4
      },
      {
        "strategy": "historical_vor_proxy",
        "mean_lineup_points": 198.7
      }
    ]
  }
}
```

What to notice:

- this is evidence for comparative board behavior on holdout seasons
- it is not a guarantee about your future league
- read the strategy names as internal policy labels, not as proof that one
  strategy is universally best
- if the artifact disagrees with the dashboard evidence surface, treat that as a
  staging or freshness problem to investigate before draft use

### Staged GitHub Pages Copy

Purpose: derived publishing surface for GitHub Pages.

Derived paths:

```text
site/index.html
site/dashboard_payload.json
site/publish_provenance.json
```

Minimal provenance example:

```json
{
  "schema_version": "publish_provenance_v1",
  "season_year": 2026,
  "source_html": "draft_board_2026.html",
  "source_payload": "dashboard_payload_2026.json",
  "surface_sync": {
    "status": "synchronized"
  }
}
```

What to notice:

- `site/` is derived and publish-oriented
- `surface_sync.status` should be `synchronized` when the staged HTML, payload,
  and provenance agree
- `publish_provenance.json` records how the staged copy was built
- if the staged copy disagrees with authoritative runtime artifacts, use the
  runtime artifacts as source truth and restage rather than editing `site/` by
  hand

## Optional Analyses

These require separate commands and should not be presented as default `ffbayes pre-draft` outputs.

### Bayesian Versus VOR Comparison

Purpose: compare Bayesian and VOR approaches directly.

Command:

```bash
ffbayes bayesian-vor
```

This is not "just get VOR rankings." It is a comparison command.

### Monte Carlo Historical Analysis

Purpose: run Monte Carlo analysis directly.

Command:

```bash
ffbayes mc
```

Typical optional artifact family:

```text
seasons/<year>/montecarlo_results/
```

### Rolling Forecast Validation

Purpose: compare production player forecasts against simpler baseline expectations.

Command:

```bash
ffbayes bayesian-vor
```

Typical additional artifact family:

```text
seasons/<year>/model_evaluation/
```

### Public Publish

Purpose: stage the GitHub Pages copy and mirror selected runtime artifacts into cloud storage.

Command:

```bash
ffbayes publish --year 2026
```

Derived publish paths:

```text
site/
data/
Analysis/<date>/
```

## Commands And Paths

Supported workflow commands:

```bash
ffbayes pre-draft
ffbayes pre-draft --stage-pages
ffbayes draft-strategy
ffbayes stage-dashboard --year 2026
ffbayes draft-retrospective --year 2026
```

Optional commands:

```bash
ffbayes bayesian-vor
ffbayes mc
ffbayes publish --year 2026
```
