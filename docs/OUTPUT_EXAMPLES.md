# Output Examples

Audience: operators and readers who want concrete artifact shapes without reading all of the implementation details.

Scope: supported `ffbayes pre-draft` outputs first, then clearly labeled optional analyses.

Trust boundary: runtime artifacts are authoritative. Repo `dashboard/` and repo `site/` are derived surfaces. Optional analyses are not default outputs of the supported pre-draft workflow.

## What This Is

This document shows the shapes of the outputs you should expect from the current workflow.

## When To Use It

Use this document when you need to answer:

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

Additional outputs from explicit commands:

- `seasons/<year>/montecarlo_results/`
- `seasons/<year>/model_evaluation/`
- cloud `data/` and `Analysis/<date>/` after `ffbayes publish`

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
    "winner": "draft_score"
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
- the repository uses the empirical-Bayes player forecast

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
- `publish_provenance.json` records how the staged copy was built

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

### Cloud Publish

Purpose: mirror selected runtime artifacts into cloud storage.

Command:

```bash
ffbayes publish --year 2026
```

Derived cloud paths:

```text
data/
Analysis/<date>/
```

## Commands And Paths

Supported workflow commands:

```bash
ffbayes pre-draft
ffbayes draft-strategy
ffbayes stage-dashboard --year 2026
ffbayes refresh-dashboard --year 2026
ffbayes draft-retrospective --year 2026
```

Optional commands:

```bash
ffbayes bayesian-vor
ffbayes mc
ffbayes publish --year 2026
```
