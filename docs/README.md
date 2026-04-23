# FFBayes Documentation

This guide suite explains the supported `pre-draft` workflow, the current dashboard trust surfaces, and the difference between authoritative runtime artifacts and derived copies.

## Quick Route By Audience

- operator guide: [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md)
- layperson guide: [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md)
- technical and statistician guide: [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)
- metric reference: [METRIC_REFERENCE.md](METRIC_REFERENCE.md)
- path and lineage guide: [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md)
- supported and optional output examples: [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)
- repo-level workflow summary: [../README.md](../README.md)

## Trust Model

- authoritative runtime artifacts live under the configured runtime root
- use `<runtime-root>` as the generic local path label in this guide suite
- repo `dashboard/` is a derived local shortcut for convenience
- repo `site/` is a staged GitHub Pages copy for publishing, not the authoritative local draft surface
- dashboard evidence is internal holdout evidence, not external validation

## Essential Commands

```bash
ffbayes pre-draft
ffbayes draft-strategy
ffbayes stage-dashboard --year 2026
ffbayes refresh-dashboard --year 2026
ffbayes draft-retrospective --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* --ingest-only --year 2026
```

## Start Here

If you are drafting today:

1. Read [../README.md](../README.md) for the supported workflow.
2. Read [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md) for the local dashboard, Pages staging, finalize flow, and retrospective path.
3. Use [METRIC_REFERENCE.md](METRIC_REFERENCE.md) if a dashboard term is unclear.

If you want a plain-language explanation:

1. Read [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md).
2. Use [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) to see the artifact shapes.

If you want the implemented math:

1. Read [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md).
2. Use [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md) to connect code, paths, and emitted artifacts.

## Authority Levels

### Authoritative Runtime

- `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- `seasons/<year>/draft_strategy/draft_board_<year>.html`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_<year>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_validation_<year_range>.json`
- `seasons/<year>/diagnostics/validation/player_forecast_validation_summary_<year_range>.json`

### Derived Local Shortcut

- `<runtime-root>/dashboard/index.html`
- `<runtime-root>/dashboard/dashboard_payload.json`
- `dashboard/index.html`
- `dashboard/dashboard_payload.json`

### Staged Pages Copy

- `site/index.html`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`

### Optional Publish Targets

- cloud `data/`
- cloud `Analysis/<date>/`

## Documentation Conventions

The guide suite follows one stable pattern:

- audience, scope, and trust boundary appear near the top
- canonical metric names stay consistent across guides
- commands and paths are paired with their purpose and authority level
- optional analyses are labeled optional instead of blended into the default workflow

## What This Guide Suite Does Not Do

- it does not treat GitHub Pages as the primary local draft surface
- it does not present optional analyses as default `ffbayes pre-draft` outputs
- it does not treat internal evidence as external proof
