# FFBayes Documentation

Use this page to choose the right doc. The guides split plain-language use,
operator workflow, technical math, metric labels, artifact examples, and path
authority.

## Quick Route By Audience

- drafting today: [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md)
- plain-language explanation: [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md)
- implemented math and statistics: [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)
- dashboard label lookup: [METRIC_REFERENCE.md](METRIC_REFERENCE.md)
- artifact shapes: [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)
- path authority and lineage: [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md)
- repo setup and quick workflow: [../README.md](../README.md)

## Trust Model

- authoritative runtime artifacts live under the configured runtime root
- use `<runtime-root>` as the generic local path label in this guide suite
- repo `dashboard/` is a derived local shortcut for convenience
- repo `site/` is a staged GitHub Pages copy for publishing, not the authoritative local draft surface
- dashboard evidence is internal holdout evidence, not external validation
- `n/a` or `not estimable` validation entries mean the slice could not support that estimate cleanly

## Start Here

If you are drafting today:

1. Read [../README.md](../README.md) for the supported workflow.
2. Read [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md) for the local dashboard, Pages staging, finalize flow, and retrospective path.
3. Use [METRIC_REFERENCE.md](METRIC_REFERENCE.md) if a dashboard term is unclear.

If you want a plain-language explanation:

1. Read [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md).
2. Use [METRIC_REFERENCE.md](METRIC_REFERENCE.md) when a dashboard label needs a short definition.
3. Use [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) to see the artifact shapes.

If you want the implemented math:

1. Read [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md).
2. Use [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md) to connect code, paths, and emitted artifacts.

## Commands And Paths

This index does not duplicate the workflow commands or artifact authority table.
Use:

- [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md) for the supported
  draft-day commands, dashboard opening path, finalize flow, and retrospective
  import sequence
- [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md) for authoritative
  runtime artifacts, derived local shortcuts, staged Pages copies, and optional
  publish targets

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
