# FFBayes documentation

FFBayes is a local fantasy-football draft board. The public repository contains
the engine, validation rules, and a generic example league profile; users keep
their own league settings in ignored `config/leagues/*.local.json` files.

The operator workflow is one command:

```bash
ffbayes dashboard --year 2026
```

## Guide map

- [Dashboard operator guide](DASHBOARD_OPERATOR_GUIDE.md) — setup and draft-day use.
- [Data lineage and paths](DATA_LINEAGE_AND_PATHS.md) — public inputs, run roots, and provenance.
- [Metric reference](METRIC_REFERENCE.md) — equations and interpretation.
- [Output examples](OUTPUT_EXAMPLES.md) — validated payload and snapshot shape.
- [Technical deep dive](TECHNICAL_DEEP_DIVE.md) — implemented model flow.
- [Dashboard architecture](DASHBOARD_FRONTEND_ARCHITECTURE.md) — service and browser responsibilities.
- [Frontend maintenance](DASHBOARD_FRONTEND_CUTOVER.md) — the optional React template workflow.
- [Layperson guide](LAYPERSON_GUIDE.md) — plain-language explanation.

## How to read a result

The board is decision support, not a promise about player outcomes. Rankings
are produced from the current validated projections, league scoring, roster
demand, replacement levels, scarcity, and market ADP. Recommendations also use
the runtime draft slot and current pick when those values are available.

If a required source is inaccessible, incomplete, stale, or semantically
invalid, the pipeline reports the failure and the dashboard stays blocked. It
does not reuse an older board or invent a neutral value.
