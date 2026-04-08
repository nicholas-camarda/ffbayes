## Why

FFBayes now has one credible visualization product, the live draft dashboard, but the surrounding visualization layer has drifted into an inconsistent state. The repo currently mixes authoritative dashboard artifacts, stale or misleading local shortcuts, weak legacy static plots, and documentation that promises more diagnostics coverage than the runtime outputs actually provide.

## What Changes

- Define a visualization governance contract that separates authoritative visualization surfaces from deprecated or convenience-only artifacts.
- Narrow repo-local `dashboard/` to the convenience HTML/payload pair and direct shortcut companions instead of treating it as a broad local artifact namespace.
- Repair dashboard lifecycle expectations so the repo-root shortcut, runtime dashboard, and staged `site/` bundle cannot silently diverge in content or trust state.
- Keep the current staged Pages inline-payload bootstrap in scope for this change and defer any lighter-weight load-path redesign to a later follow-up.
- Tighten dashboard evidence, freshness, and provenance presentation so degraded or stale outputs explain why they are degraded instead of merely surfacing ambiguous status.
- Reduce or retire low-value legacy visualization paths that are no longer the product surface, including pipeline-referenced legacy generators, standalone generators that are only exposed through unused manual CLI entrypoints, and redundant compatibility wrappers.
- Align visualization documentation with the outputs the repo actually generates, and make missing or intentionally removed diagnostics categories explicit.
- Add validation coverage for visualization synchronization and local shortcut integrity in addition to the existing Pages-focused checks.

## Capabilities

### New Capabilities
- `visualization-surface-governance`: define which visualization surfaces are authoritative, which are derived convenience copies, which unused legacy generators should be deprecated for removal, and how documented visualization output categories map to real artifacts.

### Modified Capabilities
- `dashboard-artifact-lifecycle-automation`: strengthen synchronization requirements across runtime, repo-local shortcut, and `site/` staging so stale or mismatched dashboard artifacts are detectable and repairable.
- `decision-evidence-panel`: refine evidence-surface behavior so evidence status, limitations, redundancy, and disagreement summaries remain interpretable across runtime and staged dashboards.
- `freshness-governance`: strengthen how degraded or mixed freshness states are explained in visualization-facing surfaces rather than only serialized.
- `publish-provenance`: refine staged provenance behavior so Pages viewers can understand stale, mixed, or override-driven states with concrete reasons.

## Impact

- Affects dashboard generation and synchronization code in `src/ffbayes/draft_strategy/draft_decision_system.py`, `src/ffbayes/refresh_dashboard.py`, and `src/ffbayes/publish_pages.py`.
- Affects legacy or overlapping visualization modules under `src/ffbayes/visualization/` and related diagnostics emitters such as `src/ffbayes/analysis/draft_strategy_comparison.py`, with `create_pre_draft_visualizations.py` treated as legacy but still pipeline-referenced, `create_team_aggregation_visualizations.py` and `draft_strategy_comparison.py` treated as deprecated unused manual-entrypoint surfaces targeted for removal, and `create_consolidated_hybrid_visualizations.py` treated as a removal candidate.
- Affects runtime artifacts under `runs/<year>/pre_draft/artifacts/`, `runs/<year>/pre_draft/diagnostics/`, repo-local `dashboard/`, and repo-tracked `site/`.
- Affects visualization docs and examples in `README.md`, `docs/README.md`, and `docs/OUTPUT_EXAMPLES.md`.
- Affects validation coverage in Python tests and dashboard smoke paths; no phase migration is intended, and canonical `pre_draft` artifact naming should remain stable.
