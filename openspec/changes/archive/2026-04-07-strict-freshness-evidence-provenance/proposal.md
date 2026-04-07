## Why

FFBayes already has a credible pre-draft command center, but its trust contract is weaker than its presentation. The current pipeline can proceed with stale seasons, the staged Pages dashboard does not clearly expose artifact lineage, and the backtest is available only as a thin summary instead of a decision-evidence surface that helps users judge when to trust the board.

## What Changes

- Enforce explicit freshness behavior across the pre-draft pipeline so stale inputs are never silently tolerated.
- Add a decision-evidence panel to the draft dashboard and exported artifacts that turns the existing backtest into inspectable evidence, limitations, and failure-mode context.
- Add publish-time provenance metadata for staged `site/` artifacts so users can see when the dashboard was generated, what source windows it used, and whether freshness requirements were satisfied.
- Surface degraded or blocked freshness states in the CLI, runtime artifacts, and Pages payload instead of relying on implicit config or hidden diagnostics.
- Update tests and documentation for the stricter freshness policy, the new evidence panel, and the expanded Pages metadata.

## Capabilities

### New Capabilities
- `freshness-governance`: define and enforce explicit freshness policy for pre-draft inputs, backtests, and runtime gating.
- `decision-evidence-panel`: present backtest evidence, limitations, and decision-support context in the draft dashboard and related exports.
- `publish-provenance`: attach publish-time lineage and freshness metadata to staged Pages artifacts and exposed dashboard payloads.

### Modified Capabilities
- None.

## Impact

- Affected CLI and pipeline surfaces: `ffbayes pre-draft`, `ffbayes draft-backtest`, `ffbayes draft-strategy`, and `ffbayes publish-pages`.
- Affected code areas likely include `src/ffbayes/utils/analysis_windows.py`, `src/ffbayes/analysis/draft_decision_backtest.py`, `src/ffbayes/draft_strategy/draft_decision_system.py`, `src/ffbayes/publish_pages.py`, `config/pipeline_pre_draft.json`, `site/`, and related tests.
- Runtime artifacts under `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/...` and staged Pages payloads under repo-tracked `site/` will gain stricter freshness semantics and additional provenance/evidence fields.
- Path and environment-variable risk: any current workflow that depends on `FFBAYES_ALLOW_STALE_SEASON` or manually staged `site/` payloads may need migration messaging and explicit opt-in handling.
- Phase risk is limited because the repo only supports `pre_draft`, but this change must preserve current artifact names and locations unless an explicit migration path is documented.
