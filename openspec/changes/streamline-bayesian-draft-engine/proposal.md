## Why

The current pre-draft modeling stack mixes an active empirical-Bayes player model, a legacy hybrid Monte Carlo plus random-forest uncertainty path, and several older "Bayesian" or "hierarchical" surfaces that no longer describe the supported draft board honestly. That makes the dashboard hard to trust, the code hard to maintain, and the statistical claims harder to defend than the actual implementation.

This change replaces the mixed model stack with one explicit draft-engine contract: a primary player forecast model that targets season-total fantasy value for the dashboard, a rigorously validated draft-decision layer built on that forecast, and a single documented path through the `pre-draft` pipeline. The hybrid MC/random-forest path is removed completely rather than retained as a fallback or compatibility layer.

## What Changes

- Replace the current mixed player-model story with a single primary draft forecast stack centered on the empirical-Bayes player model, while implementing a sampled hierarchical evaluation lane against the same dashboard target and keeping empirical Bayes as the production default unless the sampled path satisfies explicit promotion criteria.
- If the sampled hierarchical evaluation lane does not materially improve holdout quality, calibration, rookie handling, and operational stability, it will not remain in the supported workflow and should be removed rather than retained as dormant statistical machinery.
- Redefine the player forecast target around season-total fantasy value for the supported scoring presets, with explicit decomposition of scoring-rate and availability contributions in the model design and diagnostics.
- Add required model structure beyond player and position, including:
  - team-season effects
  - rookie draft-capital priors
  - rookie combine priors
  - rookie depth-chart priors
  - validated team-context features
  - limited schedule-strength context features when they improve holdout performance
- Require explicit exploration of sampled-Bayesian training and inference settings so the full Bayesian path is tuned against holdout performance, calibration, convergence quality, and runtime cost instead of being fixed from one arbitrary configuration.
- **BREAKING** Remove the hybrid Monte Carlo plus random-forest uncertainty path from the supported pipeline, codebase entrypoints, runtime artifacts, docs, and dashboard/evidence surfaces.
- **BREAKING** Remove legacy hybrid integration modules, hybrid draft strategy outputs, and obsolete "Bayesian hierarchical" labels that no longer match the supported implementation.
- Add rolling out-of-sample reporting for forecast accuracy, ranking quality, and calibration, including rookie/veteran and position slices.
- Explicitly map the redesigned player-model outputs into the live dashboard contract, including recommendation lanes, inspector fields, evidence content, glossary terms, and `war_room_visuals` payload semantics.
- Define an exact, minimal output structure for runtime artifacts, diagnostics, repo-local dashboard shortcuts, and staged Pages files so the redesign does not create output sprawl.
- **BREAKING** Rename the canonical runtime path contract from `data/` and `runs/` to `inputs/` and `seasons/`, remove the redundant `pre_draft/` and `artifacts/` layers, and delete the legacy runtime `datasets/` tree.
- Add a small but interpretable stress-test fixture that exercises the end-to-end pre-draft pipeline, dashboard payload generation, dashboard smoke path, and publication/staging path without requiring a full production run.
- Sequence the remaining implementation work so cleanup, hybrid retirement, dashboard simplification, and documentation alignment are completed before any sampled-hierarchical evaluation lane is expanded further.
- Update documentation so README, technical docs, dashboard operator docs, and evidence descriptions all describe the same supported model stack and artifact flow.
- Require a final documentation sweep so README, `docs/`, docstrings, and essential code comments accurately describe the implemented system and its operator-facing behavior.
- Add final implementation-audit checks at the end of the change to confirm the shipped code, tests, dashboard, and docs match the proposal and no retired hybrid path remains.

## Capabilities

### New Capabilities
- `player-forecast-model-stack`: Defines the single supported player forecast architecture for the draft engine, including season-total target definition, empirical-Bayes primary behavior, team-season effects, rookie priors, and sampled hierarchical extension requirements.
- `model-validation-and-stress-harness`: Defines rolling holdout evaluation, calibration reporting, slice-based diagnostics, and a small end-to-end stress-test fixture for the pipeline, dashboard, and Pages staging path.
- `hybrid-analysis-retirement`: Defines complete removal of the hybrid Monte Carlo plus random-forest path, associated CLI/runtime/doc surfaces, and any unsupported fallback logic.

### Modified Capabilities
- `decision-evidence-panel`: The dashboard evidence contract changes to show the new primary-model diagnostics, out-of-sample accuracy/calibration summaries, and explicit provenance for the supported player forecast stack.
- `war-room-decision-visualizations`: The live dashboard visualization contract changes so timing, scarcity, and inspector-linked comparative views are driven by the redesigned player-model semantics and supported dashboard payload fields.
- `documentation-guide-suite`: The published documentation contract changes to remove hybrid-path guidance and describe the single supported pre-draft model/pipeline path with current artifact names and examples.
- `visualization-surface-governance`: The authoritative-versus-derived surface contract changes to adopt the new runtime path structure and to forbid duplicate runtime `datasets/`, `data/`, `runs/`, or flattened shadow trees.

## Impact

- Affected modules include `src/ffbayes/analysis/bayesian_player_model.py`, `src/ffbayes/analysis/bayesian_vor_comparison.py`, `src/ffbayes/draft_strategy/draft_decision_system.py`, `src/ffbayes/draft_strategy/draft_decision_strategy.py`, `src/ffbayes/run_pipeline_split.py`, `src/ffbayes/cli.py`, and the current hybrid/legacy analysis and draft modules under `src/ffbayes/analysis/` and `src/ffbayes/draft_strategy/`.
- Affected commands include `ffbayes pre-draft`, `ffbayes draft-strategy`, `ffbayes draft-backtest`, `ffbayes bayesian-vor`, and any hybrid-specific commands or aliases that currently expose retired behavior.
- Runtime artifacts under `runs/<year>/pre_draft/artifacts/` will change in content and provenance; any hybrid-specific runtime outputs become unsupported and should be removed rather than mirrored or staged. New outputs MUST fit the declared runtime structure instead of creating ad hoc artifact families.
- Runtime path constants, docs, and tests will migrate from `data/` and `runs/<year>/pre_draft/artifacts/...` to `inputs/` and `seasons/<year>/...`; cloud mirrors remain under `data/` and dated `Analysis/<date>/` according to the existing workspace-governor-style split layout.
- `site/` Pages staging and local dashboard shortcuts remain supported, but dashboard payload evidence content, inspector semantics, glossary text, and `war_room_visuals` fields will change to reflect the new primary model stack.
- Documentation updates will touch README, technical/statistical guidance, dashboard operator docs, and any examples or contracts that still mention the retired hybrid path.
- This change increases dependence on the existing `ffbayes` conda environment and the current `nflreadpy` data backend for rookie priors and contextual feature collection, while preserving the `pre_draft` phase and canonical runtime path policy in `src/ffbayes/utils/path_constants.py`.
