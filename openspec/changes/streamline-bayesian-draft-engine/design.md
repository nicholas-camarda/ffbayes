## Context

The supported `ffbayes pre-draft` workflow currently produces the live dashboard and draft workbook through a path that already depends on `src/ffbayes/analysis/bayesian_player_model.py`, `src/ffbayes/draft_strategy/draft_decision_system.py`, and `src/ffbayes/analysis/draft_decision_backtest.py`. That active path is statistically more coherent than the repo's older hybrid Monte Carlo plus random-forest uncertainty lane, but the codebase and docs still expose both stories at once.

This creates four concrete problems:

1. The supported draft board is driven by an empirical-Bayes player-posterior table and a decision-policy layer, but legacy commands, modules, and artifacts still present the hybrid path as if it were an equally valid production surface.
2. The current player model target is not cleanly named relative to what the dashboard actually needs. The code aggregates player-season history from weekly `FantPt` means, while the draft engine conceptually serves a season-long decision.
3. Rookie handling is under-specified. The current player model can shrink rookies to position-level priors, but it does not incorporate draft capital, combine, or depth-chart context strongly enough to justify dashboard-facing uncertainty claims for unseen players.
4. Validation is split across useful but under-integrated surfaces. `bayesian_vor_comparison.py` already provides strong holdout diagnostics, while the dashboard evidence and pipeline stress coverage do not yet treat those diagnostics as a first-class contract.

Stakeholders are:

- the draft operator who needs one trustworthy dashboard path
- the maintainer who needs one supported modeling story
- the technical/statistical reader who needs the docs and evidence to match the code

Constraints:

- only the `pre_draft` phase is supported
- canonical runtime artifacts will migrate to `seasons/<year>/...` as part of this change
- repo-local `dashboard/` and staged `site/` remain derived surfaces
- the repo must not keep deprecated logic as fallback or compatibility rescue code

## Goals / Non-Goals

**Goals:**

- Define one supported player forecast contract for the draft engine and remove the hybrid MC/random-forest path completely.
- Align the player-model estimand with the dashboard's real need: season-total fantasy value for draft decisions.
- Preserve the current strength of the empirical-Bayes player model while extending it with defensible hierarchy:
  - team-season effects
  - rookie draft-capital priors
  - rookie combine priors
  - rookie depth-chart priors
  - validated team/opponent context features
- Make out-of-sample accuracy, ranking quality, and calibration a required artifact of the supported modeling path.
- Add a small but interpretable fixture that stress-tests the full supported path:
  - pre-draft modeling
  - dashboard payload generation
  - dashboard smoke behavior
  - Pages staging behavior
- Specify the explicit path from redesigned model outputs to a working dashboard, including draft board rows, inspector fields, glossary terminology, normalized `war_room_visuals`, repo-local `dashboard/`, and staged `site/`.
- Update dashboard evidence and documentation so they describe only the supported path.
- Require end-of-change specialist review passes for implementation validity, documentation consistency, and regression-risk coverage.

**Non-Goals:**

- This change does not introduce a second production forecast engine that competes with the primary player model.
- This change does not preserve hybrid MC/random-forest artifacts for backward compatibility, optional toggles, or hidden fallbacks.
- This change does not expand beyond the `pre_draft` phase into a new runtime phase model.
- This change does not make college-conference effects a required production feature; they remain explicitly out of scope unless later evidence shows incremental predictive value.
- This change does not require immediate promotion of sampled Bayes to the default production estimator if the empirical-Bayes baseline remains better calibrated, simpler, or more stable.

## Decisions

### Decision: The dashboard-serving estimand SHALL be season-total fantasy value

The supported player forecast target will be the posterior predictive distribution of season-total fantasy points under the active scoring preset, because that is the quantity the draft board needs for starter/replacement value and roster construction.

Rationale:

- Draft decisions are season-long allocation decisions, not weekly one-off picks.
- Players with similar per-game rates but different availability profiles should not collapse to the same draft-facing forecast.
- Rookie and fragile-player uncertainty is dominated by role and availability as much as by talent.

Alternatives considered:

- **Per-game fantasy points only**: simpler, but misaligned with season-long roster value and overstates players with unstable availability.
- **One direct black-box season-total model**: possible, but harder to interpret and less useful for decomposing uncertainty into scoring-rate and availability components.

### Decision: The primary model stack SHALL use a two-part player forecast architecture

The supported player forecast stack will model season-total fantasy value through two linked components:

1. scoring rate when active
2. expected games/availability

Season-total posterior summaries will be derived from posterior predictive simulation across those components rather than from a single opaque score or a purely analytic shortcut.

Rationale:

- Separates talent/role from durability/availability.
- Produces clearer dashboard semantics for players such as:
  - a veteran WR with elite rate but missed-game risk
  - a first-round rookie WR with uncertain target share but strong draft-capital prior
  - a committee RB with stable health but unstable role
- Creates cleaner hooks for team-season effects and rookie priors.

Alternatives considered:

- **Single direct season-total regression**: easier to ship, but harder to audit and harder to explain in dashboard evidence.
- **Weekly simulation-first architecture**: closer to the retired hybrid lane and less aligned with the current draft engine.

### Decision: Empirical Bayes remains the primary production estimator; sampled hierarchical Bayes is an extension path on the same contract

The production path will keep the empirical-Bayes player model as the primary estimator while redesigning it around the new target and hierarchical structure. A sampled hierarchical model can be prototyped against the same contract and promoted only if it wins on holdout accuracy, calibration, and operational stability.

Rationale:

- The current empirical-Bayes path is auditable and already integrated with the dashboard.
- The repo already depends on PyMC, so a sampled extension is feasible without changing the environment contract.
- Keeping one contract and two estimators avoids inventing another incompatible pipeline branch.
- "Hierarchical" describes the model structure, while "empirical Bayes" and "sampled Bayes" describe different inference strategies for that same structure. The supported comparison is therefore hierarchical empirical Bayes versus hierarchical sampled Bayes, not one model family versus another unrelated one.
- If the sampled hierarchical path does not materially improve the supported contract, it should not linger as a supported side branch. It may be retained temporarily as an implementation-phase evaluation artifact, but the shipped supported workflow should either promote it or remove it.

Alternatives considered:

- **Immediate full replacement with sampled Bayes**: too much simultaneous model and operational risk.
- **Keep empirical Bayes and sampled Bayes as co-primary runtime modes**: recreates the current confusion.
- **Keep a non-winning sampled estimator indefinitely in the supported repo path**: rejected because it recreates dormant model clutter and violates the repo rule against carrying deprecated statistical machinery.

### Decision: The sampled Bayesian path SHALL include bounded parameter exploration for training and inference

If a sampled hierarchical estimator is implemented in this change, it must not ship as a single untuned configuration. The implementation must include a bounded exploration procedure for training and inference settings, evaluated against the same production contract.

Settings to explore may include:

- prior scales and shrinkage choices for hierarchical effects
- centered vs non-centered parameterizations where relevant
- target acceptance / step-size adaptation settings
- chain count, warmup count, and retained draw count
- posterior predictive composition choices for rate and availability
- feature subsets or regularization choices specific to the sampled path

The first implementation MUST keep this search space bounded and concrete. The intended initial search space is:

- parameterization:
  - centered
  - non-centered
- prior scale families:
  - conservative shrinkage
  - medium shrinkage
  - weaker shrinkage
- sampler settings:
  - `target_accept` in `{0.9, 0.95}`
  - `chains=4`
  - warmup/draw budgets in `{1000/1000, 1500/1500}`
- structural variants:
  - base hierarchical model
  - base hierarchical model plus team-season effects
  - base hierarchical model plus team-season effects and rookie priors

Evaluation criteria must include:

- convergence quality: R-hat, ESS, divergences, treedepth pathologies
- predictive quality: holdout MAE/RMSE, ranking quality, calibration
- operational quality: runtime and artifact stability

Rationale:

- Sampled Bayes is sensitive to parameterization and inference settings.
- A single arbitrary configuration can look "more Bayesian" while actually being less stable, less calibrated, or too slow for the supported workflow.
- A bounded exploration pass gives the sampled estimator a fair comparison against the empirical-Bayes baseline.

Alternatives considered:

- **Pick one PyMC configuration and hardcode it**: too brittle and too easy to misread as validated.
- **Open-ended tuning without limits**: too expensive and too vague for a production-facing repo change.

### Decision: Team-season and rookie priors are required production features; conference effects are not

The supported model stack will add:

- team-season effects
- rookie draft-capital priors
- rookie combine priors
- rookie depth-chart priors

The first production cut of contextual structure beyond player and position will retain:

- team-season effects
- team-change indicators
- depth-chart or roster-competition context
- schedule-strength aggregates only if they improve holdout results

College-conference effects remain excluded. Full opponent-by-opponent modeling is also excluded from the first production cut unless later evidence justifies it.

Rationale:

- The current data backend can realistically support draft picks, combine results, players, rosters, depth charts, injuries, and schedules via `nflreadpy`.
- Team-season context is more plausibly predictive for fantasy role/value than NFL conference labels.
- College conference is a secondary proxy and should not outrank draft capital or landing spot in the first production redesign.

Alternatives considered:

- **Conference effects now**: additional data work with weak expected signal.
- **Rookie fallback to position mean only**: statistically thin and dashboard-hostile.

### Decision: The hybrid MC/random-forest lane SHALL be removed rather than hidden

The following path is retired and removed outright:

- `src/ffbayes/analysis/hybrid_mc_bayesian.py`
- hybrid integration/risk-ranking consumers
- hybrid draft strategy and hybrid Excel generation surfaces
- hybrid-specific CLI aliases and docs
- hybrid runtime artifacts and their evidence claims

If any retained module still serves a purpose, it must be rewritten against the primary player-forecast contract rather than preserved as a hybrid compatibility surface.

Rationale:

- The hybrid lane is not statistically aligned with the supported draft engine.
- The random-forest uncertainty score is not the same object as a player posterior.
- Keeping it around violates the repo rule against deprecated fallback logic.

Alternatives considered:

- **Mark as deprecated but keep runnable**: rejected by repo policy and would preserve confusion.
- **Hide behind an experimental flag**: still a fallback path unless fully isolated from production contracts.

### Decision: Cleanup and documentation alignment SHALL precede sampled-Bayes expansion

The next implementation tranche for this change will prioritize finishing hybrid retirement, output-structure cleanup, dashboard/evidence simplification, and documentation alignment before expanding the sampled hierarchical evaluation lane.

Rationale:

- The repo still carries legacy hybrid references outside the main supported path.
- The dashboard and docs are the operator-facing product contract and should be stabilized before another modeling lane is introduced.
- Adding sampled-Bayes work before cleanup is finished would raise the risk of another partially integrated side branch.

Alternatives considered:

- **Start sampled-Bayes evaluation immediately**: rejected because it compounds modeling work on top of unresolved cleanup and documentation drift.

### Decision: Validation is a first-class artifact, not an optional analysis

The supported player-model path will emit deterministic validation artifacts for:

- rolling holdout MAE/RMSE
- ranking correlation
- top-k draft-candidate quality
- interval coverage / calibration
- rookie/veteran slices
- position slices
- scoring-rate and availability component diagnostics

These artifacts feed the dashboard evidence surface and are required for supported dashboard generation.

Rationale:

- Internal validation should be visible, explicit, and testable.
- The current `bayesian_vor_comparison.py` already provides strong pieces of this and should be promoted rather than sidelined.
- Evidence claims in the dashboard should come from the production model stack, not from unrelated analyses.

Alternatives considered:

- **Keep validation in ad hoc research scripts only**: not enough for a production-facing dashboard.
- **Use only policy backtests**: incomplete without player-forecast calibration and slice reporting.

### Decision: Dashboard incorporation SHALL be explicit from runtime model outputs to rendered UI

The change will treat the dashboard as a first-class implementation target rather than a passive downstream consumer. The supported path from model redesign to working UI will be:

1. player-model contract emits renamed or newly structured posterior fields
2. `draft_decision_system.py` maps those fields into board rows, recommendation lanes, and inspector semantics
3. payload builders map supported-model diagnostics and uncertainty summaries into `decision_evidence`, glossary entries, and normalized `war_room_visuals`
4. canonical runtime HTML and JSON are regenerated from that contract
5. repo-local `dashboard/` and staged `site/` copies are derived from the canonical runtime pair
6. the fixture-driven smoke path verifies the UI behaves correctly with the redesigned semantics

Required dashboard-facing outputs include:

- recommendation row values derived from the supported player-model contract
- inspector fields that explain season-total value, rate vs availability risk when available, rookie prior context when relevant, and supported-model provenance
- glossary terminology aligned with the redesigned model semantics
- evidence panel content sourced from supported validation artifacts
- normalized `war_room_visuals` semantics for timing and scarcity views that remain stable even if internal model formulas change

The main board should remain lean. Rate-versus-availability decomposition belongs in the selected-player inspector and evidence-linked surfaces, not in the primary board columns.
The decision-evidence surface should also remain compact: one summary block, one compact validation table, and deeper expandable detail when needed, rather than a dense stack of primary tables in the first visible evidence surface.

Rationale:

- Without an explicit dashboard mapping plan, the repo could complete a statistically cleaner model redesign while still shipping stale or misleading UI semantics.
- The live dashboard is the actual operator product; the spec should therefore carry the design through to the rendered artifact surfaces.

Alternatives considered:

- **Leave dashboard integration implicit in implementation tasks**: too easy for payload/UI drift.
- **Treat the dashboard as purely presentational**: incorrect, because evidence, inspector semantics, and visual explanations are part of the product contract.

### Decision: Output structure SHALL remain minimal, explicit, and authority-scoped

The redesign will not introduce new ad hoc output families. All supported outputs must fit into a small, declared structure with stable purpose and authority:

```text
inputs/
  raw/
    season_datasets/
    manifests/
  processed/
    combined_datasets/
    snake_draft_datasets/
    unified_dataset/

seasons/
  <year>/
    draft_strategy/
      draft_board_<year>.xlsx                 # canonical workbook
      draft_board_<year>.html                 # canonical runtime dashboard HTML
      dashboard_payload_<year>.json           # canonical runtime dashboard payload
      draft_decision_backtest_<year_range>.json
      model_outputs/
        player_forecast/
          player_forecast_<year>.json         # canonical player forecast export
          player_forecast_validation_<range>.json
          player_forecast_diagnostics_<year>.json
      finalized_drafts/
    vor_strategy/
    diagnostics/
      validation/
        player_forecast_stress_fixture_<year>.json
        player_forecast_stress_fixture_summary_<year>.json
```

Derived surfaces remain:

```text
dashboard/
  index.html
  dashboard_payload.json

site/
  index.html
  dashboard_payload.json
  publish_provenance.json
```

Rules:

- The canonical working input tree is local runtime `inputs/`, not runtime `data/`.
- Stable cloud mirrors remain under cloud `data/` and dated publish snapshots under cloud `Analysis/<date>/`.
- The canonical dashboard pair remains under `seasons/<year>/draft_strategy/`.
- Player-model outputs live under `seasons/<year>/draft_strategy/model_outputs/player_forecast/`.
- Stress-fixture and verification summaries live under `seasons/<year>/diagnostics/validation/`.
- Repo-local `dashboard/` and repo `site/` remain shallow derived copies only.
- The redesign MUST remove the redundant `pre_draft/` and `artifacts/` wrappers instead of recreating them under a new name.
- The redesign MUST remove the legacy runtime `datasets/` tree instead of preserving it beside `inputs/`.
- The redesign MUST NOT create parallel `hybrid_*`, `bayesian_hierarchical_*`, `experimental_*`, duplicate dashboard artifact families, or parallel runtime trees such as both `inputs/` and `data/` or both `seasons/` and `runs/`.

Rationale:

- Operators need one obvious place to find the real dashboard and supporting model outputs.
- Maintainers need deterministic locations for validation, publication, and regression checks.
- Explicit output boundaries prevent the redesign from replacing one kind of conceptual sprawl with filesystem sprawl.
- Renaming local runtime `data/` to `inputs/` avoids collision with the cloud mirror contract, where `data/` is already the stable published backup surface.
- Renaming `runs/` to `seasons/` better reflects that these outputs are season-scoped working products, not arbitrary execution runs.

Alternatives considered:

- **Let each module emit its own convenient files**: produces output drift and duplicate truth surfaces.
- **Keep everything in one flat directory**: easier to generate, much harder to understand and test.
- **Keep `runs/` but flatten beneath it**: better than the current shape, but still uses execution-oriented naming for season-scoped outputs.
- **Keep local runtime `data/` and cloud `data/` with different meanings**: workable, but unnecessarily confusing.

### Decision: Local runtime inputs remain authoritative; cloud data remains a mirrored backup and publish surface

The local runtime input tree remains the working source of truth for collection, preprocessing, and dashboard generation. Cloud storage remains a mirrored backup and publish-oriented surface under the existing `analysis_registry.yaml` contract:

- local runtime working inputs: `inputs/`
- cloud stable mirror: `data/`
- cloud dated publish snapshots: `Analysis/<date>/`

Rationale:

- This matches the workspace-governor assessment outcome: the repo is already compliant, no workspace move is needed, and cloud publishing is a reviewable mirror step rather than the live execution root.
- Local runtime paths are faster and operationally safer for iterative pipeline work than using a synced cloud folder as the direct working tree.
- The naming split makes the authority boundary explicit instead of relying on tribal knowledge.

Alternatives considered:

- **Make cloud `data/` the only authoritative input tree**: possible for one maintainer, but introduces sync semantics into the core pipeline contract and weakens local operational safety.
- **Mirror everything indiscriminately to cloud**: conflicts with the existing publish denylist and increases clutter.

### Decision: The change SHALL end with a code-and-doc accuracy sweep

The final stage of implementation must include an explicit sweep for documentation accuracy inside the code and around it. This includes:

- README and durable `docs/`
- public-facing glossary/evidence/help text in the dashboard
- module and function docstrings for changed public interfaces
- concise orienting comments for non-obvious model logic, path migration logic, and dashboard mapping logic

The purpose is not to maximize comments. The purpose is to ensure that the implemented system is legible and that changed interfaces do not drift away from their documentation.

Rationale:

- This change touches statistical semantics, runtime paths, and dashboard behavior at the same time.
- Those are exactly the kinds of changes that become confusing later if code and docs are only partially updated.
- A final sweep is cheaper than letting stale docstrings and stale dashboard copy become the next source of truth conflict.

Alternatives considered:

- **Rely on README updates alone**: insufficient for changed code interfaces and internal model semantics.
- **Add comments everywhere**: noisy and contrary to the repo’s editing guidance.

### Decision: Add a small, interpretable pre-draft stress fixture

A dedicated fixture-driven test harness will exercise the full supported path with a small dataset that still covers important edge cases. The fixture should be small enough for CI/local iteration while still interpretable by humans.

Required fixture characteristics:

- at least three seasons so rolling holdout logic can execute
- players across QB/RB/WR/TE
- at least one rookie with draft-capital/combine/depth-chart context
- at least one veteran with a team change
- at least one player with injury/availability instability
- at least one player with missing or weak market context to test robust feature handling
- enough data to produce a draft board, evidence payload, and a meaningful dashboard smoke run

Example fixture cohort:

- stable QB baseline with durable team context
- veteran alpha WR comparison point with stable role
- team-changing RB or similar veteran role-change example in 2026
- first-round rookie WR entering a crowded depth chart
- availability-risk TE or similar missed-games example

Alternatives considered:

- **Use the full real pipeline for all stress tests**: too slow and too noisy for routine iteration.
- **Use tiny unit-only fixtures**: too narrow to test pipeline/evidence/dashboard integration.

## Risks / Trade-offs

- **[Estimand redesign touches many modules]** → Mitigation: keep canonical artifact names stable while changing content and provenance incrementally behind one change.
- **[Removing hybrid modules may break hidden consumers]** → Mitigation: inventory imports first, replace only with the supported player-forecast contract, and add explicit removal tests for retired CLI/runtime/doc references.
- **[Sampled hierarchical prototype may underperform or slow the pipeline]** → Mitigation: keep empirical Bayes as production baseline and gate any promotion on holdout metrics plus runtime constraints.
- **[Rookie priors may depend on new data joins and name-resolution edge cases]** → Mitigation: standardize ID joins through the existing data pipeline and include fixture examples that force rookie edge cases.
- **[Dashboard evidence could drift from emitted validation artifacts]** → Mitigation: treat evidence as a structured contract and test it in payload, Pages staging, and smoke coverage.
- **[Stress fixture can become unrepresentative if too small]** → Mitigation: design the fixture around explicit edge-case coverage rather than random subsampling and document what it is and is not proving.

## Migration Plan

1. Inventory active and legacy model consumers in CLI, pipeline config, analysis modules, dashboard payload generation, and docs.
2. Redefine the supported player-forecast contract and update the empirical-Bayes implementation to the season-total/two-part design.
3. Add rookie/team-season/context feature ingestion through the existing unified data path.
4. Map the redesigned player-model contract into draft board rows, inspector fields, glossary content, normalized `war_room_visuals`, runtime payloads, and canonical dashboard HTML.
5. Migrate local runtime inputs from `data/` to `inputs/`, migrate season outputs from `runs/` to `seasons/`, remove `pre_draft/` and `artifacts/`, and delete the legacy runtime `datasets/` tree.
6. Create the explicit runtime and diagnostics output structure for player forecast exports, validation artifacts, and stress-fixture summaries without introducing duplicate truth surfaces.
7. Wire validation artifacts into runtime outputs and decision evidence.
8. Add the small fixture-driven stress harness and keep `node tests/dashboard_smoke.mjs` in the supported verification path.
9. Implement the sampled hierarchical evaluation lane, run the bounded training/inference exploration, and keep empirical Bayes as the production default unless the sampled path satisfies the promotion criteria.
10. Remove hybrid modules, hybrid artifact generation, hybrid CLI exposure, and hybrid documentation in the same implementation window.
11. If the sampled hierarchical path does not satisfy the declared promotion criteria, remove it from the supported workflow before finalizing the change rather than leaving it as a dormant supported branch.
12. Run final specialist review lanes:
   - implementation validity
   - documentation consistency
   - robustness/stress coverage

Rollback strategy:

- There is no long-lived fallback hybrid path.
- If the redesigned player model is not implementation-ready, the change should not ship.
- During implementation, retain work on a branch/change only until the supported single-path stack is complete.

## Resolved Decisions

- Season-total posterior summaries will be composed through posterior predictive simulation of scoring-rate and availability components.
- The sampled hierarchical estimator will be implemented in this change as an evaluation lane against the same production contract.
- Empirical Bayes remains the production default unless the sampled hierarchical path satisfies explicit promotion criteria on holdout quality, calibration, and operational stability.
- If the sampled hierarchical path does not satisfy those promotion criteria, it will not remain in the supported workflow and should be removed rather than left behind as inactive supported code.
- The first production cut of context beyond player and position includes team-season effects, team-change indicators, depth-chart or roster-competition context, and schedule-strength aggregates only if validated.
- Rate-versus-availability decomposition will surface in the selected-player inspector and evidence-linked dashboard surfaces, not as extra clutter in the main board columns.
- The bounded sampled-Bayes search space remains intentionally small: parameterization choice, prior-scale family, `target_accept`, draw budget, and a short list of structural variants.
