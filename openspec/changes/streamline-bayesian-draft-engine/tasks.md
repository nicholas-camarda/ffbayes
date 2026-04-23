## 1. Inventory and target contract

- [x] 1.1 Inventory every supported and legacy consumer of the current player-model, hybrid-analysis, dashboard-evidence, and draft-strategy paths under `src/ffbayes`, `config/`, `docs/`, and `tests/`.
- [x] 1.2 Define the canonical player-forecast contract around season-total fantasy value and record the exact emitted fields needed by `draft_decision_system.py`, dashboard payload generation, and evidence surfaces.
- [x] 1.3 Identify every location where the current code or docs describe the active forecast target imprecisely or inconsistently and map each one to a required code or documentation update.
- [x] 1.4 Record the resolved composition decision in code-facing notes and docs: season-total posteriors are derived through posterior predictive simulation of rate and availability components.
- [x] 1.5 Inventory every consumer of runtime `data/`, runtime `datasets/`, `runs/<year>/pre_draft/`, and `artifacts/` paths so the path migration can be completed in one coherent cutover.
- [x] 1.6 Record the resolved model-governance decisions in implementation notes and docs: empirical Bayes remains the production default, sampled hierarchical Bayes is an evaluation lane on the same contract, and college-conference effects remain out of scope.

## 2. Primary player-model redesign

Implementation order note: finish hybrid cleanup, dashboard/evidence simplification, and docs alignment before expanding the sampled hierarchical evaluation lane in tasks `2.7`-`2.11`.

- [x] 2.1 Refactor `src/ffbayes/analysis/bayesian_player_model.py` so the supported production contract is driven by season-total fantasy value rather than a loosely described per-game season aggregate.
- [x] 2.2 Implement the two-part player forecast structure for scoring-rate and availability components and expose an auditable path from those components to season-total posterior summaries.
- [x] 2.3 Preserve empirical Bayes as the default production estimator while restructuring the code so a sampled hierarchical estimator can target the same contract without creating a second production path.
- [x] 2.4 Add team-season effects to the supported player forecast stack and ensure those effects flow through to emitted posterior outputs.
- [x] 2.5 Add explicit rookie priors driven by draft capital, combine data, and depth-chart context rather than position-only fallback shrinkage.
- [x] 2.6 Restrict the first production cut of context beyond player and position to team-season effects, team-change indicators, depth-chart or roster-competition context, and schedule-strength aggregates only if they improve holdout performance against the supported contract.
- [x] 2.7 If a sampled hierarchical estimator is included in this change, implement it as an evaluation lane against the same contract and emit convergence and posterior-predictive diagnostics without promoting it automatically to production.
- [x] 2.8 If a sampled hierarchical estimator is included in this change, implement a bounded exploration workflow for training and inference settings and record the compared configurations, convergence diagnostics, predictive metrics, and runtime costs.
- [x] 2.9 Derive season-total posterior summaries through posterior predictive simulation of scoring-rate and availability components rather than a purely analytic shortcut.
- [x] 2.10 Explicitly exclude college-conference effects and full opponent-by-opponent modeling from the first production implementation unless later evidence justifies a follow-up change.
- [x] 2.11 If the sampled hierarchical estimator does not satisfy the declared promotion criteria, remove it from the supported workflow before completion rather than leaving a dormant supported branch behind.

## 3. Data ingestion and feature plumbing

- [x] 3.1 Extend the current `nflreadpy`-backed ingestion path to collect the rookie-prior inputs required by the design, including draft picks, combine data, players, rosters, and depth-chart context.
- [x] 3.2 Normalize and join the new rookie/context fields through the existing unified dataset flow without introducing standalone path logic outside `src/ffbayes/utils/path_constants.py`.
- [x] 3.3 Add deterministic handling for missing rookie/context fields so the supported player-model contract remains explicit about missingness without reintroducing fallback model lanes.
- [x] 3.4 Update any name-resolution or ID-join logic needed so rookies and team-changing veterans align correctly across the new data sources.
- [x] 3.5 Migrate canonical runtime working-input paths from `data/raw` and `data/processed` to `inputs/raw` and `inputs/processed` while preserving the cloud mirror contract under cloud `data/`.

## 4. Validation artifacts and dashboard evidence

- [x] 4.1 Promote `src/ffbayes/analysis/bayesian_vor_comparison.py` or its successor into the supported player-model validation path for rolling holdout accuracy, ranking quality, and calibration artifacts.
- [x] 4.2 Add validation slices for rookies vs veterans, QB/RB/WR/TE, and separate scoring-rate vs availability diagnostics where the model emits those components.
- [x] 4.3 Wire the supported validation artifacts into the runtime draft artifact set so dashboard evidence derives from the production player-model stack and not from retired hybrid outputs.
- [x] 4.4 Update `src/ffbayes/draft_strategy/draft_decision_system.py` and related payload builders so the evidence panel exposes supported-model provenance, internal-validation scope, calibration context, and degraded states deterministically.
- [x] 4.5 Ensure the staged `site/` payload and repo-local `dashboard/` shortcut preserve the same decision-evidence semantics as the canonical runtime payload.

## 5. Dashboard integration

- [x] 5.1 Map the redesigned player-model outputs into the draft board row schema consumed by `src/ffbayes/draft_strategy/draft_decision_system.py`.
- [x] 5.2 Update the selected-player inspector contract so it can explain season-total value, supported-model provenance, rookie-prior context when relevant, and rate-vs-availability semantics, while keeping that decomposition out of the main board columns.
- [x] 5.3 Update glossary and dashboard-facing labels so the UI names and explains the redesigned model outputs without hybrid or stale terminology.
- [x] 5.4 Update normalized `war_room_visuals` payload generation so timing, scarcity, and comparative visual semantics are derived from the redesigned supported model contract rather than retired hybrid assumptions.
- [x] 5.5 Regenerate and verify canonical runtime dashboard HTML and JSON so the live UI reflects the new model semantics end to end.
- [x] 5.7 Simplify the decision-evidence UI so the first visible evidence surface stays compact: one summary block, one compact validation table, and deeper expandable detail instead of a dense stack of peer tables.
- [x] 5.6 Verify that repo-local `dashboard/` and staged `site/` remain narrow derived surfaces of the canonical runtime dashboard after the redesign.

## 6. Output structure and artifact discipline

- [x] 6.1 Migrate season-scoped outputs from `runs/<year>/pre_draft/artifacts/...` to `seasons/<year>/...` and remove the redundant `pre_draft/` and `artifacts/` path layers.
- [x] 6.2 Create the exact runtime output structure for canonical player forecast exports under `seasons/<year>/draft_strategy/model_outputs/player_forecast/`.
- [x] 6.3 Create the exact diagnostics output structure for stress-fixture and validation summaries under `seasons/<year>/diagnostics/validation/`.
- [x] 6.4 Keep the canonical dashboard pair under `seasons/<year>/draft_strategy/` and verify no duplicate dashboard truth surface is introduced.
- [x] 6.5 Delete the legacy runtime `datasets/` tree and ensure no supported code, docs, or tests continue to rely on it.
- [x] 6.6 Ensure repo-local `dashboard/` and staged `site/` remain shallow derived copies only and do not accumulate diagnostics, historical clutter, or parallel payload families.
- [x] 6.7 Add tests that fail if ad hoc artifact families such as `hybrid_*`, `experimental_*`, duplicate dashboard payloads, duplicate canonical player forecast outputs, or parallel `data/`/`inputs/` or `runs/`/`seasons/` runtime trees are introduced into the supported path.

## 7. Hybrid retirement and pipeline cleanup

- [x] 7.1 Remove `src/ffbayes/analysis/hybrid_mc_bayesian.py` from the supported `pre_draft` pipeline configuration and replace its role with the supported player-model path.
- [x] 7.2 Remove hybrid-specific integration, risk-ranking, hybrid draft-strategy, and hybrid Excel surfaces that exist only to support the retired hybrid lane.
- [x] 7.3 Remove hybrid-facing command exposure from `src/ffbayes/cli.py`, `pyproject.toml`, and any other published entrypoint/help surface that still presents the retired hybrid lane as supported.
- [x] 7.4 Remove hybrid runtime artifact expectations from any artifact builders, comparison helpers, or dashboard/publish flows that still reference them.
- [x] 7.5 Add regression checks that fail if the retired hybrid lane is reintroduced into pipeline config, CLI help, runtime artifact contracts, or supported docs.

## 8. Stress fixture and automated verification

- [x] 8.1 Create a small, interpretable pre-draft stress fixture with at least three seasons, QB/RB/WR/TE coverage, a rookie with draft-capital/combine/depth-chart context, a team-changing veteran, and an availability-risk example.
- [x] 8.2 Build a fixture-driven pipeline test path that exercises preprocessing, unified dataset creation, player forecasting, draft-strategy artifact generation, dashboard payload generation, canonical dashboard HTML generation, and evidence payload emission without requiring a full production runtime.
- [x] 8.3 Add or update pytest coverage for the redesigned player-model contract, rookie priors, team-season effects, validation artifacts, dashboard payload mapping, output-structure rules, and hybrid-retirement guards.
- [x] 8.4 Extend dashboard-facing tests so the fixture artifacts drive `tests/dashboard_smoke.mjs`, `tests/test_publish_pages.py`, and any relevant payload/HTML contract tests with interpretable expected results after the `inputs/` and `seasons/` path migration.
- [x] 8.5 Add strict JSON and artifact-lineage assertions for canonical runtime payloads, canonical runtime HTML, repo-local dashboard shortcuts, staged `site/` copies, and player-forecast output directories produced by the fixture-driven path.
- [x] 8.6 Add focused tests that prove the evidence panel, inspector, glossary, and `war_room_visuals` surface supported-model provenance, internal-validation scope, degraded-state handling, and the absence of retired hybrid references.
- [x] 8.7 If the sampled Bayesian path is implemented, add deterministic checks that the chosen sampled-model configuration comes from the bounded exploration results and that convergence failures or pathological inference settings fail closed.
- [x] 8.8 Add regression checks that empirical Bayes remains the production dashboard path unless the sampled hierarchical estimator satisfies the explicit promotion criteria declared in the implementation.
- [x] 8.9 Add regression checks that a non-winning sampled hierarchical estimator cannot remain wired into supported CLI, pipeline, dashboard, or documentation surfaces after the change is finalized.

## 9. Documentation and operator guidance

- [x] 9.1 Update `README.md` so the supported `ffbayes pre-draft` workflow, CLI surface, artifact descriptions, dashboard generation path, and exact output structure reflect the single production model stack, the `inputs/` and `seasons/` runtime layout, and the removal of hybrid analysis from the supported path.
- [x] 9.2 Update the technical guide under `docs/` so it truthfully describes the new forecast target, the two-part player-model structure, team-season effects, rookie priors, validation contract, interval semantics, how those outputs appear in the dashboard, and exactly where each supported artifact lives under `inputs/` and `seasons/`.
- [x] 9.3 Update operator-facing and layperson-facing docs so they explain the supported dashboard behavior, evidence interpretation, inspector semantics, concrete examples such as a team-changing veteran and a draft-capital-driven rookie prior, and which output paths are authoritative versus derived.
- [x] 9.4 Update documentation/path lineage references so local runtime inputs, season outputs, canonical dashboard HTML and JSON, player-forecast outputs, repo-local dashboard copies, staged `site/` outputs, and cloud `data/` / `Analysis/` mirrors remain clearly distinguished after the path migration.
- [x] 9.5 Add documentation drift checks or update existing contracts/tests so retired hybrid-path guidance, stale dashboard semantics, stale `data/` or `runs/` runtime path references, or unapproved output families cannot quietly re-enter the guide suite.
- [x] 9.6 Perform a final code-facing documentation sweep so changed public modules, functions, and CLI/path interfaces have accurate docstrings and any non-obvious model, migration, or dashboard-mapping logic has concise orienting comments where needed.

## 10. Final verification and specialist review

- [x] 10.1 Run the full local verification set for touched surfaces, including targeted pytest suites, fixture-driven stress tests, dashboard smoke coverage, publication/staging checks, Ruff, and any relevant type checks.
- [x] 10.2 Run a final implementation-auditor subagent review focused on whether the shipped code actually matches the supported player-model, dashboard-integration, output-structure, hybrid-retirement, and validation requirements.
- [x] 10.3 Run a final documentation-wizard subagent review focused on whether README, `docs/`, changed docstrings, dashboard semantics, CLI surface, artifact paths, and explicit authority levels match the implemented system.
- [x] 10.4 Run a final robustness-test-designer or equivalent specialist review focused on whether the fixture and regression tests are strong enough to detect reintroduction of the retired hybrid path, drift in the dashboard/evidence/visualization contract, and output-structure sprawl.
- [x] 10.5 Resolve any findings from the final specialist review pass and rerun the relevant verification steps before considering the change implementation-ready.
