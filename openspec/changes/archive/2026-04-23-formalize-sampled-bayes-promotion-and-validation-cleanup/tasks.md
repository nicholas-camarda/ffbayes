## 1. Specification closeout
- [x] 1.1 Keep the canonical third-outcome artifact key as `keep_empirical_bayes_with_sampled_prior_followup` and use shorter operator-facing wording only in docs or explanatory prose.
- [x] 1.2 Keep sampled-versus-empirical comparison as an internal diagnostics and explicit-tooling workflow rather than a new default supported CLI surface.
- [x] 1.3 Keep sampled-comparison outcomes out of the live dashboard until a formal promotion outcome exists.

## 2. Validation-metric semantics cleanup
- [x] 2.1 Audit every supported rank-correlation and ranking-quality metric emitted by `src/ffbayes/analysis/bayesian_vor_comparison.py` and related consumers.
- [x] 2.2 Replace `np.nan_to_num(..., nan=0.0)` or equivalent silent coercions that currently collapse undefined rank metrics into neutral numeric values.
- [x] 2.3 Introduce one normalized unavailable-metric representation for persisted validation artifacts, including an explicit reason field such as `constant_input` or `insufficient_variation` when determinable.
- [x] 2.4 Ensure persisted validation outputs distinguish:
  - estimated near-zero association
  - unavailable or undefined association
  - omitted slices due to insufficient support
- [x] 2.5 Update evidence-facing payload builders and renderers so unavailable validation metrics surface as `n/a`, `not estimable`, or equivalent direct wording rather than numeric `0.00`.
- [x] 2.6 Add regression tests that fail if constant-input slices are silently converted back to numeric `0.0`.
- [x] 2.7 Add regression tests that assert the unavailable-state reason is preserved in structured validation artifacts.

## 3. Advanced-stat warning cleanup
- [x] 3.1 Audit each warning-producing grouped transform in `src/ffbayes/analysis/advanced_stats_calculator.py`, including the current boom or bust, floor or ceiling, early versus late, WR big-play dependency, TE reliability, recent-form, season-trend, and consistency-over-time calculations.
- [x] 3.2 Replace each deprecated `groupby.apply(...)` path with explicit grouped aggregation logic on the actual columns or grouped frames needed for the calculation.
- [x] 3.3 Preserve the current statistical intent of each advanced-stat transform while removing reliance on deprecated grouping-column semantics.
- [x] 3.4 Do not implement warning suppression, compatibility shims, or fallback branches as the solution.
- [x] 3.5 Add targeted regression tests that execute the touched advanced-stat paths and fail if the deprecated pandas grouping warning reappears.
- [x] 3.6 Where a cleaned grouped calculation changes output shape or edge-case handling, document the exact intended semantics in code comments or docstrings near the implementation rather than leaving the grouped logic implicit.

## 4. Sampled-versus-empirical governance workflow
- [x] 4.1 Audit the current bounded search and selector behavior in `src/ffbayes/analysis/sampled_player_model.py`, including search-space construction, per-configuration evaluation, rejection gates, and final outcome selection.
- [x] 4.2 Extend the sampled evaluation lane so it always produces one persisted comparison artifact family rather than only in-memory or console-level decisions.
- [x] 4.3 Persist, at minimum:
  - declared search space
  - per-configuration results
  - rejected configurations and reasons
  - convergence diagnostics
  - runtime costs
  - overall forecast metrics
  - calibration metrics
  - ranking metrics
  - rookie versus veteran or sparse-history slices
  - final decision outcome
  - decision rationale
- [x] 4.4 Ensure the comparison is run against the same season-total forecast contract, holdout framing, and dashboard-relevant semantics as the empirical-Bayes production baseline.
- [x] 4.5 Extend the selector so it supports all three outcomes:
  - `promote_sampled_bayes`
  - `keep_empirical_bayes`
  - `keep_empirical_bayes_with_sampled_prior_followup`
- [x] 4.6 Encode the rookie or sparse-history follow-up case explicitly instead of flattening it into binary pass or fail logic.
- [x] 4.7 Keep empirical Bayes as the production estimator unless the persisted comparison artifact explicitly records a promotion outcome.
- [x] 4.8 If sampled Bayes fails to win overall, ensure the supported CLI, dashboard, and docs do not treat it as production merely because evaluation outputs exist.
- [x] 4.9 If sampled Bayes improves rookie or sparse-history behavior without winning overall, persist enough detail to inform future prior-design work without silently keeping dormant production wiring around.
- [x] 4.10 Keep sampled-comparison outputs inside canonical `seasons/<year>/...` diagnostics or model-output paths and reject any new ad hoc runtime artifact family.
- [x] 4.11 Add regression tests that fail if sampled evaluation can silently promote itself without a persisted winning artifact.
- [x] 4.12 Add regression tests that fail if rejected sampled configurations lose their structured rejection reasons.

## 5. Dashboard evidence and operator-surface updates
- [x] 5.1 Update the structured decision-evidence payload so non-estimable validation metrics remain explicit unavailable states all the way through dashboard serialization.
- [x] 5.2 Update dashboard rendering so unavailable validation metrics display as `n/a`, `not estimable`, or equivalent explicit language rather than fabricated numeric neutrality.
- [x] 5.3 Ensure dashboard evidence copy distinguishes:
  - poor measured performance
  - missing or degraded evidence
  - non-estimable validation slices
- [x] 5.4 Ensure the live dashboard payload and UI do not surface sampled-comparison outcomes while sampled Bayes remains unpromoted.
- [x] 5.5 Add or update tests so runtime payloads, staged `site/` payloads, and any evidence-facing HTML remain aligned on unavailable-metric semantics and on the absence of unpromoted sampled-comparison content.

## 6. Documentation and truth-surface updates
- [x] 6.1 Update the authoritative technical guide to describe:
  - the warning cleanup as implementation cleanup rather than a model redesign
  - non-estimable validation metrics as unavailable states rather than zero-correlation findings
  - the sampled governance workflow and its trinary outcomes
  - the empirical-Bayes production-default rule
- [x] 6.2 Update operator documentation so `n/a` or `not estimable` is explained directly and sampled diagnostics are not mistaken for live production behavior.
- [x] 6.3 Update layperson-facing docs so they explain unavailable validation metrics in plain language without implying neutral performance.
- [x] 6.4 Update any glossary, metric reference, or evidence-panel copy that currently suggests undefined correlations are numeric zeroes.
- [x] 6.5 Ensure README and `docs/` state only the current behavior of the repo and do not imply sampled production promotion unless a promotion artifact actually exists.
- [x] 6.6 Add or update documentation regression checks where the repo already enforces documentation contracts for dashboard semantics and supported workflow descriptions.

## 7. Verification and review
- [x] 7.1 Run targeted Python tests covering:
  - `advanced_stats_calculator.py`
  - `bayesian_vor_comparison.py`
  - `sampled_player_model.py`
  - dashboard evidence consumers
- [x] 7.2 Run the relevant dashboard smoke or payload-alignment tests if decision-evidence rendering changes.
- [x] 7.3 Verify that the touched advanced-stat code no longer emits the pandas grouping deprecation warning under supported test paths.
- [x] 7.4 Verify that constant-input validation slices now yield explicit unavailable metrics and reasons rather than numeric `0.0`.
- [x] 7.5 Verify that sampled-versus-empirical comparison artifacts are persisted, auditable, and constrained to canonical output paths.
- [x] 7.6 Verify that empirical Bayes still presents as the production estimator unless a persisted promotion artifact explicitly says otherwise.
- [x] 7.7 Perform a final code-and-doc sweep on all touched modules so docstrings, comments, docs, and evidence copy describe the implemented semantics accurately and do not reintroduce historical or migration framing.

## 8. Full-data interpretation and repository decision
- [x] 8.1 Run the supported production-mode workflow end to end on the complete available data surface rather than only on fixtures or reduced evaluation slices.
- [x] 8.2 Confirm that the production-mode run produced the expected full-data artifacts needed to interpret warning cleanup, validation semantics, and sampled-governance outcomes.
- [x] 8.3 Write one formal interpretation report that summarizes:
  - the full-data production-mode run
  - warning-cleanup results
  - validation-semantic results
  - sampled-versus-empirical comparison results
  - statistical interpretation
  - operational interpretation
- [x] 8.4 Require the report to state one explicit move-forward decision:
  - promote sampled Bayes into the supported production path
  - keep empirical Bayes as production because sampled evaluation remains incomplete
  - keep empirical Bayes as production and retain sampled prior follow-up only
  - keep empirical Bayes as production and retire the sampled workflow from the supported path
- [x] 8.5 Ensure the final implementation summary and any updated docs reflect that explicit decision rather than leaving the repo in an unresolved evaluation state.
