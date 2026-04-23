## Context

`streamline-bayesian-draft-engine` established the production contract for the draft engine: empirical Bayes is the production estimator, the forecast target is season-total fantasy value, and the sampled hierarchical estimator exists only as an evaluation lane unless it earns promotion. That major cleanup left three follow-on issues that are now tightly coupled but not identical.

First, several advanced-stat transforms in `src/ffbayes/analysis/advanced_stats_calculator.py` still rely on `groupby.apply(...)` patterns that trigger pandas deprecation warnings. Those warnings are not merely cosmetic. They indicate that the current implementations depend on grouping-column behavior that pandas will change, which makes future runs noisier and increases the risk of silent aggregation drift.

Related pipeline cleanup is also still needed at the source of two non-fatal orchestration warnings: the current-player nflverse adapter still assumes `recent_team` even though the installed `nflreadpy` player master surface now exposes `latest_team`, and the `ffbayes.draft_strategy` package currently preloads executable modules during package import, which is enough to trigger `runpy` warnings when those same modules are later executed via `python -m` during the pre-draft pipeline.

Second, the validation layer in `src/ffbayes/analysis/bayesian_vor_comparison.py` currently coerces undefined rank-correlation values to `0.0` through `np.nan_to_num(...)`. That collapses two materially different states:

- estimated zero monotonic association
- no estimable rank correlation because one side is constant or the slice lacks variation

Those states must be distinguished in both persisted validation artifacts and dashboard evidence, otherwise the repo presents undefined validation slices as if they were legitimate neutral estimates.

Third, the sampled hierarchical estimator in `src/ffbayes/analysis/sampled_player_model.py` is technically implemented as a bounded evaluation lane, but it is not yet embedded in a decision-grade comparison workflow. The current code can search configurations and choose a preferred sampled result, but the repo still lacks a formal artifact contract that records:

- the searched configurations
- rejected configurations and reasons
- convergence diagnostics
- calibration and ranking comparisons
- rookie-specific versus overall outcomes
- a final promotion decision that can be audited later

This means the repo currently has evaluation code but not a complete governance workflow for deciding whether sampled Bayes ever supersedes empirical Bayes.

## Goals / Non-Goals

**Goals:**
- Remove the remaining pandas deprecation warnings from the advanced-stat pipeline by replacing deprecated aggregation idioms with deterministic, explicit grouped calculations.
- Redefine validation semantics so non-estimable rank metrics are represented explicitly as unavailable or `null`, with structured reasons where needed, instead of silently converted to `0.0`.
- Preserve clean distinctions between:
  - warning cleanup
  - validation-metric semantics
  - sampled-model promotion governance
- Formalize a repeatable sampled-versus-empirical comparison workflow that produces persisted decision artifacts and promotion outcomes against the same season-total draft contract.
- Require the sampled comparison workflow to support three decision outcomes:
  - sampled Bayes is promoted
  - empirical Bayes remains production
  - sampled Bayes is not promoted overall but yields actionable learning for rookie or sparse-history priors
- Update evidence and documentation surfaces so they explain undefined validation metrics and sampled-comparison outcomes truthfully without implying automatic sampled promotion.

**Non-Goals:**
- This change does not automatically promote the sampled hierarchical estimator into the production dashboard, CLI, or supported workflow.
- This change does not redesign the empirical-Bayes production model target, hierarchy, or output structure established by the previous change.
- This change does not introduce new dashboard sections beyond what is needed to represent comparison outcomes and non-estimable validation metrics truthfully.
- This change does not treat warning suppression as an acceptable fix for deprecated pandas behavior.
- This change does not reopen the retired hybrid MC/random-forest lane or create a second production board path.

## Decisions

### 1. Undefined rank metrics are represented as not estimable, not as zero

**Decision**

Any rank-correlation metric that cannot be estimated because inputs are constant or insufficiently variable SHALL be represented as `null` or an equivalent explicit unavailable value in persisted artifacts, and SHALL render as `n/a` or equivalent in evidence-facing surfaces.

**Why**

`0.0` is a statistical claim: it says the repo estimated no monotonic relationship. An undefined correlation is a different state. The repo must not erase that distinction.

**Alternatives considered**

- Convert undefined correlations to `0.0`: rejected because it misstates the underlying estimability problem.
- Omit the metric silently: rejected because it hides why the value is absent and makes comparisons harder to audit.
- Preserve raw warnings only: rejected because warnings are not a stable artifact contract.

### 2. Warning cleanup should replace deprecated grouped operations, not suppress them

**Decision**

The `advanced_stats_calculator.py` cleanup SHALL convert warning-producing grouped operations into explicit aggregation logic on the relevant columns or grouped frames, without adding warning suppression or compatibility branches as the primary solution.

**Why**

This repo already has a no-patchwork rule. Suppressing warnings would preserve the current ambiguity and still leave future behavior changes exposed.

**Alternatives considered**

- Suppress pandas warnings in tests or pipeline runs: rejected because it hides real drift.
- Add version-conditional `include_groups`-style patches everywhere: rejected unless absolutely necessary, because it spreads API-compatibility scaffolding instead of clarifying the grouped calculations.

### 2a. Orchestration warnings should be fixed at the source, not tolerated

**Decision**

The remaining pre-draft orchestration warnings SHALL be resolved by normalizing the installed nflverse player-team alias surface at the backend adapter layer and by removing eager package-level imports that preload executable draft-strategy modules before `python -m` orchestration runs them.

**Why**

Both warnings are implementation-contract problems, not expected operational noise. The backend loader should adapt the real installed `nflreadpy` schema, and the draft-strategy package should not import executable modules just to populate package-level convenience symbols.

**Alternatives considered**

- Leave the warnings in place as harmless noise: rejected because they obscure whether a run contains real analysis warnings.
- Keep eager package exports and special-case the warning output: rejected because it preserves the import-order bug instead of removing it.

### 3. Sampled Bayes promotion must be governed by a persisted comparison artifact

**Decision**

The sampled hierarchical estimator SHALL remain an evaluation lane unless a persisted comparison artifact shows that it materially outperforms the empirical-Bayes baseline according to the declared contract.

That comparison artifact must record:
- search space
- per-configuration results
- rejected configurations
- convergence diagnostics
- runtime costs
- overall forecast metrics
- calibration metrics
- ranking metrics
- rookie/veteran or sparse-history slices
- the chosen decision outcome and rationale

**Why**

The current code can run a bounded search, but promotion remains too implicit. A formal artifact makes the decision reproducible and reviewable.

**Alternatives considered**

- Promote sampled Bayes if one configuration looks best locally: rejected because that is not auditable enough.
- Keep sampled Bayes permanently as a research-only helper with no formal decision report: rejected because the repo would continue to carry unresolved governance ambiguity.

### 4. Promotion outcomes are trinary, not binary

**Decision**

The formal sampled-comparison workflow SHALL support three outcomes:

1. `promote_sampled_bayes`
2. `keep_empirical_bayes`
3. `keep_empirical_bayes_with_sampled_prior_followup`

The third outcome covers cases where sampled Bayes does not win overall but does surface actionable improvements for rookies or sparse-history players.

**Why**

The user’s actual interest is not “is sampled Bayes globally cooler?” It is whether it improves the draft engine, especially for thin-data players. A strict binary gate would lose useful information.

**Alternatives considered**

- Binary promote/fail only: rejected because it hides the rookie-specific learning case.
- Always keep sampled code regardless of comparison outcome: rejected because it reintroduces dormant machinery without a clear supported role.

### 5. Dashboard and documentation surfaces stay empirical-first unless promotion occurs

**Decision**

Dashboard evidence, operator docs, and public workflow descriptions SHALL continue to identify empirical Bayes as the production estimator unless the sampled comparison artifact explicitly yields a promotion outcome.

If the sampled workflow remains unpromoted, docs and evidence may summarize the comparison process and result, but they must not imply that the sampled estimator is already powering the live board.

**Why**

This preserves the current supported workflow while still documenting the evaluation work honestly.

**Alternatives considered**

- Expose sampled-comparison language in the main dashboard as if it were live production behavior: rejected because it would overstate the current operational state.

### 6. Sampled comparison remains diagnostics-first, not a new supported CLI surface

**Decision**

The formal sampled-versus-empirical comparison SHALL remain an internal diagnostics and explicit-tooling workflow rather than a new default supported CLI command for operators.

**Why**

The sampled estimator may never be promoted. Building a first-class operator CLI path around an evaluation lane would add user-facing surface area and maintenance cost before the repo has evidence that the path belongs in the supported workflow.

**Alternatives considered**

- Add a new dedicated operator-facing CLI command now: rejected because it would harden unsupported surface area around a lane that may still be retired.
- Hide the workflow entirely inside tests: rejected because the comparison still needs to be rerunnable and auditable outside test-only contexts.

### 7. The third outcome keeps its long canonical key and gets shorter operator wording

**Decision**

The formal artifact key SHALL remain `keep_empirical_bayes_with_sampled_prior_followup`, while operator-facing documentation may describe that outcome with shorter wording such as "Keep empirical Bayes; use sampled findings for prior follow-up."

**Why**

The long key is precise and stable for artifacts and tests. Operator-facing prose should stay readable without weakening the underlying governance meaning.

**Alternatives considered**

- Shorten the canonical artifact key itself: rejected because it would make the stored outcome less explicit.
- Force the long canonical string into all operator-facing prose: rejected because it is unnecessarily clumsy in docs.

### 8. Sampled-comparison outcomes stay out of the live dashboard until promotion

**Decision**

Sampled-versus-empirical comparison outcomes SHALL remain in diagnostics and documentation until a promotion outcome exists. The live dashboard SHALL not surface sampled-comparison results while sampled Bayes remains an evaluation lane.

**Why**

Evaluation work should not appear in the live board unless it changes the live estimator. Surfacing it early would add clutter and imply operational relevance that the workflow has not earned.

**Alternatives considered**

- Show evaluation outcomes in dashboard evidence before promotion: rejected because it would imply that evaluation activity is part of the live recommendation surface.

### 9. The change ends with a full-data production-mode interpretation and decision report

**Decision**

This change SHALL end with a formal interpretation report produced from full production-mode execution on the complete available data surface, not only from fixtures or narrow evaluation slices.

That report SHALL:
- confirm that the supported production workflow was run end to end on the full available data
- summarize the observed warning cleanup results
- summarize the observed validation-semantic results
- summarize sampled-versus-empirical comparison results
- interpret those results in statistical and operational terms
- state the final move-forward decision for the repository

The final move-forward decision SHALL be one of:
1. promote sampled Bayes into the supported production path
2. keep empirical Bayes as production because sampled evaluation remains incomplete
3. keep empirical Bayes as production and retain sampled follow-up only for prior-design learning
4. keep empirical Bayes as production and retire the sampled lane from the supported workflow

The intermediate comparison artifact may still record `keep_empirical_bayes`, but the final interpretation report SHALL resolve that non-promotion result according to what actually happened:
- if the bounded sampled evaluation did not complete, the final report SHALL record an incomplete-evaluation outcome rather than pretending the workflow was retired
- if the bounded sampled evaluation completed and showed targeted rookie or sparse-history value, the final report SHALL record prior follow-up
- if the bounded sampled evaluation completed and showed no supported ongoing role, the final report SHALL record retirement

**Why**

The repository needs a decision, not just implementation. A final report grounded in the full production-mode run is the only defensible place to interpret the evidence and decide how the repo moves forward.

**Alternatives considered**

- End with test-only evidence and no formal interpretation: rejected because it would leave the key governance decision unresolved.
- Base the final decision only on fixtures or reduced slices: rejected because the user explicitly wants the decision grounded in production-mode execution on the full data surface.

## Risks / Trade-offs

- **[Risk] Cleanup changes alter advanced-stat outputs slightly** → **Mitigation:** require targeted regression tests around the affected grouped calculations and compare the cleaned outputs against existing fixture expectations.
- **[Risk] Replacing `0.0` with `null` breaks downstream assumptions** → **Mitigation:** update evidence payload contracts, dashboard rendering, docs, and tests together so `n/a` becomes the supported representation.
- **[Risk] Sampled-comparison artifacts become another output family sprawl** → **Mitigation:** keep them inside canonical diagnostics or model-output locations under the existing `seasons/<year>/...` structure and explicitly forbid extra parallel surfaces.
- **[Risk] Promotion criteria remain too coarse for rookie-specific gains** → **Mitigation:** require separate rookie or sparse-history reporting and support the third decision outcome rather than forcing a binary conclusion.
- **[Risk] Comparison workflow becomes expensive enough to discourage reruns** → **Mitigation:** preserve the bounded search space and persist results so the decision is inspectable without rerunning the whole search every time.
- **[Risk] Backend alias drift breaks current-player enrichment again** → **Mitigation:** normalize installed nflverse aliases in one adapter layer and pin that behavior with backend regression tests.
- **[Risk] Package convenience imports reintroduce `runpy` warnings** → **Mitigation:** keep `draft_strategy/__init__.py` side-effect free and test that package import does not preload executable strategy modules.

## Migration Plan

1. Normalize validation semantics first:
   - define artifact representation for non-estimable rank metrics
   - update evidence-facing display expectations
2. Replace warning-producing grouped operations in `advanced_stats_calculator.py`
   - keep functional behavior aligned with existing advanced-stat intent
   - remove warning-producing pandas idioms
3. Extend the sampled evaluation lane into a formal comparison workflow
   - persist comparison results
   - persist promotion outcome and rationale
4. Update docs and any non-dashboard evidence surfaces to describe:
   - undefined validation metrics
   - sampled-comparison outcomes
   - empirical-Bayes production-default rule
5. Add regression coverage for:
   - absence of deprecated-warning patterns in the touched calculations
   - `null`/not-estimable validation semantics
   - sampled-promotion decision outcomes
   - no accidental sampled promotion without an explicit winning artifact
6. Run the full supported production-mode workflow on the complete available data surface and write the formal interpretation report that decides how the repo moves forward.

Rollback is straightforward because the current production estimator remains empirical Bayes. If the comparison workflow proves too disruptive, the sampled-evaluation surface can remain internal to diagnostics while the warning and validation cleanup still ship independently.

## Resolved Follow-ups

- Sampled-versus-empirical comparison remains a diagnostics and explicit-tooling workflow, not a new default supported CLI surface.
- The canonical third-outcome artifact key remains `keep_empirical_bayes_with_sampled_prior_followup`, while shorter operator-facing wording is allowed in documentation.
- Sampled-comparison outcomes remain out of the live dashboard until a formal promotion outcome exists.
- The change closes with a production-mode full-data interpretation report and an explicit move-forward decision for the repository.
