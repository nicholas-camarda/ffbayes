## Purpose
Defines the sampled-Bayes promotion governance contract for comparison artifacts, promotion decisions, and production model ownership.

## Requirements

### Requirement: Sampled-versus-empirical evaluation SHALL produce a persisted comparison artifact
The repository MUST evaluate the sampled hierarchical estimator against the empirical-Bayes production baseline through one persisted comparison artifact family rather than through ad hoc console output or implicit selector behavior.

This evaluation workflow MUST remain diagnostics-first and MUST NOT be introduced as a default operator-facing production CLI path while sampled Bayes remains an evaluation lane.

The persisted comparison artifact MUST summarize:
- the declared search space
- the evaluated configurations
- rejected configurations and rejection reasons
- convergence diagnostics
- runtime costs
- overall forecast metrics
- calibration metrics
- ranking metrics
- rookie versus veteran or sparse-history slices
- the final decision outcome and rationale

#### Scenario: Comparison uses the same production contract
- **WHEN** the sampled hierarchical estimator is evaluated for possible promotion
- **THEN** the comparison MUST use the same season-total forecast contract, holdout framing, and dashboard-relevant output semantics as the empirical-Bayes baseline

#### Scenario: Comparison artifact is auditable after the run
- **WHEN** a sampled-versus-empirical comparison finishes
- **THEN** the repo MUST persist a structured artifact that allows a later reviewer to see which configurations were searched, which ones were rejected, and why the final decision was reached

#### Scenario: Comparison artifact stays inside canonical output structure
- **WHEN** the repository writes sampled-comparison results
- **THEN** those outputs MUST live under the canonical `seasons/<year>/...` diagnostics or model-output tree and MUST NOT create a parallel dashboard or ad hoc artifact family

### Requirement: Sampled-model governance SHALL support trinary outcomes
The sampled-versus-empirical comparison workflow MUST support three explicit decision outcomes rather than a binary promote-or-fail result.

The allowed outcomes are:
1. `promote_sampled_bayes`
2. `keep_empirical_bayes`
3. `keep_empirical_bayes_with_sampled_prior_followup`

#### Scenario: Sampled Bayes materially wins overall
- **WHEN** a sampled configuration satisfies convergence and runtime gates and materially improves the declared forecast, calibration, and ranking criteria over the empirical-Bayes baseline
- **THEN** the comparison artifact MUST record the outcome `promote_sampled_bayes` together with the winning configuration and the supporting evidence

#### Scenario: Sampled Bayes does not win overall
- **WHEN** the sampled evaluation does not materially outperform the empirical-Bayes baseline on the declared contract
- **THEN** the comparison artifact MUST record the outcome `keep_empirical_bayes`

#### Scenario: Sampled Bayes helps rookies or sparse-history players only
- **WHEN** the sampled evaluation does not win overall but surfaces materially better rookie or sparse-history behavior that is still relevant for future prior design
- **THEN** the comparison artifact MUST record the outcome `keep_empirical_bayes_with_sampled_prior_followup` and MUST explain which slices improved and why that did not justify promotion

### Requirement: Empirical Bayes SHALL remain production until a promotion artifact says otherwise
The repository MUST continue to treat empirical Bayes as the production estimator unless the persisted sampled-comparison artifact explicitly records a promotion outcome.

#### Scenario: No silent production promotion occurs
- **WHEN** sampled-model evaluation code runs without producing a promotion outcome
- **THEN** the production dashboard, supported CLI workflow, and default operator-facing documentation MUST continue to identify empirical Bayes as the live estimator

#### Scenario: Evaluation workflow does not become a default operator command
- **WHEN** sampled Bayes remains unpromoted
- **THEN** the comparison workflow MUST remain a diagnostics or explicit-tooling surface rather than a default operator-facing production command

#### Scenario: Promotion is explicit and reviewable
- **WHEN** the sampled evaluation produces `promote_sampled_bayes`
- **THEN** the repo MUST require the promotion artifact to be available for inspection before sampled Bayes is described as the production estimator in dashboard evidence or docs

### Requirement: Search-space exploration SHALL be reviewable, not decorative
The bounded sampled-model search in `sampled_player_model.py` MUST remain reviewable as part of the governance workflow instead of behaving like a hidden optimizer.

#### Scenario: Rejected configurations explain failure mode
- **WHEN** a sampled configuration is rejected because of divergences, weak effective sample size, poor rank or calibration performance, excessive runtime, or other gate failures
- **THEN** the comparison artifact MUST preserve the rejection reason in structured form

#### Scenario: Search output distinguishes overall and slice-level behavior
- **WHEN** multiple sampled configurations are compared
- **THEN** the persisted results MUST distinguish overall results from rookie, veteran, or sparse-history slice behavior rather than flattening all comparison evidence into one score

### Requirement: Governance SHALL end with a full-data interpretation report and repository decision
The sampled-governance workflow MUST end with one formal interpretation artifact produced after running the supported production-mode workflow on the complete available data surface.

That interpretation artifact MUST:
- reference the full-data production-mode run it was derived from
- summarize empirical-Bayes production results
- summarize sampled-versus-empirical comparison results
- summarize any remaining warning or metric-semantics issues that materially affect interpretation
- interpret the observed evidence in statistical and operational terms
- state one explicit repository decision about how to move forward

The allowed final decisions are:
1. `promote_sampled_bayes`
2. `keep_empirical_bayes_evaluation_incomplete`
3. `keep_empirical_bayes_with_sampled_prior_followup`
4. `keep_empirical_bayes_and_retire_sampled_workflow`

The earlier comparison artifact may record the intermediate non-promotion outcome `keep_empirical_bayes`, but the final interpretation artifact MUST distinguish:
- incomplete sampled evaluation, which resolves to `keep_empirical_bayes_evaluation_incomplete`
- completed sampled evaluation with actionable rookie or sparse-history learning, which resolves to `keep_empirical_bayes_with_sampled_prior_followup`
- completed sampled evaluation with no supported ongoing role, which resolves to `keep_empirical_bayes_and_retire_sampled_workflow`

#### Scenario: Final interpretation is grounded in the full production run
- **WHEN** the repository closes the sampled-governance evaluation
- **THEN** it MUST base the final interpretation artifact on a full production-mode run over the complete available data surface rather than only on reduced fixtures or isolated unit-test slices

#### Scenario: Final interpretation makes an explicit move-forward decision
- **WHEN** the final interpretation artifact is written
- **THEN** it MUST state one allowed final decision explicitly rather than leaving the repository in an unresolved evaluation state

#### Scenario: Incomplete sampled evaluation does not justify retirement
- **WHEN** the bounded sampled evaluation did not complete and the sampled estimator therefore never finished the declared comparison contract
- **THEN** the final interpretation artifact MUST record `keep_empirical_bayes_evaluation_incomplete` rather than treating the sampled workflow as retired
