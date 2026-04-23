## ADDED Requirements

### Requirement: The supported player forecast stack SHALL target season-total fantasy value for the dashboard
The supported `pre_draft` player forecast stack MUST produce posterior predictive summaries for season-total fantasy points under the active scoring preset. The player forecast contract MUST be the authoritative source for draft-facing player value, uncertainty, and replacement-level comparisons used by the dashboard and draft decision engine.

#### Scenario: Draft board consumes season-total contract
- **WHEN** `ffbayes pre-draft` or `ffbayes draft-strategy` generates the canonical runtime draft artifacts
- **THEN** the player forecast table used by the draft board MUST expose season-total posterior summaries rather than a production contract centered only on per-game means

#### Scenario: Forecast target is explicit in artifacts and docs
- **WHEN** the runtime payload, evidence metadata, or technical documentation describes the supported player forecast
- **THEN** it MUST identify the forecast target as season-total fantasy value and MUST NOT describe the production model as a vague or mislabeled generic Bayesian ranking

### Requirement: The primary player model SHALL use a two-part forecast structure
The supported player forecast stack MUST model season-total fantasy value through separate scoring-rate and availability components, and the emitted season-total posterior summaries MUST be derived from those components in a way that remains auditable.

The first supported implementation MUST compose season-total posterior summaries through posterior predictive simulation from those components rather than through a purely analytic shortcut that obscures uncertainty propagation.

#### Scenario: High-rate low-availability veteran is distinguished from stable-volume player
- **WHEN** the forecast stack evaluates a veteran wide receiver with elite historical scoring efficiency but repeated missed games and a durable wide receiver with lower rate but near-full-season availability
- **THEN** the resulting season-total posterior summaries MUST preserve the distinction between rate upside and availability risk instead of collapsing both players to one undifferentiated uncertainty score

#### Scenario: Rookie total value is not reduced to position-average fallback
- **WHEN** the forecast stack evaluates a first-round rookie wide receiver with no NFL history but known draft capital, combine results, and depth-chart context
- **THEN** the season-total posterior MUST be derived from structured rookie priors rather than from a position-average fallback that ignores those covariates

### Requirement: The supported model SHALL include team-season and rookie prior structure
The supported player forecast stack MUST incorporate team-season effects and rookie priors informed by draft capital, combine results, and depth-chart context. These features MUST enter the same supported player-model contract rather than being delegated to a separate legacy or hybrid pipeline.

The first supported production cut of contextual structure beyond player and position MUST be limited to:

- team-season effects
- team-change indicators
- depth-chart or roster-competition context
- schedule-strength aggregates only if validated against holdout performance

College-conference effects and full opponent-by-opponent modeling MUST remain out of scope for this first production contract.

#### Scenario: Team landing spot changes rookie prior
- **WHEN** two rookie running backs have similar draft capital and combine profiles but materially different team-season and depth-chart contexts
- **THEN** the supported player model MUST allow their prior season-total outlooks to differ because of landing-spot context

#### Scenario: Team change affects veteran outlook
- **WHEN** a veteran wide receiver changes teams between seasons
- **THEN** the supported player model MUST include team-season information in the resulting forecast rather than treating team context as absent from the production contract

### Requirement: Empirical Bayes SHALL remain the production baseline on the supported contract
The supported production forecast path MUST retain an empirical-Bayes estimator as the default player-model implementation unless and until another estimator wins against the same contract on holdout performance, calibration, and operational readiness.

#### Scenario: Sampled extension is present but not proven
- **WHEN** a sampled hierarchical estimator exists for research comparison but has not yet satisfied the required validation and operational criteria
- **THEN** the pipeline MUST continue to use the empirical-Bayes estimator as the production forecast path for dashboard generation

#### Scenario: One contract supports multiple estimators
- **WHEN** maintainers compare an empirical-Bayes estimator and a sampled hierarchical estimator
- **THEN** both estimators MUST produce outputs against the same season-total player-forecast contract rather than creating incompatible downstream dashboard logic

### Requirement: The sampled hierarchical extension SHALL be evaluated on the same production target
If a sampled hierarchical estimator is implemented, it MUST model the same season-total player forecast target and MUST emit convergence, posterior predictive, and calibration diagnostics that can be compared directly with the empirical-Bayes baseline.

#### Scenario: Sampled model emits diagnostics
- **WHEN** the sampled hierarchical estimator runs in an evaluation path
- **THEN** it MUST emit convergence diagnostics, posterior predictive summaries, and holdout-facing diagnostics that are attributable to the same production target

#### Scenario: Sampled model is not promoted by branding alone
- **WHEN** the sampled estimator is available
- **THEN** the system MUST NOT present it as the production model solely because it uses MCMC or posterior draws

#### Scenario: Empirical Bayes remains default until promotion criteria are met
- **WHEN** the sampled hierarchical estimator has not yet met the declared holdout, calibration, and operational-readiness criteria
- **THEN** the supported dashboard and draft-generation path MUST continue to use the empirical-Bayes estimator by default

#### Scenario: Non-winning sampled estimator does not remain in the supported workflow
- **WHEN** the sampled hierarchical estimator fails to materially improve the supported contract on the declared promotion criteria
- **THEN** it MUST NOT remain wired into supported CLI, pipeline, dashboard, or documentation surfaces as a dormant alternative production path

### Requirement: The sampled hierarchical extension SHALL explore bounded training and inference settings
If a sampled hierarchical estimator is implemented, the repo MUST evaluate a bounded set of training and inference configurations and select from that set using explicit convergence, predictive, calibration, and runtime criteria.

The first bounded exploration MUST include a small concrete search over:

- centered and non-centered parameterizations
- conservative, medium, and weaker hierarchical prior-scale families
- `target_accept` values of `0.9` and `0.95`
- `chains=4`
- warmup/draw budgets of `1000/1000` and `1500/1500`
- structural variants spanning:
  - base hierarchical model
  - base hierarchical model plus team-season effects
  - base hierarchical model plus team-season effects and rookie priors

#### Scenario: Multiple sampled configurations are compared
- **WHEN** the sampled hierarchical estimator is evaluated for possible use in the supported workflow
- **THEN** the implementation MUST compare more than one training or inference configuration rather than treating a single arbitrary configuration as the validated default

#### Scenario: Selection criteria are explicit
- **WHEN** one sampled configuration is retained as the preferred evaluated configuration
- **THEN** that selection MUST be attributable to explicit convergence, predictive, calibration, and runtime criteria rather than to convenience alone

#### Scenario: Pathological sampled configuration fails closed
- **WHEN** a sampled configuration exhibits unacceptable convergence or inference pathologies such as poor R-hat, low ESS, divergences, or unstable runtime behavior
- **THEN** the system MUST reject that configuration rather than silently using it in the supported workflow
