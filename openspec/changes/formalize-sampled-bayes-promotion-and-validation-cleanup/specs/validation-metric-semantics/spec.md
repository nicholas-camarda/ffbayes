## ADDED Requirements

### Requirement: Non-estimable rank metrics SHALL be represented explicitly
Validation code MUST distinguish between an estimated near-zero rank association and a non-estimable rank metric caused by constant input, insufficient variation, or insufficient support.

A non-estimable rank metric MUST be represented as `null` or an equivalent explicit unavailable value in persisted artifacts, and it MUST carry a structured reason when the repository can determine that reason.

#### Scenario: Constant-input slice makes rank correlation undefined
- **WHEN** a holdout slice has constant truth values, constant predictions, or otherwise lacks enough variation to estimate a rank correlation
- **THEN** the validation artifact MUST record the correlation value as unavailable rather than coercing it to numeric `0.0`

#### Scenario: Estimated zero association remains numeric
- **WHEN** the repository can validly estimate a rank correlation and the estimate is numerically near zero
- **THEN** the validation artifact MUST preserve that numeric estimate rather than collapsing it with the unavailable state

### Requirement: Validation artifacts SHALL preserve unavailable-metric reasons
Persisted validation outputs MUST preserve the reason a metric is unavailable whenever that reason can be inferred deterministically from the evaluated slice.

#### Scenario: Artifact records reason for unavailable rank metric
- **WHEN** a rank-based validation metric is unavailable because of constant input or insufficient support
- **THEN** the persisted artifact MUST expose a structured reason such as `constant_input`, `insufficient_variation`, or another explicit unavailable state instead of omitting the explanation

#### Scenario: Tests can assert unavailable-metric semantics
- **WHEN** automated tests inspect validation artifacts
- **THEN** they MUST be able to assert both the unavailable value state and the associated reason without depending on warning strings

### Requirement: Evidence-facing surfaces SHALL render unavailable metrics honestly
Dashboard evidence, exported summaries, and related operator-facing trust surfaces MUST render unavailable validation metrics as `n/a`, `not estimable`, or equivalent explicit language rather than fabricated numeric neutrality.

#### Scenario: Dashboard evidence renders unavailable correlation
- **WHEN** a validation artifact includes an unavailable rank-correlation metric
- **THEN** the dashboard evidence layer MUST render that value as unavailable and MUST NOT display `0.00` or another fabricated neutral number

#### Scenario: Evidence text distinguishes estimability from performance
- **WHEN** the repository summarizes validation results for operators
- **THEN** the evidence text MUST distinguish between poor measured ranking performance and a slice where ranking performance was not estimable

### Requirement: Advanced-stat grouped calculations SHALL avoid deprecated pandas grouping semantics
Grouped calculations in `advanced_stats_calculator.py` that feed supported runtime outputs MUST use explicit grouped aggregation logic that does not rely on deprecated `groupby.apply(...)` grouping-column behavior.

#### Scenario: Advanced-stat calculation is implemented with explicit grouped logic
- **WHEN** the repository computes supported grouped advanced-stat transforms such as boom/bust ratios, floor/ceiling spreads, recent-form summaries, or season-trend summaries
- **THEN** those transforms MUST use explicit grouped calculations or aggregations rather than warning-producing deprecated grouping semantics

#### Scenario: Warning cleanup does not rely on suppression
- **WHEN** the touched advanced-stat calculations run in tests or supported workflows
- **THEN** they MUST avoid the deprecation warning by implementation change rather than by suppressing or filtering the warning

### Requirement: Regression coverage SHALL guard validation semantics and warning cleanup
The repository MUST include regression tests that fail if unavailable rank metrics are silently coerced back to numeric values or if the touched advanced-stat code resumes emitting the deprecated grouping warning.

#### Scenario: Undefined correlation is not silently converted
- **WHEN** regression tests exercise constant-input or low-support validation slices
- **THEN** the tests MUST fail if the reported metric value becomes numeric `0.0` instead of an explicit unavailable state

#### Scenario: Deprecated grouping warning does not reappear
- **WHEN** regression tests execute the touched advanced-stat calculations
- **THEN** the tests MUST fail if the deprecated pandas grouping warning reappears for those supported paths
