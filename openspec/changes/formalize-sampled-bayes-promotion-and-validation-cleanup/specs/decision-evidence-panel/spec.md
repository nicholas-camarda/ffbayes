## MODIFIED Requirements

### Requirement: Dashboard payload SHALL expose structured decision evidence
The draft dashboard payload MUST expose a structured decision-evidence section derived from the backtest, related trust metadata, and current validation semantics rather than only a thin summary table.

When decision evidence includes validation metrics that are non-estimable, the payload MUST preserve those values as unavailable states with explicit reasons instead of fabricated numeric substitutes. Sampled-versus-empirical comparison results MUST remain outside the live dashboard payload until a formal promotion outcome exists.

#### Scenario: Evidence is available
- **WHEN** the draft strategy artifact has access to a valid backtest and freshness metadata
- **THEN** the payload MUST include evaluation scope, strategy summary, season-level evidence, limitations, and freshness context for the evidence panel

#### Scenario: Evidence is unavailable or partial
- **WHEN** the draft strategy artifact cannot load a valid backtest or the evidence is incomplete
- **THEN** the payload MUST include an explicit unavailable or degraded evidence state with the reason and interpretation limits

#### Scenario: Evidence payload preserves non-estimable validation metrics
- **WHEN** decision evidence includes a validation slice whose rank-correlation metric is not estimable
- **THEN** the payload MUST preserve that metric as unavailable with an explicit reason rather than coercing it to `0.0`

#### Scenario: Unpromoted sampled comparison is absent from live dashboard payload
- **WHEN** sampled Bayes remains an evaluation lane without a promotion outcome
- **THEN** the live dashboard payload MUST NOT include sampled-versus-empirical comparison summaries

### Requirement: Dashboard UI SHALL render evidence with limitations
The draft dashboard UI MUST render the decision-evidence panel in a way that highlights comparative results, internal-validation scope, uncertainty context, interpretation limits, and non-estimable validation states without duplicating the same strategy-summary content across multiple panels. Metric labels used in evidence-facing UI MUST remain consistent with the glossary and inspector.

#### Scenario: Evidence panel shows result and limitations
- **WHEN** the dashboard loads a payload with structured decision evidence
- **THEN** the UI MUST display the comparative evidence and the associated limitations in the same evidence surface

#### Scenario: Evidence panel shows result, scope, and limitations
- **WHEN** the dashboard loads a payload with structured decision evidence
- **THEN** the UI MUST display the comparative evidence, the associated limitations, and an explicit statement that the evidence is internal rather than external validation in the same evidence surface

#### Scenario: Evidence summary shows season coverage and uncertainty context
- **WHEN** the dashboard loads a payload whose decision evidence includes season-count, holdout-year, or interval/uncertainty context
- **THEN** the evidence-facing summary MUST surface that coverage and uncertainty context alongside winner or delta claims rather than hiding it only in secondary details

#### Scenario: Evidence panel shows degraded state
- **WHEN** the dashboard loads a payload whose decision evidence is degraded or unavailable
- **THEN** the UI MUST display a clear degraded or unavailable message instead of implying that evidence is complete

#### Scenario: Evidence freshness is interpretable
- **WHEN** decision evidence is present but freshness is degraded, mixed, or override-driven
- **THEN** the evidence-facing UI MUST explain that degraded state in direct user-facing terms rather than only surfacing an opaque status label

#### Scenario: Redundant strategy-summary tables are avoided
- **WHEN** evidence and provenance surfaces would otherwise restate the same strategy-summary rows
- **THEN** the dashboard MUST present one authoritative comparative summary and keep the other surface focused on lineage or freshness metadata

#### Scenario: UI renders unavailable validation metrics honestly
- **WHEN** the evidence payload includes a non-estimable validation metric
- **THEN** the UI MUST render that value as `n/a`, `not estimable`, or equivalent explicit language rather than as a neutral numeric score

#### Scenario: UI does not surface unpromoted sampled comparison
- **WHEN** sampled Bayes remains unpromoted
- **THEN** the live dashboard UI MUST NOT display sampled-versus-empirical comparison outcomes in the evidence panel

### Requirement: Decision evidence SHALL be reusable across export surfaces
The same normalized decision-evidence contract MUST be reusable in the runtime JSON payload, workbook-facing summaries, and staged Pages experience, including unavailable-metric states. Sampled-comparison governance summaries belong in diagnostics and documentation until promotion occurs.

#### Scenario: Runtime and Pages share evidence semantics
- **WHEN** a dashboard payload is staged from runtime artifacts into `site/`
- **THEN** the decision-evidence content presented in the Pages payload MUST match the evidence semantics from the runtime artifact

#### Scenario: Evidence summary remains testable
- **WHEN** automated tests validate the draft dashboard and related artifact generation
- **THEN** they MUST be able to assert evidence availability, degraded states, limitations, and unavailable-metric semantics from deterministic structured fields
