## ADDED Requirements

### Requirement: Dashboard payload SHALL expose structured decision evidence
The draft dashboard payload MUST expose a structured decision-evidence section derived from the backtest and related trust metadata, rather than only a thin summary table.

#### Scenario: Evidence is available
- **WHEN** the draft strategy artifact has access to a valid backtest and freshness metadata
- **THEN** the payload MUST include evaluation scope, strategy summary, season-level evidence, limitations, and freshness context for the evidence panel

#### Scenario: Evidence is unavailable or partial
- **WHEN** the draft strategy artifact cannot load a valid backtest or the evidence is incomplete
- **THEN** the payload MUST include an explicit unavailable or degraded evidence state with the reason and interpretation limits

### Requirement: Dashboard UI SHALL render evidence with limitations
The draft dashboard UI MUST render the decision-evidence panel in a way that highlights both the comparative results and their interpretation limits.

#### Scenario: Evidence panel shows result and limitations
- **WHEN** the dashboard loads a payload with structured decision evidence
- **THEN** the UI MUST display the comparative evidence and the associated limitations in the same evidence surface

#### Scenario: Evidence panel shows degraded state
- **WHEN** the dashboard loads a payload whose decision evidence is degraded or unavailable
- **THEN** the UI MUST display a clear degraded or unavailable message instead of implying that evidence is complete

### Requirement: Decision evidence SHALL be reusable across export surfaces
The same normalized decision-evidence contract MUST be reusable in the runtime JSON payload, workbook-facing summaries, and staged Pages experience.

#### Scenario: Runtime and Pages share evidence semantics
- **WHEN** a dashboard payload is staged from runtime artifacts into `site/`
- **THEN** the decision-evidence content presented in the Pages payload MUST match the evidence semantics from the runtime artifact

#### Scenario: Evidence summary remains testable
- **WHEN** automated tests validate the draft dashboard and related artifact generation
- **THEN** they MUST be able to assert evidence availability, degraded states, and limitations from deterministic structured fields
