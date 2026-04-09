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
The draft dashboard UI MUST render the decision-evidence panel in a way that highlights comparative results, internal-validation scope, uncertainty context, and interpretation limits without duplicating the same strategy-summary content across multiple panels. Metric labels used in evidence-facing UI MUST remain consistent with the glossary and inspector.

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

### Requirement: Decision evidence SHALL be reusable across export surfaces
The same normalized decision-evidence contract MUST be reusable in the runtime JSON payload, workbook-facing summaries, and staged Pages experience.

#### Scenario: Runtime and Pages share evidence semantics
- **WHEN** a dashboard payload is staged from runtime artifacts into `site/`
- **THEN** the decision-evidence content presented in the Pages payload MUST match the evidence semantics from the runtime artifact

#### Scenario: Evidence summary remains testable
- **WHEN** automated tests validate the draft dashboard and related artifact generation
- **THEN** they MUST be able to assert evidence availability, degraded states, and limitations from deterministic structured fields

### Requirement: Evidence terminology SHALL remain consistent across surfaces
The dashboard payload and UI MUST use consistent user-facing terminology for ranking and recommendation metrics across cards, inspector views, and glossary definitions.

#### Scenario: Metric label matches glossary and inspector
- **WHEN** the dashboard presents a metric such as `draft_score` or `board_value_score`
- **THEN** the visible label shown in cards, inspector text, and glossary entries MUST not conflict about what that metric is called

#### Scenario: Evidence disagreement summaries use canonical baseline names
- **WHEN** the dashboard compares the contextual draft policy against a simpler baseline
- **THEN** the disagreement summary MUST use one consistent baseline name across payload fields and UI copy
