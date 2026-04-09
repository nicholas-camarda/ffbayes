## MODIFIED Requirements

### Requirement: Dashboard UI SHALL render evidence with limitations
The draft dashboard UI MUST render the decision-evidence panel in a way that highlights comparative results, interpretation limits, and freshness context without duplicating the same strategy-summary content across multiple panels. Metric labels used in evidence-facing UI MUST remain consistent with the glossary and inspector.

#### Scenario: Evidence panel shows result and limitations
- **WHEN** the dashboard loads a payload with structured decision evidence
- **THEN** the UI MUST display the comparative evidence and the associated limitations in the same evidence surface

#### Scenario: Evidence panel shows degraded state
- **WHEN** the dashboard loads a payload whose decision evidence is degraded or unavailable
- **THEN** the UI MUST display a clear degraded or unavailable message instead of implying that evidence is complete

#### Scenario: Evidence freshness is interpretable
- **WHEN** decision evidence is present but freshness is degraded, mixed, or override-driven
- **THEN** the evidence-facing UI MUST explain that degraded state in direct user-facing terms rather than only surfacing an opaque status label

#### Scenario: Redundant strategy-summary tables are avoided
- **WHEN** evidence and provenance surfaces would otherwise restate the same strategy-summary rows
- **THEN** the dashboard MUST present one authoritative comparative summary and keep the other surface focused on lineage or freshness metadata

## ADDED Requirements

### Requirement: Evidence terminology SHALL remain consistent across surfaces
The dashboard payload and UI MUST use consistent user-facing terminology for ranking and recommendation metrics across cards, inspector views, and glossary definitions.

#### Scenario: Metric label matches glossary and inspector
- **WHEN** the dashboard presents a metric such as `draft_score` or `board_value_score`
- **THEN** the visible label shown in cards, inspector text, and glossary entries MUST not conflict about what that metric is called

#### Scenario: Evidence disagreement summaries use canonical baseline names
- **WHEN** the dashboard compares the contextual draft policy against a simpler baseline
- **THEN** the disagreement summary MUST use one consistent baseline name across payload fields and UI copy
