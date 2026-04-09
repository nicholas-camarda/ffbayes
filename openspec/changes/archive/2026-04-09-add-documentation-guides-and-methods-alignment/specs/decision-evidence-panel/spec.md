## MODIFIED Requirements

### Requirement: Dashboard UI SHALL render evidence with limitations
The draft dashboard UI MUST render the decision-evidence panel in a way that highlights comparative results, internal-validation scope, uncertainty context, and interpretation limits without duplicating the same strategy-summary content across multiple panels. Metric labels used in evidence-facing UI MUST remain consistent with the glossary and inspector.

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
