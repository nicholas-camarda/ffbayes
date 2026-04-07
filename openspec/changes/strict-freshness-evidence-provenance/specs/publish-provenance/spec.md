## ADDED Requirements

### Requirement: Pages staging SHALL include publish-time provenance
`ffbayes publish-pages` MUST stage publish-time provenance for the dashboard artifacts it copies into `site/`.

#### Scenario: Fresh publish records provenance
- **WHEN** `ffbayes publish-pages` stages a dashboard generated from fresh runtime artifacts
- **THEN** the staged output MUST record publish time, source artifact identity, generation time, and fresh status

#### Scenario: Degraded publish records provenance
- **WHEN** `ffbayes publish-pages` stages a dashboard generated under degraded freshness conditions
- **THEN** the staged output MUST record the degraded state and its cause instead of presenting the site as fully current

### Requirement: Provenance SHALL be visible to dashboard consumers
The staged dashboard payload and/or staged provenance artifact MUST expose enough provenance information for Pages viewers to inspect artifact lineage without accessing runtime-only files.

#### Scenario: Pages viewer sees lineage fields
- **WHEN** a viewer opens the staged dashboard from `site/`
- **THEN** the dashboard data available to the UI MUST include publish-time lineage fields sufficient to show when the artifact was generated, staged, and under what freshness status

#### Scenario: Missing provenance is explicit
- **WHEN** required provenance inputs are missing during staging
- **THEN** the staged artifact MUST surface missing provenance explicitly rather than omitting the field silently

### Requirement: Provenance SHALL preserve existing artifact locations
Adding provenance MUST NOT rename the canonical staged dashboard targets or move them out of the existing `site/` structure.

#### Scenario: Existing Pages targets remain stable
- **WHEN** provenance support is added to `ffbayes publish-pages`
- **THEN** the staged dashboard MUST still publish through `site/index.html` and `site/dashboard_payload.json`

#### Scenario: Additional provenance artifact does not break staging
- **WHEN** publish-time provenance is emitted as an additional staged file
- **THEN** it MUST coexist with current staged files without changing the required path contract for existing viewers and tests
