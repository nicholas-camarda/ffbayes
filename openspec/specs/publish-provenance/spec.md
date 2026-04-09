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

### Requirement: Published provenance SHALL explain degraded or ambiguous states
The staged Pages provenance contract MUST include actionable context when the published dashboard is mixed, degraded, override-driven, or otherwise not fully trustworthy.

#### Scenario: Degraded publish has actionable explanation
- **WHEN** `ffbayes publish-pages` stages a dashboard whose analysis freshness is degraded, mixed, or override-driven
- **THEN** the staged provenance available to Pages viewers MUST include enough explanation for the UI to describe the cause or limits of that degraded state

#### Scenario: Explanation is incomplete
- **WHEN** the staged provenance cannot provide a concrete degraded-state reason
- **THEN** the staged artifact MUST explicitly say that the reason detail is unavailable rather than leaving the degraded state unexplained

### Requirement: Staged provenance SHALL distinguish lineage from synchronization status
The staged dashboard provenance MUST distinguish analysis lineage from whether the staged HTML and staged payload are themselves synchronized.

#### Scenario: Analysis is valid but staged HTML is stale
- **WHEN** the staged payload carries valid lineage metadata but `site/index.html` is stale against that payload
- **THEN** the staged visualization contract MUST allow the UI or checks to identify the surface as stale without conflating that drift with the underlying analysis lineage

#### Scenario: Staged bundle is synchronized
- **WHEN** the staged HTML and staged payload match regeneration from the authoritative runtime artifacts
- **THEN** provenance and synchronization state MAY both be presented as aligned
