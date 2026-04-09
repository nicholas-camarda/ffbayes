## ADDED Requirements

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
