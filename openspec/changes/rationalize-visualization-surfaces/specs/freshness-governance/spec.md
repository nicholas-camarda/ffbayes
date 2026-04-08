## ADDED Requirements

### Requirement: Visualization-facing freshness states SHALL explain degraded status
Freshness-sensitive visualization surfaces MUST explain why a state is fresh, degraded, mixed, stale, or override-driven rather than only surfacing the status label itself.

#### Scenario: Mixed or override-driven dashboard is shown
- **WHEN** a runtime or staged dashboard is generated from inputs whose freshness state is mixed or override-driven
- **THEN** the visualization-facing surface MUST describe the degraded condition in concrete user-facing terms in addition to the raw status

#### Scenario: Freshness warnings are unavailable
- **WHEN** a visualization surface has a degraded freshness state but no detailed warning list is available
- **THEN** the system MUST surface that explanatory gap explicitly instead of implying that there are no reasons for degradation

### Requirement: Visualization drift SHALL not appear fresh
Visualization surfaces MUST NOT present a synchronized or trustworthy state when the rendered HTML and its paired payload are out of sync.

#### Scenario: Staged site HTML is stale against staged payload
- **WHEN** `site/index.html` does not match regeneration from `site/dashboard_payload.json`
- **THEN** the system MUST treat the staged visualization surface as stale rather than allowing freshness metadata alone to imply it is current

#### Scenario: Local shortcut HTML is stale against paired payload
- **WHEN** repo-local or runtime-local dashboard HTML diverges from regeneration from its paired payload
- **THEN** the system MUST surface the local visualization surface as stale
