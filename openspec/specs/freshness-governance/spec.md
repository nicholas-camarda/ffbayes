## Purpose
Defines the freshness governance contract for pre-draft inputs, serialized freshness state, and user-facing degraded-state interpretation.

## Requirements

### Requirement: Pre-draft freshness SHALL fail closed by default
The `ffbayes` pre-draft pipeline and related freshness-sensitive commands MUST reject runs that are missing the latest expected season unless the operator has provided an explicit degraded-data override.

#### Scenario: Missing latest season blocks default execution
- **WHEN** a freshness-sensitive pre-draft command resolves its analysis window and the latest expected season is missing
- **THEN** the command MUST stop with an explicit freshness failure instead of silently continuing

#### Scenario: Explicit override permits degraded execution
- **WHEN** the operator provides the supported stale-data override for a freshness-sensitive pre-draft command
- **THEN** the command MAY continue, but it MUST mark the run as degraded and override-driven in emitted artifacts and user-visible status

### Requirement: Freshness state SHALL be serialized into runtime artifacts
Freshness-sensitive draft artifacts MUST record the resolved analysis window, freshness status, missing years, and whether an explicit override was used.

#### Scenario: Runtime artifact captures fresh execution
- **WHEN** a pre-draft run completes with a fully fresh analysis window
- **THEN** its runtime metadata MUST record a fresh status with no override flag

#### Scenario: Runtime artifact captures degraded execution
- **WHEN** a pre-draft run completes under an explicit stale-data override
- **THEN** its runtime metadata MUST record degraded status, the missing freshness conditions, and the override indicator

### Requirement: Freshness semantics SHALL be consistent across pipeline and backtest surfaces
The draft backtest, draft strategy generation, and publish staging flow MUST use the same freshness interpretation so a degraded run cannot appear fresh in another artifact surface.

#### Scenario: Backtest inherits degraded freshness state
- **WHEN** the draft backtest is generated from a degraded analysis window
- **THEN** the backtest-derived evidence exposed to downstream artifacts MUST identify the degraded freshness state

#### Scenario: Publish staging preserves freshness truth
- **WHEN** `ffbayes publish-pages` stages a dashboard produced from degraded inputs
- **THEN** the staged site metadata MUST preserve the degraded freshness state rather than implying a fresh publish

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
