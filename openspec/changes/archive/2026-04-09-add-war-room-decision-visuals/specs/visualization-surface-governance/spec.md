## ADDED Requirements

### Requirement: Dashboard visual additions SHALL be decision-first
New visualization elements added to the live draft dashboard MUST justify their space by answering a clear draft-time decision question rather than serving as decorative or generic analytics.

#### Scenario: Proposed visual answers one clear question
- **WHEN** a new dashboard visual is added to the supported war-room surface
- **THEN** it MUST map to a concrete operator question such as whether to pick now, wait, or react to positional scarcity

#### Scenario: Redundant decorative view is rejected
- **WHEN** a proposed visual duplicates existing board, evidence, or inspector information without adding distinct decision support
- **THEN** the dashboard surface MUST treat it as out of scope for the supported war-room visualization set

### Requirement: Dashboard visual additions SHALL use progressive disclosure
New war-room visuals MUST preserve the usability of the existing dashboard by using compact defaults and progressive disclosure for heavier explanation surfaces.

#### Scenario: Primary war-room controls remain visible
- **WHEN** new visuals are integrated into the live dashboard
- **THEN** they MUST preserve the usability of recommendation lanes, board controls, and the player table rather than crowding them out

#### Scenario: Scarcity view stays close to the board without owning default space
- **WHEN** positional scarcity is integrated into the war room
- **THEN** it MUST stay near the player board while using a collapsed-by-default presentation so the board remains the primary default action surface

#### Scenario: Secondary explanation remains collapsible
- **WHEN** a visualization serves mainly as comparative explanation or trust support
- **THEN** it MUST be attached to collapsible or inspector-linked UI instead of always-open top-level dashboard real estate

### Requirement: Visualization contracts SHALL remain model-semantic rather than formula-specific
Supported dashboard visuals MUST depend on stable semantic concepts such as contextual score, baseline score, timing risk, or cliff strength instead of hardcoding today’s internal model feature weights or field names as the lasting UI contract.

#### Scenario: Current formula changes but semantics remain
- **WHEN** the draft model changes how it computes contextual recommendations while still exposing equivalent semantic concepts
- **THEN** the supported war-room visuals MUST remain valid without requiring a full redesign

#### Scenario: Baseline label changes in a future model
- **WHEN** the comparison baseline changes from the current simple VOR proxy to another supported baseline
- **THEN** the dashboard MUST update user-facing labels and semantic payload values without breaking the visualization surface contract

#### Scenario: New semantics extend existing visuals
- **WHEN** future model improvements introduce richer timing, scarcity, or comparative semantics
- **THEN** the supported dashboard visualization contract MUST accept additive extension without requiring the war-room layout itself to be rebuilt
