## MODIFIED Requirements

### Requirement: Dashboard payload SHALL expose a normalized war-room visualization contract
The live dashboard payload MUST expose a normalized `war_room_visuals` contract for interactive draft-decision visuals so the UI depends on stable semantic fields rather than raw model-formula internals. The contract MUST be driven by the supported player-model and draft-decision stack and MUST NOT depend on retired hybrid Monte Carlo or random-forest semantics.

#### Scenario: Payload includes semantic visualization sections
- **WHEN** `ffbayes draft-strategy` generates the canonical runtime dashboard payload
- **THEN** the payload MUST include normalized sections for timing decisions, positional scarcity, and contextual-versus-baseline explanation with semantic values and user-facing labels derived from the supported model stack

#### Scenario: Visualization section is unavailable
- **WHEN** the dashboard cannot derive one of the normalized visualization sections from the available draft artifacts
- **THEN** the payload MUST expose that section as unavailable or degraded with an explicit reason instead of silently omitting it

#### Scenario: Visualization contract evolves additively
- **WHEN** later model upgrades add or refine war-room visualization semantics
- **THEN** the normalized payload contract MUST evolve through additive or explicitly versioned semantic fields rather than forcing existing dashboard surfaces to bind to volatile raw model internals

#### Scenario: Supported-model provenance is visible to visualization builders
- **WHEN** visualization sections are generated from the runtime artifacts
- **THEN** the payload MUST be able to identify that those semantics came from the supported player-model and decision-policy contract rather than a retired hybrid analysis lane

### Requirement: War room dashboard SHALL render a timing ladder as a decision aid
The live draft war room MUST render a compact `wait-vs-pick` timing visualization that helps operators compare immediate selection against waiting for the next pick window, without relying on overlapping scatter marks.

#### Scenario: Timing ladder renders relevant current candidates
- **WHEN** the dashboard loads a payload with timing visualization data
- **THEN** it MUST render a timing-oriented decision view that distinguishes take-now and waitable candidates using survival, regret, or equivalent semantic timing values in a readable per-candidate ladder or equivalent non-overlapping presentation

#### Scenario: Frontier reacts to live board state
- **WHEN** local dashboard state changes through `taken`, `mine`, queue, current pick, next pick, or scoring-preset updates
- **THEN** the timing visualization MUST update to reflect the new local decision context

#### Scenario: Timing semantics remain stable after model redesign
- **WHEN** the underlying player-model architecture changes from the current mixed stack to the supported season-total player-model contract
- **THEN** the timing visualization MUST continue to consume stable semantic timing fields rather than binding directly to estimator-specific implementation details

### Requirement: War room dashboard SHALL render positional scarcity cliffs
The live draft war room MUST render a positional cliff view that highlights where a position group drops off materially and which candidates sit just before those cliffs.

#### Scenario: Cliff map highlights sharp drop-offs
- **WHEN** the dashboard has normalized positional scarcity data
- **THEN** it MUST surface the strongest current drop-offs by position instead of leaving scarcity encoded only in tables or prose

#### Scenario: Cliff view uses ordered tier-break strips
- **WHEN** the dashboard renders positional scarcity for the active draft state
- **THEN** it MUST show each relevant position as an ordered player strip with a clearly emphasized primary break, rather than as a text-heavy annotation cluster

#### Scenario: Cliff view is collapsed near the board by default
- **WHEN** the dashboard first renders the positional cliff view
- **THEN** it MUST appear in a collapsed-by-default panel directly above or adjacent to the player board, with a concise closed-state summary of the strongest current breaks

#### Scenario: Cliff map supports board exploration
- **WHEN** the operator filters positions or selects a player
- **THEN** the positional cliff view MUST support corresponding emphasis, filtering, or highlighting rather than remaining disconnected from the board

#### Scenario: Cliff map defaults to relevant positions
- **WHEN** the dashboard first renders the positional cliff view
- **THEN** it MUST default to the positions most relevant to the active recommendation lanes or selected-player context rather than showing the full position universe by default

### Requirement: Comparative explanation SHALL remain extensible across model upgrades
The live dashboard MUST support contextual-versus-baseline comparison visuals without hardcoding the visual contract to today’s exact baseline implementation or score formula.

#### Scenario: Current baseline is simple VOR proxy
- **WHEN** the current model export uses the simple VOR proxy as the baseline comparison
- **THEN** the dashboard MUST label and render the comparative explainer using that baseline name while still sourcing it from the normalized visualization contract

#### Scenario: Future baseline changes
- **WHEN** a later model upgrade replaces or augments the current baseline comparison
- **THEN** the comparative visualization contract MUST remain valid through updated semantic labels and values without requiring the UI layout itself to be rewritten

#### Scenario: Comparative explanation can surface rate and availability context
- **WHEN** the supported player-model contract exposes separate rate and availability context for a selected player
- **THEN** the inspector-linked comparative explanation MAY surface those supported semantics without requiring a new standalone dashboard surface

### Requirement: Visuals SHALL use progressive disclosure inside the war room
The new decision visuals MUST fit inside the existing war-room dashboard as compact, action-supporting components rather than as a separate analytics surface.

#### Scenario: Primary visuals stay near recommendations
- **WHEN** the dashboard renders the timing frontier or positional cliff view
- **THEN** those visuals MUST appear in the main war-room flow near recommendations or board controls rather than on a detached report page

#### Scenario: Secondary explanation stays collapsible
- **WHEN** the dashboard renders the contextual-versus-baseline explainer
- **THEN** it MUST appear through inspector- or evidence-linked progressive disclosure instead of permanently occupying equal visual weight with the primary recommendation controls

#### Scenario: Comparative explainer is inspector-first
- **WHEN** the first version of the contextual-versus-baseline explainer is rendered
- **THEN** it MUST live primarily inside the selected-player inspector rather than as a persistent standalone dashboard panel
