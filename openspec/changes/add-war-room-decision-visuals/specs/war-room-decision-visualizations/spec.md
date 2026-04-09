## ADDED Requirements

### Requirement: Dashboard payload SHALL expose a normalized war-room visualization contract
The live dashboard payload MUST expose a normalized `war_room_visuals` contract for interactive draft-decision visuals so the UI depends on stable semantic fields rather than raw model-formula internals.

#### Scenario: Payload includes semantic visualization sections
- **WHEN** `ffbayes draft-strategy` generates the canonical runtime dashboard payload
- **THEN** the payload MUST include normalized sections for timing decisions, positional scarcity, and contextual-versus-baseline explanation with semantic values and user-facing labels

#### Scenario: Visualization section is unavailable
- **WHEN** the dashboard cannot derive one of the normalized visualization sections from the available draft artifacts
- **THEN** the payload MUST expose that section as unavailable or degraded with an explicit reason instead of silently omitting it

#### Scenario: Visualization contract evolves additively
- **WHEN** later model upgrades add or refine war-room visualization semantics
- **THEN** the normalized payload contract MUST evolve through additive or explicitly versioned semantic fields rather than forcing existing dashboard surfaces to bind to volatile raw model internals

### Requirement: War room dashboard SHALL render a timing frontier as a decision aid
The live draft war room MUST render a compact `wait-vs-pick` timing visualization that helps operators compare immediate selection against waiting for the next pick window.

#### Scenario: Frontier renders relevant current candidates
- **WHEN** the dashboard loads a payload with timing visualization data
- **THEN** it MUST render a timing-oriented decision view that distinguishes take-now and waitable candidates using survival, regret, or equivalent semantic timing values

#### Scenario: Frontier reacts to live board state
- **WHEN** local dashboard state changes through `taken`, `mine`, queue, current pick, next pick, or scoring-preset updates
- **THEN** the timing visualization MUST update to reflect the new local decision context

### Requirement: War room dashboard SHALL render positional scarcity cliffs
The live draft war room MUST render a positional cliff view that highlights where a position group drops off materially and which candidates sit just before those cliffs.

#### Scenario: Cliff map highlights sharp drop-offs
- **WHEN** the dashboard has normalized positional scarcity data
- **THEN** it MUST surface the strongest current drop-offs by position instead of leaving scarcity encoded only in tables or prose

#### Scenario: Cliff view uses ordered tier-break strips
- **WHEN** the dashboard renders positional scarcity for the active draft state
- **THEN** it MUST show each relevant position as an ordered player strip with a clearly emphasized primary break, rather than as a text-heavy annotation cluster

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
