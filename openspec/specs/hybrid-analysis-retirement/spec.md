## Purpose
Defines the production contract for retiring the hybrid Monte Carlo plus random-forest analysis path.

## Requirements

### Requirement: The hybrid Monte Carlo plus random-forest analysis path SHALL be removed completely
The repository MUST remove the hybrid Monte Carlo plus random-forest uncertainty path from the supported `pre_draft` workflow, runtime artifacts, CLI surfaces, and documentation. The system MUST NOT retain that path as a fallback, compatibility toggle, or silent rescue branch.

#### Scenario: Supported pipeline no longer runs hybrid step
- **WHEN** the supported `ffbayes pre-draft` workflow executes
- **THEN** it MUST NOT depend on the retired hybrid Monte Carlo plus random-forest analysis step to produce the draft board, evidence payload, or staged dashboard surfaces

#### Scenario: No fallback branch remains
- **WHEN** the supported pipeline is missing a hybrid artifact that used to exist
- **THEN** the system MUST continue through the single supported player-model path rather than invoking legacy hybrid logic

### Requirement: Hybrid-facing modules and entrypoints SHALL not survive as production surfaces
Hybrid integration modules, hybrid draft strategy surfaces, and hybrid-facing CLI exposure MUST be removed or rewritten against the supported player-forecast contract. Names or files that remain after the change MUST correspond to the supported system and not to a retired hybrid lane.

#### Scenario: Retired CLI exposure is gone
- **WHEN** a maintainer inspects the unified CLI and published script entrypoints
- **THEN** hybrid-specific commands, aliases, or help text MUST NOT describe the retired Monte Carlo plus random-forest path as a supported analysis surface

#### Scenario: Retired draft helpers are not left behind as compatibility code
- **WHEN** maintainers inspect draft strategy and analysis modules after the change
- **THEN** hybrid integration helpers, hybrid risk-ranking code, and other hybrid-specific production surfaces MUST NOT remain as dormant compatibility code

### Requirement: Runtime outputs SHALL not preserve retired hybrid artifacts as first-class results
The runtime artifact tree, Pages staging flow, and documentation MUST stop treating hybrid outputs as supported artifacts.

#### Scenario: Runtime artifact tree drops hybrid contract
- **WHEN** the supported pre-draft pipeline writes canonical runtime outputs
- **THEN** it MUST NOT emit or rely on hybrid model summaries as part of the supported dashboard/evidence contract

#### Scenario: Documentation does not direct operators to hybrid outputs
- **WHEN** a user reads README, technical docs, or operator guidance
- **THEN** the docs MUST NOT instruct them to inspect, compare, or trust retired hybrid artifacts as part of the supported workflow

### Requirement: Retirement SHALL be testable
The repository MUST include deterministic checks that detect accidental reintroduction of the retired hybrid lane.

#### Scenario: Tests fail if hybrid path is reintroduced
- **WHEN** automated verification inspects pipeline config, CLI wiring, or supported documentation
- **THEN** those checks MUST fail if the retired hybrid Monte Carlo plus random-forest path is reintroduced into the supported workflow or operator guidance
