## ADDED Requirements

### Requirement: Retrospective command SHALL operate on finalized draft artifacts plus realized season outcomes
The retrospective feedback loop MUST read finalized draft snapshots, pick receipts, exported dashboard artifacts, and realized season outcome data for the drafted season, rather than rerunning the full draft-model generation pipeline.

#### Scenario: Existing finalized artifacts are available
- **WHEN** a finalized draft snapshot, its associated receipts, and matching season outcome data exist for a season
- **THEN** the retrospective command MUST generate a report from those artifacts without rebuilding the draft board model

#### Scenario: Finalized artifacts are missing or incomplete
- **WHEN** one or more required finalized draft artifacts are missing
- **THEN** the command MUST report an explicit unavailable state and identify the missing inputs

#### Scenario: Season outcome data is unavailable
- **WHEN** the drafted season's realized player outcomes are missing, incomplete, or schema-incompatible
- **THEN** the command MUST fail or degrade explicitly and identify that outcome-grounded evaluation could not be completed

### Requirement: Finalized draft bundles SHALL have a canonical runtime landing zone
The retrospective feedback loop MUST define a year-scoped runtime location for finalized draft bundles under the existing `draft_strategy` artifact tree, and the system MUST provide a cheap way to ingest browser-downloaded finalized artifacts into that canonical location.

#### Scenario: Browser downloads are ingested into the canonical runtime folder
- **WHEN** an operator imports a finalized draft bundle after the local dashboard download completes
- **THEN** the system MUST store the bundle under the year-scoped `draft_strategy/finalized_drafts/` runtime directory without rerunning draft modeling

#### Scenario: Browser save location is not canonical
- **WHEN** the browser initially saves finalized artifacts into `Downloads/` or another user-managed folder
- **THEN** the system MUST treat that location as a transient source and MUST rely on the canonical runtime import location for routine retrospective discovery

#### Scenario: Retrospective discovery uses canonical finalized drafts
- **WHEN** imported finalized draft JSON files exist in the canonical runtime folder for the requested season
- **THEN** `ffbayes draft-retrospective` MUST auto-discover those JSON files before requiring explicit `--finalized-json` paths

### Requirement: Retrospective outputs SHALL prioritize realized roster performance evaluation
The retrospective feedback loop MUST treat realized roster outcomes as the primary learning surface and summarize how the drafted roster actually performed relative to what the board expected.

#### Scenario: Outcome-grounded evaluation is available
- **WHEN** realized season outcomes are available for drafted players
- **THEN** the retrospective report MUST include expected-versus-realized roster metrics, realized starter or lineup performance, and player-level or archetype-level hit/miss summaries

#### Scenario: Wait policy can be audited against realized outcomes
- **WHEN** the finalized artifacts preserve wait or pass context and the corresponding players have realized outcomes
- **THEN** the retrospective report MUST summarize realized pass regret or equivalent wait-policy calibration metrics

### Requirement: Retrospective outputs SHALL summarize model-following and pivots as secondary audit context
The retrospective feedback loop MUST summarize how often draft decisions followed the model and how often they pivoted when the receipts support that audit trail, but those metrics MUST remain secondary to realized outcome evaluation.

#### Scenario: Follow and pivot rates are available
- **WHEN** finalized pick receipts include model-follow and pivot information
- **THEN** the retrospective report MUST include follow rate, pivot rate, and a pick-level summary of decision outcomes

#### Scenario: Model-follow data is incomplete
- **WHEN** some pick receipts do not contain complete follow or pivot metadata
- **THEN** the retrospective report MUST mark the affected metrics as partial or degraded rather than inventing values

### Requirement: Retrospective outputs SHALL evaluate roster construction and decision quality over time
The retrospective feedback loop MUST summarize roster composition and decision-quality trends using finalized artifacts and realized season outcomes over time.

#### Scenario: Single-season review
- **WHEN** the command is run for one season
- **THEN** the report MUST summarize roster construction outcomes and outcome-grounded decision-quality metrics for that season

#### Scenario: Multi-season review
- **WHEN** the command is run across multiple finalized draft seasons
- **THEN** the report MUST include an aggregate trend view that compares seasons over time

### Requirement: Retrospective reporting SHALL emit canonical JSON plus a companion HTML report
The retrospective feedback loop MUST emit a machine-readable JSON artifact as the canonical report contract and an HTML artifact as a derived review surface.

#### Scenario: Retrospective generation succeeds
- **WHEN** the command completes successfully
- **THEN** it MUST write both retrospective JSON and retrospective HTML artifacts for the analyzed season scope

#### Scenario: Automation consumes retrospective output
- **WHEN** tests or future tooling inspect retrospective results
- **THEN** the JSON artifact MUST be sufficient to recover provenance, degraded-state markers, and computed metrics without parsing the HTML presentation layer

### Requirement: Retrospective outputs SHALL remain runtime-local in the initial release
The retrospective feedback loop MUST write its artifacts under the existing runtime draft-strategy artifact tree and MUST NOT treat GitHub Pages staging as part of the base workflow.

#### Scenario: Runtime-local retrospective generation
- **WHEN** the retrospective command is run successfully
- **THEN** it MUST stage outputs under the year-scoped runtime artifact directory alongside the finalized draft bundle

#### Scenario: Public Pages bundle remains unchanged
- **WHEN** the first retrospective capability is implemented
- **THEN** it MUST NOT require writing retrospective artifacts into `site/` or publishing them through `publish-pages`

### Requirement: Retrospective artifacts SHALL preserve provenance
The retrospective feedback loop MUST record which source artifacts were analyzed, when the retrospective was generated, and which season or seasons were included.

#### Scenario: Provenance is available
- **WHEN** the report is generated successfully
- **THEN** the output MUST include source paths, generation timestamp, and season coverage metadata

#### Scenario: Source version is unsupported
- **WHEN** a finalized artifact uses an unsupported schema version
- **THEN** the command MUST fail or degrade explicitly with a reason that is visible in the report output
