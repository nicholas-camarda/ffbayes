## ADDED Requirements

### Requirement: The supported player forecast stack SHALL emit rolling holdout accuracy and calibration artifacts
The supported `pre_draft` modeling path MUST emit deterministic validation artifacts covering rolling holdout forecast accuracy, ranking quality, and calibration for the supported player forecast stack.

#### Scenario: Holdout artifacts include forecast and ranking quality
- **WHEN** the supported player forecast validation runs across eligible holdout seasons
- **THEN** it MUST report at least MAE or RMSE, rank correlation, top-k candidate quality, and interval or predictive calibration summaries

#### Scenario: Validation is attributable to the supported production path
- **WHEN** validation artifacts are generated for dashboard evidence or operator review
- **THEN** they MUST describe the supported player forecast contract rather than an unrelated hybrid or deprecated analysis lane

### Requirement: Validation SHALL include interpretable diagnostic slices
The supported validation artifacts MUST include slice-level reporting that helps the maintainer assess where the model is working well or poorly.

#### Scenario: Rookie and veteran slices are reported
- **WHEN** validation artifacts are produced
- **THEN** they MUST distinguish rookie and veteran performance so sparse-history behavior can be inspected directly

#### Scenario: Position slices are reported
- **WHEN** validation artifacts are produced
- **THEN** they MUST report at least QB, RB, WR, and TE slices when enough observations exist

#### Scenario: Rate and availability diagnostics are separately inspectable
- **WHEN** the two-part model emits validation artifacts
- **THEN** the artifacts MUST make the scoring-rate and availability components separately inspectable rather than collapsing all errors into one unlabeled aggregate

### Requirement: A small pre-draft stress fixture SHALL exercise the supported end-to-end path
The repository MUST include a small but interpretable fixture-driven test path that exercises the supported player forecast stack, draft strategy generation, dashboard payload generation, dashboard smoke path, and Pages staging path.

#### Scenario: Fixture covers edge cases
- **WHEN** the fixture dataset is defined
- **THEN** it MUST cover at least one rookie with draft-capital and combine context, one player with team change, one player with injury-driven availability instability, and multiple fantasy positions

#### Scenario: Fixture archetypes stay interpretable
- **WHEN** the stress fixture is reviewed or extended
- **THEN** it SHOULD preserve clear draft-day archetypes such as a stable QB baseline, a team-changing RB or similar veteran role-change example, a rookie WR with prior inputs, a stable veteran WR comparison point, and an availability-risk TE or similar missed-games example

#### Scenario: Fixture is fast enough for routine use
- **WHEN** the fixture-driven stress path runs in local or CI verification
- **THEN** it MUST complete materially faster than a full production pipeline while still producing interpretable draft artifacts and evidence

#### Scenario: Fixture drives dashboard smoke
- **WHEN** the fixture-driven stress path completes
- **THEN** the resulting artifacts MUST be usable by the dashboard smoke test without requiring production-size runtime data

### Requirement: Pages staging and local dashboard derivations SHALL be exercised by the stress harness
The stress harness MUST validate both runtime artifact generation and derived dashboard surfaces.

#### Scenario: Runtime, dashboard shortcut, and Pages copies stay aligned
- **WHEN** the fixture-driven stress path regenerates the canonical runtime dashboard payload and HTML
- **THEN** the derived repo-local `dashboard/` surface and staged `site/` surface MUST preserve the supported evidence semantics and valid payload structure

#### Scenario: Staged payload remains standards-compliant
- **WHEN** the stress harness validates staged or derived dashboard payloads
- **THEN** those payloads MUST parse as strict JSON and MUST expose the evidence contract expected by the supported dashboard

### Requirement: Final implementation review SHALL verify model, docs, and stress coverage together
The change MUST end with explicit specialist review checks that verify the implemented system matches the supported design, documentation, and stress-test expectations.

#### Scenario: Final implementation audit is required
- **WHEN** implementation work for this change is complete
- **THEN** a final review checklist MUST include implementation-validity review, documentation-consistency review, and robustness or stress-harness review before the change is considered complete
