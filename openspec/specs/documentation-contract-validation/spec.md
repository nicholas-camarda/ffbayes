## Purpose
Defines the documentation validation contract for CLI commands, path references, dashboard payload fields, and glossary terminology.

## Requirements

### Requirement: Documented CLI commands SHALL be validated against the supported command surface
The repository MUST validate that user-facing `ffbayes` command examples in durable docs refer only to real supported CLI commands and supported argument names.

#### Scenario: Command example matches CLI parser
- **WHEN** a documented command example is checked in repo docs
- **THEN** automated validation MUST confirm that the command and documented flags exist in the current CLI/parser contract

#### Scenario: Stale command example is rejected
- **WHEN** a repo doc references a removed or renamed command or flag
- **THEN** automated validation MUST fail instead of allowing the documentation drift to merge silently

### Requirement: Documented path references SHALL be validated against canonical artifact policy
The repository MUST validate that durable docs describe canonical runtime, repo-local shortcut, staged Pages, and cloud publish paths in terms that match the current path contract and artifact layout.

#### Scenario: Canonical path reference remains aligned
- **WHEN** durable docs describe runtime, `dashboard/`, `site/`, or cloud artifact locations
- **THEN** automated validation MUST confirm those descriptions remain aligned with path constants and staging policy

#### Scenario: Drifted path reference is rejected
- **WHEN** durable docs describe a noncanonical or outdated path as authoritative
- **THEN** automated validation MUST fail instead of letting the misleading path guidance ship

### Requirement: Documentation-critical dashboard payload fields SHALL be contract-tested
The repository MUST validate the presence and structure of dashboard payload fields that the guide suite depends on for explanations and trust messaging.

#### Scenario: Guide-facing payload fields are present
- **WHEN** dashboard payload contract tests run
- **THEN** they MUST assert the presence of guide-facing fields such as runtime controls, analysis provenance, decision evidence, metric glossary, model overview, and Bayesian-vs-baseline summary content

#### Scenario: Guide-facing field removal is blocked
- **WHEN** an implementation change removes or renames a payload field that durable docs depend on
- **THEN** automated validation MUST fail unless the docs and contract are updated together

### Requirement: Staged Pages artifacts SHALL be validated as a documentation trust surface
The repository MUST validate that committed `site/` artifacts stay internally consistent with each other and preserve the trust signals the guide suite tells users to inspect.

#### Scenario: Staged payload and provenance are synchronized
- **WHEN** committed `site/dashboard_payload.json` and `site/publish_provenance.json` are validated
- **THEN** automated validation MUST confirm their publish provenance and freshness semantics are internally consistent

#### Scenario: Staged trust surface drifts
- **WHEN** committed `site/` artifacts diverge in provenance or synchronization semantics
- **THEN** automated validation MUST fail instead of allowing documentation to describe a trust surface that the repo snapshot does not actually provide

### Requirement: Documentation terminology SHALL remain aligned with current glossary semantics
The repository MUST validate that durable documentation uses the same canonical metric and trust-surface terminology that the current dashboard payload exposes.

#### Scenario: Guide terminology matches dashboard glossary
- **WHEN** durable docs describe key dashboard metrics or trust surfaces
- **THEN** automated validation or review checks MUST ensure those terms align with the canonical glossary and model-overview language rather than inventing conflicting names

#### Scenario: Audience-adapted wording preserves canonical terms
- **WHEN** durable docs adapt glossary explanations for different audiences
- **THEN** validation MUST allow audience-specific explanatory wording as long as the canonical metric names and trust-surface terminology remain aligned

#### Scenario: Conflicting terminology is blocked
- **WHEN** durable docs introduce a conflicting label for an existing dashboard metric or trust surface
- **THEN** the change MUST fail validation or review rather than allowing two competing names for the same concept

#### Scenario: Guide structure exposes audience and trust framing
- **WHEN** the durable guide suite is validated
- **THEN** each guide MUST include its required audience/scope framing and trust or interpretation-limit framing so those conventions do not drift away over time

#### Scenario: Commands and paths are not presented without context
- **WHEN** the durable docs list workflow commands or artifact paths
- **THEN** validation MUST confirm the surrounding guide structure identifies their purpose and authority category rather than presenting them as unlabeled literals

#### Scenario: Validation stays structural rather than prose-exact
- **WHEN** documentation contract validation runs
- **THEN** it MUST focus on structure, command/path correctness, required sections, and terminology alignment rather than enforcing exact explanatory prose
