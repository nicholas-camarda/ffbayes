## Purpose
Defines visualization surface governance for supported output structure, authority boundaries, and local runtime input layout.

## Requirements

### Requirement: Supported output structure SHALL remain minimal and authority-scoped
The supported `pre_draft` workflow MUST emit only a small, declared set of authoritative and derived artifact families for the redesigned draft engine. New model, dashboard, and validation outputs MUST fit the declared structure rather than creating ad hoc sibling directories or duplicate truth surfaces.

#### Scenario: Canonical runtime draft artifacts remain centralized
- **WHEN** the redesigned draft engine emits supported runtime artifacts for a given year
- **THEN** the canonical dashboard HTML, canonical dashboard payload, canonical workbook, and canonical player-forecast outputs MUST live under `seasons/<year>/draft_strategy/` and its declared subdirectories rather than in scattered feature-specific directories

#### Scenario: Validation and stress summaries remain under diagnostics
- **WHEN** the redesigned model emits holdout-validation, calibration, or stress-fixture summaries
- **THEN** those summaries MUST live under `seasons/<year>/diagnostics/validation/` rather than creating parallel top-level artifact families beside the canonical draft outputs

#### Scenario: Derived surfaces do not accumulate extra artifact families
- **WHEN** repo-local `dashboard/` or staged `site/` copies are regenerated from the canonical runtime dashboard
- **THEN** they MUST remain shallow derived copies of the canonical HTML and payload pair and MUST NOT accumulate extra diagnostics, model dumps, or historical artifact families

#### Scenario: Ad hoc model-output families are rejected
- **WHEN** a redesign step attempts to introduce ad hoc output families such as `hybrid_*`, `experimental_*`, duplicate player-forecast dumps, or duplicate canonical dashboard payloads for the same year
- **THEN** the supported workflow MUST reject that structure in favor of the declared canonical directories and filenames

### Requirement: Local runtime inputs SHALL use the `inputs/` tree and legacy runtime trees SHALL be removed
The supported local runtime contract MUST use `inputs/raw/` and `inputs/processed/` as the canonical working input tree. The legacy runtime `data/` and `datasets/` trees MUST NOT remain as parallel supported working roots after migration.

#### Scenario: Canonical local input root is explicit
- **WHEN** collection, preprocessing, or unified-dataset generation writes local runtime inputs
- **THEN** those outputs MUST be written under the canonical `inputs/` tree rather than under a parallel local `data/` or `datasets/` root

#### Scenario: Cloud mirror keeps the published `data/` name
- **WHEN** selected runtime inputs are mirrored to the maintainer-configured cloud home
- **THEN** the cloud mirror MAY continue using cloud `data/` and dated `Analysis/` roots without implying that cloud `data/` is the local working source of truth

#### Scenario: Parallel local roots are rejected
- **WHEN** the migrated runtime layout is verified
- **THEN** supported code, docs, and tests MUST NOT require parallel local `inputs/` and `data/` trees or a surviving legacy local `datasets/` tree
