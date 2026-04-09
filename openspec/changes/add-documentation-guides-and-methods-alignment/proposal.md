## Why

The repo now has a usable pre-draft workflow and a richer dashboard trust surface, but the durable documentation does not faithfully explain the implemented draft engine, the operator workflow, or the interpretation limits of the outputs. This creates a trust problem now because the current docs can misstate the math, overstate certainty, and leave both technical and non-technical users without a reliable guide.

## What Changes

- Add a durable documentation guide suite for the supported pre-draft workflow, including a dashboard operator guide, a layperson interpretation guide, a statistician-facing methods guide, a metric reference, and a data-lineage/path guide.
- Merge the current technical deep dive into one authoritative technical/methods guide so overlapping mathematical documentation is reduced instead of expanded.
- Align the documented model and metric explanations with the implemented draft engine, including the empirical-Bayes player model, decision-policy scoring, freshness/provenance surfaces, and retrospective workflow.
- Incorporate any shipped war-room visualization changes, including timing, scarcity, and comparative decision visuals, into the operator-facing and interpretation documentation rather than leaving those semantics documented only in a feature-specific change.
- Add documentation contract checks so user-facing commands, paths, payload fields, and staged `site/` artifacts cannot drift silently away from the docs.
- Remove or clearly mark misleading examples and legacy/optional output categories from current docs when they are not part of the supported `ffbayes pre-draft` workflow.

## Capabilities

### New Capabilities
- `documentation-guide-suite`: Provide authoritative user, operator, layperson, and statistician documentation for the supported pre-draft workflow, dashboard usage, result interpretation, and artifact lineage.
- `documentation-contract-validation`: Validate that documented commands, paths, payload fields, and staged Pages artifacts remain synchronized with the actual CLI, runtime outputs, and dashboard contract.

### Modified Capabilities
- `visualization-surface-governance`: Expand visualization governance so the authoritative-vs-derived surface distinctions and documented output categories remain aligned with the emitted runtime and Pages artifacts.

## Impact

- Affected docs: `README.md`, `docs/README.md`, `docs/TECHNICAL_DEEP_DIVE.md`, `docs/OUTPUT_EXAMPLES.md`, plus new docs under `docs/`.
- Affected dashboard/runtime contract: staged `site/` payload/HTML and related guide-facing payload fields already emitted by `src/ffbayes/draft_strategy/draft_decision_system.py`.
- Affected validation/tests: `tests/`, dashboard smoke coverage, `tests/test_publish_pages.py`, and new docs contract checks.
- Affected commands and systems: `ffbayes pre-draft`, `ffbayes draft-strategy`, `ffbayes refresh-dashboard`, `ffbayes publish-pages`, and `ffbayes draft-retrospective`.
- Coordination risk: this change must stay aligned with the active `add-war-room-decision-visuals` implementation so the dashboard operator docs and interpretation guides describe any shipped timing, scarcity, and comparative visuals accurately.
- Risk areas: path/source-of-truth messaging across runtime vs repo `dashboard/` vs repo `site/`, phase-scoped `pre_draft` artifact naming, and keeping Pages staging/documentation in sync with runtime artifacts.
