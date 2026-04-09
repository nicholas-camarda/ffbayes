## Context

FFBayes already has a supported `pre-draft` operator workflow, a canonical runtime artifact tree, and a richer draft dashboard payload that includes `decision_evidence`, `metric_glossary`, `model_overview`, freshness metadata, and publish provenance. The current durable docs do not line up cleanly with that system. In particular, the repo-level docs mix supported and optional outputs, describe parts of the implemented model inaccurately, and leave the dashboard interaction model mostly implicit in generated HTML.

This change is cross-cutting because it spans repo documentation, dashboard-facing payload semantics, staged `site/` output, and automated validation. It also touches trust-sensitive surfaces: the difference between canonical runtime artifacts and derived copies, the meaning of the current board score, and how strongly the system claims internal evidence supports recommendations.

There is also an active change, `add-war-room-decision-visuals`, that may add new timing, scarcity, and comparative decision visuals to the dashboard. This documentation change should not redesign that feature, but it does need to absorb any shipped user-facing semantics so the guide suite does not immediately drift behind the implementation.

The repo already has strong ingredients we can reuse:
- `src/ffbayes/draft_strategy/draft_decision_system.py` owns the current glossary, model overview text, decision-evidence contract, and dashboard UI sections.
- `src/ffbayes/utils/path_constants.py` owns canonical runtime, repo, Pages, and cloud path policy.
- `src/ffbayes/cli.py`, `src/ffbayes/refresh_dashboard.py`, `src/ffbayes/publish_pages.py`, and `src/ffbayes/analysis/draft_retrospective.py` define the supported operator command surface.
- Existing smoke and publish tests already validate many artifact/UI behaviors, but they do not yet treat docs as a contract.

## Goals / Non-Goals

**Goals:**
- Publish a durable guide suite for the supported `pre-draft` workflow, the dashboard operator flow, layperson interpretation, statistician-facing methods, and runtime/path lineage.
- Consolidate overlapping technical documentation by turning `docs/TECHNICAL_DEEP_DIVE.md` into the single authoritative technical/methods guide rather than maintaining two overlapping math documents.
- Align documented model claims, targets, intervals, baselines, and trust boundaries with the implemented draft engine and emitted payload fields.
- Teach unclear terms by breaking them into their components and explaining what is happening, not only naming the metric or process.
- Add automated checks that catch drift between docs and the CLI, path contract, dashboard payload contract, and staged `site/` bundle.
- Remove or clearly label legacy or optional output categories so the docs describe the supported workflow truthfully.

**Non-Goals:**
- Re-architect the draft engine, replace the current player model, or migrate away from the `pre_draft` phase.
- Build a full doc-generation system or move guides outside repo-tracked Markdown.
- Introduce new runtime artifact families unless they are needed to support already-emitted dashboard semantics.
- Turn GitHub Pages into the primary local operator surface; canonical runtime artifacts remain authoritative.
- Expand the amount of detail already shown in the dashboard UI as part of this change; this pass is documentation-first.

## Decisions

### Decision: The guide suite will live as first-class Markdown under `docs/`
The change will add durable guides under `docs/` rather than relying on README expansion or embedded dashboard copy alone.

Rationale:
- The repo currently lacks audience-specific documents.
- Markdown docs can be reviewed, diffed, linked from `docs/README.md`, and validated in tests.
- This keeps the operator surface local to the repository rather than hidden only in generated HTML.
- This also gives the repo one place to incorporate any shipped war-room visual semantics from the active visualization change.

Alternatives considered:
- Expand only `README.md`.
  Rejected because one file would conflate operator setup, model methods, interpretation, and artifact lineage.
- Treat dashboard copy as the only documentation.
  Rejected because generated HTML is a downstream surface, not a durable reviewable source.

### Decision: Documentation terms will be anchored to code-owned trust surfaces
User-facing explanations for metrics and model interpretation will be derived from the current payload contract in `draft_decision_system.py`, especially `metric_glossary`, `model_overview`, `decision_evidence`, and canonical command/path helpers.

Rationale:
- The audit found that the best current interpretation language already exists in code, while repo docs drifted.
- Reusing code-owned terminology reduces the chance that a guide invents names or overclaims semantics.

Alternatives considered:
- Maintain freehand documentation text disconnected from payload fields.
  Rejected because that is the drift path the repo is already on.
- Generate all docs directly from the payload.
  Rejected because guides need audience-specific structure and narrative, not only raw payload text.

### Decision: The existing technical deep dive will become the single authoritative technical/methods guide
The repo will avoid overlapping technical documentation by updating and repurposing `docs/TECHNICAL_DEEP_DIVE.md` into the statistician-facing methods guide instead of maintaining two parallel mathematical references.

Rationale:
- The user wants one authoritative technical explanation rather than overlapping deep-dive and methods documents.
- The current deep dive already has the right place in the docs structure, but its content needs reconciliation with the implemented system.

Alternatives considered:
- Add a second separate statistician guide.
  Rejected because it would duplicate and fragment technical documentation.

### Decision: The methods guide will explicitly separate implemented, conceptual, optional, and deprecated paths
The authoritative technical/methods guide will classify model/data content into four buckets:
- implemented current draft board behavior
- conceptual intuition
- optional analyses reachable by separate commands
- deprecated or compatibility-only surfaces

Rationale:
- The current deep dive mixes these categories and makes the implemented draft board look like a different model.
- This classification also lets the docs distinguish supported `pre-draft` outputs from optional analyses such as standalone Monte Carlo or comparison runs.

Alternatives considered:
- Keep one unified “how it works” narrative.
  Rejected because it blurs the implemented draft system and optional research utilities.

### Decision: Documentation will explain the existing dashboard detail rather than expanding the dashboard in this change
The dashboard will keep its current level of detail. This change will document the existing evidence, glossary, provenance, and trust surfaces rather than introducing a larger or more opinionated dashboard experience.

Rationale:
- The current dashboard already contains the level of detail the user wants.
- The main gap is durable explanation and interpretation, not the absence of UI detail.

Alternatives considered:
- Tighten or expand dashboard UI copy in parallel with the doc pass.
  Deferred because the requested scope is documentation first.

### Decision: Durable guides will follow a shared documentation convention set
The guide suite will follow a common documentation convention set aligned with the documentation-wizard review lane: each guide should declare its audience, scope, and trust level up front; use stable section patterns across guides; preserve canonical terminology; and pair commands or paths with their purpose and authority level.

Rationale:
- The user wants these docs to follow the documentation-wizard conventions rather than becoming a one-off rewrite.
- Shared structure makes it easier for readers to move between operator, lay, and technical guides without relearning the navigation each time.
- Stable conventions also give contract tests something concrete to validate without freezing prose.

Alternatives considered:
- Let each guide evolve its own structure and naming style.
  Rejected because that makes drift and audience confusion more likely.

### Decision: Docs drift will be caught through lightweight contract tests
The implementation will add pytest-level checks for:
- documented CLI commands
- documented canonical paths and artifact categories
- documentation-critical payload fields
- `site/dashboard_payload.json` and `site/publish_provenance.json` consistency
- terminology alignment between docs and code-owned glossary/model overview sources where practical

Rationale:
- Existing tests cover HTML/payload behavior but not the documentation interface.
- Lightweight contract checks fit the repo’s current tooling and can run in CI without introducing a new doc toolchain.

Alternatives considered:
- Rely on manual doc review.
  Rejected because the current drift happened under manual review.

## Risks / Trade-offs

- [Risk: Methods guide becomes another stale narrative] → Mitigation: anchor guide claims to current code modules, use docs contract tests, and classify optional/deprecated paths explicitly.
- [Risk: Dashboard copy and repo docs diverge again] → Mitigation: treat `metric_glossary`, `model_overview`, and `decision_evidence` as guide-facing sources of truth and add payload contract coverage.
- [Risk: Scope expands into a broad documentation rewrite] → Mitigation: limit the change to supported `pre-draft` workflow truth, trust surfaces, and documented optional analyses that are still intentionally exposed.
- [Risk: Evidence UI changes require staged `site/` refresh and create temporary repo drift] → Mitigation: keep `refresh-dashboard --check --json` and `publish-pages` in the workflow, and document the canonical runtime vs staged Pages distinction clearly.
- [Risk: Tests become brittle against prose edits] → Mitigation: validate commands, paths, fields, and required phrases/sections rather than full-document exact text.

## Migration Plan

1. Add the new guide files and update `docs/README.md` to index them by audience and workflow.
2. Reconcile `README.md`, `docs/TECHNICAL_DEEP_DIVE.md`, and `docs/OUTPUT_EXAMPLES.md` so supported/optional/deprecated surfaces are classified correctly.
3. Reconcile guide-facing payload references, existing trust-surface explanations, and any shipped war-room visualization behavior with the docs, then refresh derived dashboard HTML and staged `site/` copies only if implementation changes affect guide-facing emitted content.
4. Add docs contract tests and staged-surface consistency tests.
5. Run pytest and dashboard smoke coverage, then use `refresh-dashboard --check --json` to verify derived surfaces before restaging Pages if needed.

Rollback:
- Repo docs can be reverted independently if needed.
- Validation-only changes can be reverted without altering the broader pipeline or artifact naming.

## Open Questions

No open questions remain for the initial documentation structure. The guide suite will adapt glossary language for each audience while preserving canonical terminology, and it may include minimal payload snippets when they materially improve understanding without turning the docs into raw schema dumps.
