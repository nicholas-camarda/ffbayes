## 1. Reconcile Existing Documentation

- [ ] 1.1 Audit `README.md`, `docs/README.md`, `docs/TECHNICAL_DEEP_DIVE.md`, and `docs/OUTPUT_EXAMPLES.md` against the implemented CLI, path contract, draft engine terminology, and any shipped behavior from `add-war-room-decision-visuals`.
- [ ] 1.2 Fix broken or stale command examples, including `refresh-dashboard` argument names and any commands currently described with the wrong purpose.
- [ ] 1.3 Remove, reclassify, or clearly mark optional and legacy output examples that are not emitted by the supported `ffbayes pre-draft` workflow.

## 2. Add The Guide Suite

- [ ] 2.1 Add a dashboard operator guide under `docs/` covering canonical runtime artifacts, repo-local shortcut usage, staged Pages behavior, finalize flow, post-draft ingest/retrospective steps, and any shipped war-room timing/scarcity/comparative visuals.
- [ ] 2.2 Add a layperson guide under `docs/` that explains the workflow from data collection to final draft use, including what key metrics do and do not mean.
- [ ] 2.3 Repurpose `docs/TECHNICAL_DEEP_DIVE.md` into the single authoritative statistician-facing methods guide that documents the implemented player model, decision-policy layer, targets, baselines, interval semantics, and internal-validation scope.
- [ ] 2.4 Add a metric reference and a path/data-lineage guide under `docs/`, using audience-adapted explanations and minimal payload snippets where they materially improve understanding, then update `docs/README.md` so all new guides are indexed by audience and purpose.
- [ ] 2.5 Apply the shared documentation conventions across the guide suite: audience/scope/trust framing in the first screenful, stable section structure, canonical terminology, and purpose-plus-authority labeling for commands and paths.

## 3. Align Existing Trust Surfaces With The Docs

- [ ] 3.1 Ensure guide-facing payload fields such as `metric_glossary`, `model_overview`, `decision_evidence`, and provenance/evidence summaries remain internally consistent with the docs without expanding the dashboard detail level.
- [ ] 3.2 Update docs and references so dashboard evidence, glossary, and provenance sections are explained using the existing trust-surface terminology and current behavior.
- [ ] 3.3 Defer screenshots and richer dashboard examples until a follow-up change after the guide structure is stable.

## 4. Add Documentation Contract Validation

- [ ] 4.1 Add pytest coverage that validates documented `ffbayes` commands and flags against the current CLI/parser contract.
- [ ] 4.2 Add pytest coverage that validates documented canonical paths and output categories against `path_constants.py`, artifact staging behavior, and supported workflow boundaries.
- [ ] 4.3 Add payload/schema tests for documentation-critical dashboard fields and Pages provenance consistency, including committed `site/dashboard_payload.json` and `site/publish_provenance.json`.
- [ ] 4.4 Add or extend tests so the guide-dependent evidence, glossary, provenance, and local-vs-staged behavior remain covered using structure and terminology checks rather than exact prose matching.
- [ ] 4.5 Add structural checks that the guide suite keeps the shared documentation conventions intact, including audience/scope/trust framing and contextual labeling for commands and paths.

## 5. Verify And Restage

- [ ] 5.1 Run the relevant pytest coverage for docs contracts, dashboard payload/staging behavior, and touched trust-surface logic.
- [ ] 5.2 Run the dashboard smoke path and any freshness/drift checks needed to confirm derived surfaces are synchronized with authoritative runtime artifacts.
- [ ] 5.3 Restage `site/` if guide-facing dashboard output changed, then verify the committed Pages bundle matches the documented trust surface.
