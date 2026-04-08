## Context

The visualization audit found that FFBayes has one actively maintained visualization product and several weaker or inconsistent surrounding surfaces.

The strongest surface is the live draft dashboard generated from `src/ffbayes/draft_strategy/draft_decision_system.py`, refreshed by `src/ffbayes/refresh_dashboard.py`, and staged into `site/` by `src/ffbayes/publish_pages.py`. That path already carries structured decision evidence, freshness state, provenance, local draft controls, and smoke coverage.

The surrounding visualization layer is less coherent:
- repo-local `dashboard/` artifacts are treated as convenience shortcuts in docs but currently function as a larger untracked artifact surface
- the repo-root local shortcut can drift away from the canonical runtime artifact pair and even carry synthetic placeholder content
- `site/index.html` can drift away from `site/dashboard_payload.json`
- static diagnostics and plot generators under `src/ffbayes/visualization/` include legacy standalone flows that overlap with the dashboard product without clearly owning a supported output surface
- docs promise more diagnostics categories than the current runtime and local mirrors actually emit

This is a cross-cutting change because it touches runtime artifact generation, repo-local convenience copies, staged Pages artifacts, documentation, tests, and legacy visualization module governance.

## Goals / Non-Goals

**Goals:**
- Define a clear authority model for runtime dashboard artifacts, repo-local shortcuts, staged Pages outputs, and supplemental diagnostics.
- Ensure the repo-root shortcut, runtime shortcut, and `site/` bundle are always recognizable as derived copies of a canonical dashboard artifact pair.
- Prevent placeholder or synthetic local dashboard payloads from masquerading as live draft outputs.
- Make visualization-facing freshness, evidence, and provenance states interpretable and non-contradictory.
- Align documented visualization categories with the outputs the system actually emits.
- Classify legacy visualization modules into active, compatibility-only, deprecated, or removable buckets before implementation proceeds, and treat unused manual-entrypoint-only surfaces as deprecated rather than preserving them as a supported class.
- Extend validation coverage to local shortcut integrity and visualization drift, not only staged Pages behavior.

**Non-Goals:**
- Do not migrate away from the supported `pre_draft` phase.
- Do not rename canonical dashboard artifact filenames or move them out of `runs/<year>/pre_draft/artifacts/draft_strategy/` or `site/`.
- Do not redesign the draft dashboard from scratch or replace its decision workflow.
- Do not promise that every historical static plot generator remains supported.
- Do not add new external visualization dependencies unless implementation later proves they are necessary.

## Decisions

1. **Adopt an explicit authority hierarchy for visualization surfaces.**
   The canonical local dashboard bundle remains the runtime artifact pair under `runs/<year>/pre_draft/artifacts/draft_strategy/`:
   - `draft_board_<year>.html`
   - `dashboard_payload_<year>.json`

   Repo-local `dashboard/` and runtime-root `dashboard/` become derived convenience copies only. `site/` remains the public staged derivative. Supplemental diagnostics remain non-authoritative and must not present themselves as the primary board.

   Alternatives considered:
   - Treat repo-local `dashboard/` as an equal source of truth. Rejected because it is untracked, user-local, and already demonstrated drift from the real runtime payload.
   - Treat `site/` as the only authoritative surface. Rejected because draft-day local use depends on runtime-local generation before Pages staging.

2. **Narrow repo-local `dashboard/` to a shortcut surface, not an artifact dump.**
   Repo-local `dashboard/` should contain only the convenience HTML/payload pair and direct shortcut companions derived from the canonical runtime pair. The allowed contents are:
   - `index.html`
   - `dashboard_payload.json`
   - optional year-qualified twins such as `draft_board_<year>.html` and `dashboard_payload_<year>.json` when preserving current shortcut behavior

   Broader diagnostic outputs, historical plots, and unrelated artifact families should remain under runtime `runs/<year>/pre_draft/artifacts/` or `runs/<year>/pre_draft/diagnostics/`, not under repo-local `dashboard/`.

   Alternatives considered:
   - Keep repo-local `dashboard/` as a general local dump area. Rejected because that blurs authority boundaries and makes drift harder to detect.

3. **Enforce synchronization from canonical runtime artifacts outward.**
   The lifecycle must only copy outward from the canonical runtime artifact pair to shortcut and staged surfaces. Validation must be able to detect drift independently for:
   - runtime artifact pair
   - repo-local shortcut pair
   - staged `site/` pair

   The local shortcut must never silently point at a synthetic sample payload when a live artifact pair exists.

   Alternatives considered:
   - Keep today’s behavior and rely on manual refresh habits. Rejected because actual drift already occurred.
   - Replace convenience copies with symlinks. Rejected for now because portability and publish-time expectations differ across repo, runtime, and staged surfaces.

4. **Keep staged Pages on the current inline-payload bootstrap for this change.**
   `site/index.html` should continue embedding the staged payload inline for the current change because the existing dashboard boot path expects `window.FFBAYES_DASHBOARD` at render time. This change should not expand into a new fetch-based or lazy-load bootstrap path.

   The lighter-weight load-path question remains a valid future optimization, but it should be handled by a follow-up change after lifecycle authority and synchronization are repaired.

   Alternatives considered:
   - Redesign Pages to lazy-load `site/dashboard_payload.json` as part of this change. Rejected because it would expand scope into bootstrap behavior, error handling, and offline semantics.

5. **Govern legacy visualization modules as product support decisions, not just implementation leftovers.**
   The repo should inventory visualization modules under `src/ffbayes/visualization/` and classify each as:
   - active supported generator
   - compatibility wrapper
   - deprecated generator
   - removable orphan

   Standalone generators that duplicate the dashboard product or compare identical inputs without meaningful signal should not remain implicitly supported.

   Based on the current audit:
   - `create_pre_draft_visualizations.py` should be treated as a deprecated generator candidate because it is legacy and overlaps the dashboard product, but it is still pipeline-referenced and should not be described as a pure orphan until the pipeline is updated.
   - `create_team_aggregation_visualizations.py` is not part of the supported pre-draft pipeline and is only exposed through a manual CLI entrypoint; it should be deprecated for removal rather than preserved as optional tooling.
   - `draft_strategy_comparison.py` is not part of the supported pre-draft pipeline and is only exposed through a manual CLI entrypoint; it should be deprecated for removal, and current pipeline/docs references that imply it is part of the execution graph should be treated as stale and cleaned up.
   - `create_consolidated_hybrid_visualizations.py` should be treated as a removable orphan candidate because it lacks current wiring and appears to compare logically identical inputs.
   - Flat re-export modules such as `visualization/model_performance_dashboard.py` should be treated as compatibility surfaces, not independent products.

   Alternatives considered:
   - Leave all legacy modules in place indefinitely. Rejected because docs and users cannot tell which surfaces are current.
   - Rebuild replacement standalone tooling immediately. Rejected because the current priority is to remove unused surfaces now, not to design a new visualization product outside the dashboard.

6. **Use real emitted artifact categories as the documentation contract.**
   README and docs should describe only:
   - canonical dashboard artifacts
   - staged Pages outputs
   - actually emitted diagnostics families
   - optional or deprecated visualization categories explicitly marked as such

   If a diagnostics category is no longer emitted, the docs should be updated rather than implying a broad plot catalog that does not exist.

   Alternatives considered:
   - Preserve aspirational docs until implementation catches up. Rejected because the current gap undermines trust.

7. **Separate evidence, freshness, and provenance semantics more cleanly in the dashboard UI.**
   The evidence surface should explain comparative results and limitations once, while provenance/freshness should explain artifact lineage and degraded state once. Metric labels must stay consistent between glossary, inspector, and visible cards.

   Alternatives considered:
   - Keep the existing dense dual-panel approach. Rejected because the current experience duplicates tables and surfaces ambiguous degraded states.

8. **Add validation where drift actually happens.**
   Existing lifecycle tests and the Pages smoke path are valuable but insufficient. The implementation should add coverage for:
   - repo-local shortcut synchronization
   - stale local shortcut detection
   - mismatch between staged HTML and staged payload
   - visualization docs/output contract drift where practical

   Alternatives considered:
   - Rely only on `.github/workflows/dashboard-sync.yml`. Rejected because that misses local shortcut integrity and user-facing local workflow drift.

## Risks / Trade-offs

- [Legacy module removal breaks hidden workflows] -> Mitigation: classify modules first, deprecate before removal, and keep compatibility wrappers explicit where needed.
- [Tighter sync checks make local workflows feel stricter] -> Mitigation: keep cheap non-mutating check/repair flows and preserve existing artifact names.
- [Docs become shorter but less aspirational] -> Mitigation: explicitly separate supported, optional, and deprecated visualization outputs.
- [Evidence/provenance simplification drops useful context] -> Mitigation: keep structured payload fields intact even if UI presentation becomes less redundant.
- [Repo-local dashboard content may still be user-mutated outside the pipeline] -> Mitigation: validate derived copies against canonical runtime artifacts and make stale state obvious.

## Migration Plan

1. Define the authority model and requirement deltas in spec files.
2. Update lifecycle logic so repo-local and staged copies are always derived from canonical runtime artifacts and checked independently.
3. Classify legacy visualization modules and remove unused manual-entrypoint surfaces while deciding which remaining modules stay active, become wrappers, or are deprecated.
4. Update dashboard UI wording for evidence, freshness, provenance, and metric labels without changing canonical artifact filenames.
5. Update README and docs to reflect only supported emitted visualization categories.
6. Add regression coverage for repo-local shortcut integrity and staged site drift.
7. Rollback strategy: retain current filenames and directory structure so any implementation can revert behavior without a path migration.
