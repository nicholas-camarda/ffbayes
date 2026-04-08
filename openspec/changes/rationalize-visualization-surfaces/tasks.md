## 1. Establish visualization authority and inventory

- [x] 1.1 Inventory visualization-producing modules and artifact surfaces under `src/ffbayes/visualization/`, `src/ffbayes/analysis/`, `dashboard/`, runtime outputs, and `site/`.
- [x] 1.2 Classify each visualization surface as canonical, derived convenience copy, supplemental diagnostic, compatibility wrapper, deprecated generator, or removable orphan, with unused manual-entrypoint-only generators treated as deprecated-for-removal rather than as a supported class.
- [x] 1.3 Encode the supported authority model in the implementation and docs without changing canonical `pre_draft` artifact names.
- [x] 1.4 Narrow repo-local `dashboard/` to the convenience HTML/payload pair and direct shortcut companions only.

## 2. Repair dashboard lifecycle synchronization

- [x] 2.1 Update dashboard staging logic so repo-local and runtime-local shortcut copies are always derived from the canonical runtime dashboard HTML and payload pair.
- [x] 2.2 Add validation logic or checks that can detect stale repo-local shortcut artifacts independently from staged `site/` drift.
- [x] 2.3 Ensure placeholder or sample local dashboard payloads cannot silently replace live generated outputs for the active year.
- [x] 2.4 Keep `refresh-dashboard`, `draft-strategy`, and `publish-pages` path ownership explicit in command behavior and validation output.
- [x] 2.5 Preserve the current staged Pages inline-payload bootstrap in this change; defer any fetch-based or lighter-weight load path to a follow-up.

## 3. Improve dashboard evidence, freshness, and provenance semantics

- [x] 3.1 Refine dashboard UI rendering so evidence, limitations, freshness, and provenance each have a clear role without redundant strategy-summary tables.
- [x] 3.2 Make degraded, mixed, stale, or override-driven freshness states explainable in user-facing dashboard copy, even when detailed warnings are incomplete.
- [x] 3.3 Align metric terminology such as `draft_score`, `board_value_score`, and simple VOR baseline naming across glossary, inspector, and visible dashboard surfaces.
- [x] 3.4 Ensure staged provenance distinguishes underlying analysis lineage from staged HTML-versus-payload synchronization state.

## 4. Rationalize legacy visualization paths

- [x] 4.1 Treat `create_pre_draft_visualizations.py` as a deprecated generator candidate and resolve its pipeline reference explicitly rather than leaving it implicitly current.
- [x] 4.2 Deprecate `create_team_aggregation_visualizations.py` and `draft_strategy_comparison.py` for removal because they are unused manual-entrypoint surfaces rather than supported pre-draft pipeline outputs.
- [x] 4.3 Remove stale CLI, pipeline-validation, and documentation references that imply `draft_strategy_comparison.py` or `create_team_aggregation_visualizations.py` remain supported visualization products.
- [x] 4.4 Retire, fence off, or clearly label stale visualization generators and empty or vestigial wrappers that no longer own a supported product surface, including `create_consolidated_hybrid_visualizations.py` if no supported dependency remains.
- [x] 4.5 Remove low-value static diagnostics that do not add unique decision support beyond the dashboard and evidence payload instead of rebuilding replacement standalone tooling in this change.

## 5. Align documentation and regression coverage

- [x] 5.1 Update `README.md`, `docs/README.md`, and `docs/OUTPUT_EXAMPLES.md` so documented visualization categories match real emitted outputs and clearly mark optional or deprecated surfaces.
- [x] 5.2 Add or extend pytest coverage for dashboard shortcut synchronization, stale local shortcut detection, and staged `site/` drift behavior.
- [x] 5.3 Extend dashboard smoke or related integration coverage to exercise the repo-local shortcut contract in addition to the staged Pages bundle.
- [x] 5.4 Verify the final implementation with the supported `ffbayes` environment commands relevant to touched lifecycle and visualization surfaces.
