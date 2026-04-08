## Why

FFBayes now exports finalized draft receipts, pick-by-pick decisions, and post-draft summaries, but those artifacts are not yet folded back into a structured retrospective loop that evaluates whether the drafted roster actually performed well. For a user who leans heavily on the system, the important learning question is not primarily "did the human follow the model?" but "did the roster and draft policy produce good realized outcomes once the season finished?"

## What Changes

- Add an outcome-grounded post-draft retrospective flow that ingests finalized draft artifacts, pick receipts, and realized season fantasy results for the drafted season.
- Add a canonical runtime landing zone for finalized draft bundles so browser-downloaded draft receipts can be imported into a deterministic year-scoped folder before retrospective analysis.
- Evaluate projected versus realized roster performance, player hit rates, wait-policy calibration, and repeatable policy errors across a completed draft, while adding a cross-season rollup when multiple finalized drafts are available.
- Keep model-following and pivot behavior as secondary audit context rather than the primary learning target.
- Expose retrospective outputs as a canonical JSON artifact plus a companion HTML report that can be reviewed alongside existing draft artifacts.
- Keep the retrospective grounded in explicit, reproducible draft receipts and collected season outcome data rather than informal notes or manual analysis.
- Keep the first release runtime-local rather than publishing internal retrospective outputs to `site/`.

## Capabilities

### New Capabilities
- `draft-retrospective-feedback-loop`: define how finalized draft artifacts are turned into reusable retrospective evidence and summary outputs.

### Modified Capabilities
- None.

## Impact

- Affects post-draft analysis surfaces, especially finalized draft payloads, collected season outcome data, and any retrospective report or CLI that consumes them.
- Likely touches `src/ffbayes/draft_strategy/draft_decision_system.py`, draft finalization/export paths, and any new retrospective analysis module under `src/ffbayes/analysis/`.
- Runtime artifacts under `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/` would gain a canonical `finalized_drafts/` landing zone plus new retrospective JSON and HTML artifacts grounded in realized season outcomes.
- No phase migration is intended; this remains inside the supported `pre_draft` workflow, but it expands the workflow from pre-draft advice into post-draft evaluation.
- No Pages publication is intended in the initial release; retrospective outputs stay local/runtime-only unless a later publish change promotes them.
