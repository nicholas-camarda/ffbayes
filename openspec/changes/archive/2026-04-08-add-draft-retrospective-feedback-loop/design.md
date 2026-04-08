## Context

FFBayes already emits a rich set of finalized draft artifacts at the end of a draft: the locked JSON/HTML snapshot, pick receipts, roster summaries, and staged dashboard exports. Those artifacts currently act like receipts, but they do not yet form a feedback loop that helps us evaluate whether the drafted roster and policy actually performed well once the season finished.

The repo is still constrained to the `pre_draft` phase, so this design must fit inside the existing runtime tree and CLI surface rather than introducing a new pipeline phase. The new capability should read from completed draft artifacts plus already-collected season outcome data and produce a retrospective report without rerunning the heavy modeling path that generates the original board.

One additional operational constraint matters here: finalized draft exports are currently emitted as plain browser downloads from the local dashboard. Browsers can suggest filenames, but they do not give FFBayes a reliable, cross-browser way to force those files into the canonical runtime tree. If retrospective analysis is going to become routine, the system needs a deterministic ingest path after download rather than depending on `Downloads/` or ad hoc user file management.

## Goals / Non-Goals

**Goals:**
- Turn finalized draft artifacts into a reusable retrospective analysis flow.
- Measure realized roster outcomes, expected-versus-realized gaps, and decision-quality trends across drafts.
- Detect systematic policy errors such as fragility miscalibration, wait-policy misses, and positional/archetype over- or under-performance.
- Preserve model-following vs pivots as secondary audit context rather than the primary learning target.
- Preserve artifact provenance so every retrospective report can be traced back to its source snapshot and receipts.
- Make the feedback loop cheap enough to run after a draft without recomputing the full draft board.
- Give finalized draft bundles a deterministic runtime home so `draft-retrospective` can discover them without brittle path assumptions.

**Non-Goals:**
- Do not change the live draft dashboard’s recommendation logic.
- Do not add a new pipeline phase or migration away from `pre_draft`.
- Do not depend on external league APIs or season standings if they are not already present in repo artifacts.
- Do not rely on browser configuration or browser-specific save dialogs as the source-of-truth path policy.
- Do not claim causal or championship-level performance from noisy season outcomes and internal artifacts alone.

## Decisions

1. Use finalized draft artifacts plus realized season outcome data as the source of truth for retrospective analysis.
   Finalized JSON snapshots and pick receipts tell us what the system actually drafted and what the user saw at the time, while collected season outcome data tells us how those drafted players actually performed. Reusing both sources avoids recomputation and keeps retrospective runs consistent with what was exported at draft time and what later happened on the field.
   Alternatives considered: re-run draft-strategy from raw data, or evaluate only from receipts without season outcomes. The former would be more expensive and blur the distinction between live recommendations and post-hoc feedback; the latter would answer the weaker question of user compliance rather than the stronger question of roster quality.

2. Add a dedicated CLI entrypoint for retrospective generation.
   A command such as `ffbayes draft-retrospective` makes the feedback loop easy to invoke from the same operational surface as the rest of FFBayes. It also keeps the runtime outputs and tests close to the existing draft workflow.
   Alternatives considered: bake retrospective generation into `draft-strategy`, or hide it behind `publish-pages`. Both would couple a post-draft workflow to unrelated live drafting or publishing concerns.

3. Keep the retrospective report year-scoped and artifact-driven.
   Retrospective outputs should live under the existing `runs/<year>/pre_draft/artifacts/draft_strategy/` tree so the feedback loop is easy to locate alongside the finalized draft bundle. Cross-season trend summaries can be derived from multiple year-scoped reports rather than inventing a new storage regime.
   Alternatives considered: create a brand-new root directory or stage the report in `site/`. Those options would complicate path policy and add avoidable publishing risk.

4. Make realized roster performance the primary evaluand and keep follow/pivot analysis secondary.
   For a user who leans heavily on the model, the central learning question is whether the roster and policy performed well, not whether the user obeyed the interface. The retrospective should therefore prioritize metrics such as realized starter and full-roster fantasy points, expected-versus-realized deltas, player hit rates, and wait-policy calibration. Follow and pivot behavior should still be recorded, but mainly as audit context.
   Alternatives considered: make follow-rate the headline metric, or omit follow/pivot information entirely. The former would optimize for user compliance rather than system quality; the latter would lose useful interpretability when a draft deviates from the board.

5. Use one retrospective command that always emits season detail and conditionally emits cross-season rollups.
   The first version should not split season-level and trend reporting into separate commands. `ffbayes draft-retrospective` should always produce a season-scoped retrospective for the requested finalized draft, and it should add a cross-season rollup when multiple finalized draft seasons are available or explicitly requested.
   Why this over separate commands: it keeps the operator workflow simple and matches the current task model, which already expects both season-level and cross-season analysis from the same capability.
   Alternatives considered: ship only single-season reports first, or add a separate trend command later. Both would fragment the feedback loop and create avoidable CLI sprawl.

6. Emit both JSON and HTML, with JSON as the canonical contract.
   The retrospective should produce a structured JSON artifact for tests, provenance, and future automation, plus an HTML report for human review. JSON should be treated as the source-of-truth machine artifact, while HTML is a derived presentation layer.
   Why this over one format only: JSON-only would make human review clumsy, and HTML-only would weaken automation and regression testing.
   Alternatives considered: HTML only, or JSON only. Both were rejected because the repo already benefits from having both machine-readable and human-readable artifacts for decision support surfaces.

7. Keep retrospective outputs local/runtime-only for the initial release.
   The first version should stage retrospective artifacts only under the existing runtime artifact tree. It should not publish to `site/`, and it should not require a cloud mirror as part of the core workflow. If the reports prove useful, a later change can extend `ffbayes publish` to mirror them into cloud storage without changing the command contract.
   Why this over immediate publish integration: the initial evaluation is still internal and noisy even when grounded in realized outcomes, so pushing it into Pages or making cloud publication part of the base workflow would overstate its maturity.
   Alternatives considered: stage retrospective outputs in `site/`, or automatically mirror them through `publish-pages`. Both were rejected because they would mix internal evaluation surfaces with public dashboard publishing before the retrospective capability has earned that exposure.

8. Degrade explicitly when the season outcome table is unavailable or incomplete.
   Realized-outcome evaluation is only meaningful when the drafted season’s player results are available in the collected data. If those outcomes are missing, partial, or schema-incompatible, the retrospective should fail or degrade explicitly rather than silently falling back to audit-only metrics and pretending it answered the stronger evaluation question.
   Alternatives considered: always emit a report even without outcome data, or silently downgrade to a receipt-only summary. Both were rejected because they would blur the distinction between outcome-grounded evaluation and lighter audit reporting.

9. Canonicalize finalized draft storage through a runtime ingest step rather than trying to force the browser download destination.
   The local dashboard can keep exporting finalized JSON and HTML bundles through normal browser downloads, but the repo should define a year-scoped runtime landing zone such as `runs/<year>/pre_draft/artifacts/draft_strategy/finalized_drafts/` and provide a cheap import path that copies or moves those browser downloads into that canonical folder. `draft-retrospective` should auto-discover from that folder first.
   Why this over forcing a browser save directory: browser download locations are user- and browser-controlled, and the current local-file dashboard uses plain anchor-triggered downloads rather than a repo-owned filesystem writer. A runtime ingest step gives FFBayes a deterministic source of truth without overpromising control it does not have.
   Alternatives considered: rely on `~/Downloads`, require `--finalized-json` every time, or attempt to coerce a specific browser save folder. All were rejected because they are brittle, user-specific, or not technically reliable across browsers.

## Risks / Trade-offs

- [Season outcomes are noisy] → One roster’s realized fantasy results reflect injuries, luck, and league context as well as draft quality. Mitigation: treat the retrospective as evaluation evidence and calibration input, not as a one-season auto-retraining signal.
- [Outcome data may be missing at evaluation time] → The retrospective may be run before the season outcome table is available. Mitigation: fail or degrade explicitly and label the report as outcome-unavailable rather than pretending the stronger evaluation succeeded.
- [Artifact drift] → Older finalized snapshot schemas may not match the new retrospective reader. Mitigation: validate schema versions and surface explicit unsupported-state messages instead of silently guessing.
- [Storage sprawl] → A new feedback report directory could become another one-off output tree. Mitigation: keep it inside the existing year-scoped `draft_strategy` artifacts structure.
- [Import ambiguity] → Operators may keep finalized bundles in arbitrary folders and assume the retrospective command will find them. Mitigation: define a canonical `finalized_drafts/` landing zone, document it clearly, and make auto-discovery prefer that folder.
- [User confusion] → Users may expect retrospective generation to recompute the latest model. Mitigation: name the command and help text to emphasize that it reads existing finalized artifacts.

## Migration Plan

1. Add the new retrospective command and artifact reader without changing the live draft recommendation flow.
2. Add a canonical `finalized_drafts/` runtime landing zone plus a cheap ingest path for browser-downloaded finalized bundles.
3. Let the command operate on existing finalized snapshots whenever the corresponding season outcome data is available so prior drafts can be backfilled immediately after import.
4. Keep the old finalized export artifacts intact; the new system should be additive.
5. If later work adds richer season outcome data, waiver context, or league standings, extend the retrospective inputs without changing the public command shape.
