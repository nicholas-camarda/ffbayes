## 1. Retrospective Data Model

- [x] 1.1 Define the retrospective input contract for finalized draft snapshots, pick receipts, exported dashboard artifacts, and the realized season outcome table for the drafted season.
- [x] 1.2 Add path helpers or constants for year-scoped retrospective outputs under the existing `pre_draft` artifact tree.
- [x] 1.3 Add explicit handling for missing or unsupported finalized artifact schemas and for missing or incomplete season outcome data.
- [x] 1.4 Add a canonical year-scoped `finalized_drafts/` runtime path under `draft_strategy/` for imported finalized bundles.

## 2. Retrospective Analysis Engine

- [x] 2.1 Implement loading and normalization of one or more finalized draft seasons plus matching realized season outcomes into a retrospective analysis frame.
- [x] 2.2 Implement primary metrics for expected-versus-realized roster performance, starter and full-roster realized fantasy points, player hit rates, and wait-policy calibration.
- [x] 2.3 Implement cross-season aggregation so the same retrospective command can compare trends over time when multiple finalized seasons are available.
- [x] 2.4 Implement secondary audit metrics for model-follow rate, pivot rate, and pick-level decision context when receipts contain that information.
- [x] 2.5 Emit a structured retrospective JSON summary with provenance and degraded-state markers.

## 3. CLI and Reporting

- [x] 3.1 Add a dedicated `ffbayes draft-retrospective` CLI entrypoint that reads existing finalized artifacts and realized season outcome data instead of rerunning draft modeling.
- [x] 3.2 Generate both canonical JSON and companion HTML retrospective artifacts in the year-scoped runtime directory.
- [x] 3.3 Make the report clearly distinguish outcome-grounded evaluation from secondary follow/pivot audit summaries.
- [x] 3.4 Update README command/reference docs to explain the retrospective workflow, required season-outcome inputs, and output locations.
- [x] 3.5 Add a cheap ingest/import workflow that copies or moves browser-downloaded finalized draft bundles into the canonical `finalized_drafts/` runtime folder without rerunning draft modeling.
- [x] 3.6 Make `ffbayes draft-retrospective` auto-discover finalized draft JSONs from the canonical `finalized_drafts/` folder before requiring explicit `--finalized-json` paths.

## 4. Output Boundaries

- [x] 4.1 Keep the initial retrospective capability runtime-local and confirm it does not publish to `site/` or require a cloud mirror.

## 5. Verification

- [x] 5.1 Add pytest coverage for successful retrospective generation from finalized artifacts plus realized season outcomes, including JSON output and HTML companion generation.
- [x] 5.2 Add pytest coverage for missing season-outcome inputs and unsupported-schema failure modes.
- [x] 5.3 Add pytest coverage showing that follow/pivot metrics degrade cleanly when receipts are partial but outcome evaluation remains available.
- [x] 5.4 Add CLI coverage for the new command and its help/output surface.
- [x] 5.5 Add pytest coverage for finalized-bundle ingest into the canonical `finalized_drafts/` runtime directory.
- [x] 5.6 Add CLI coverage showing that `ffbayes draft-retrospective` can succeed via canonical-folder auto-discovery after ingest.
