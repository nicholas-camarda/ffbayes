## 1. Define the ESPN sync contract

- [ ] 1.1 Define the `espn_live_pick_v1` event contract for userscript-emitted pick updates.
- [ ] 1.2 Define the local `espn_identity` contract and the precedence rule of team label over username for ownership detection.
- [ ] 1.3 Define duplicate-event suppression, ownership degradation to `unknown`, and current-pick monotonicity rules.

## 2. Implement local reconciliation and fallback behavior

- [ ] 2.1 Add one-way synchronization logic that maps ESPN live events into the current dashboard model for `taken`, `mine`, and current pick number.
- [ ] 2.2 Preserve manual fallback behavior so `Taken`, `Mine`, `Queue`, undo/redo, and finalize remain usable when sync is absent, partial, stale, or broken.
- [ ] 2.3 Ensure queue remains local-only and is never overwritten by ESPN sync in v1.
- [ ] 2.4 Ensure each sync batch is one undoable transaction and that manual edits still work after synced updates.

## 3. Expose the workflow in the dashboard and helper surfaces

- [ ] 3.1 Add an ESPN-specific sync entry point and status surface in the live dashboard.
- [ ] 3.2 Add explicit sync states and controls for `live`, `stale`, `disconnected`, and `unsupported page`, including pause/disconnect behavior.
- [ ] 3.3 Add a repo-owned userscript or helper artifact that observes the ESPN draft room and emits normalized live pick events into the local sync bridge.
- [ ] 3.4 Extend the unified CLI or local runtime helper surface only as needed to support the local sync bridge and identity configuration without changing the existing `pre_draft` flow.

## 4. Test and document the behavior

- [ ] 4.1 Add parser tests from representative ESPN draft-room DOM fixtures, including unsupported-page detection.
- [ ] 4.2 Add unit/state tests for valid opponent picks, valid owned picks, ambiguous identity, duplicate events, and monotone current-pick updates.
- [ ] 4.3 Add dashboard smoke or integration coverage for sync status, stale/disconnected behavior, and manual fallback after sync failure.
- [ ] 4.4 Update README or command reference text to describe ESPN sync as optional, seasonal best-effort, identity-aware, and always backed by the current manual workflow.
