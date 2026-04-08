## Why

The live draft dashboard is still manual-first, which is safe but vulnerable to board drift during a fast ESPN draft room where picks can land quickly and autodraft may fire without much warning. For this repo, the problem is no longer generic “import support.” The real problem is whether we can keep the board aligned with a live ESPN draft while preserving the current manual workflow if the integration fails.

## What Changes

- Replace the generic import/sync concept with an ESPN-specific, desktop-only live sync design driven by a userscript in the ESPN draft room.
- Add a local sync bridge and dashboard reconciliation contract so FFBayes can consume live ESPN pick events and map them into `taken`, `mine`, and current-pick state.
- Require a local ESPN identity setting so the dashboard can distinguish `mine` from `other` without depending on authentication secrets.
- Preserve the current manual dashboard workflow as the permanent fallback, including `Taken`, `Mine`, `Queue`, undo/redo, and finalize behavior.
- Make support boundaries explicit: this is a seasonal best-effort integration against ESPN’s live draft UI, not a stable official API contract.

## Capabilities

### New Capabilities
- `live-draft-import-sync`: define how the live dashboard consumes ESPN live draft events while preserving manual control, local identity matching, and explicit sync trust boundaries.

### Modified Capabilities
- None.

## Impact

- Affects the live dashboard command center and its local state-management flow, primarily in `src/ffbayes/draft_strategy/draft_decision_system.py` plus any local sync bridge/helper module.
- May add a repo-owned userscript artifact and a local runtime/config surface for ESPN identity, without changing the existing canonical runtime artifact names.
- Must remain compatible with the supported `pre_draft` phase and the current local dashboard/finalize flow.
- Must not make live sync mandatory; the dashboard must still work entirely through manual controls if sync is absent, stale, unsupported, or broken.
