## Context

The live draft dashboard already has a workable manual control model: local browser state drives `taken`, `mine`, `queue`, undo/redo, and finalize. That model is safe because it does not depend on any upstream system, but it can drift during a fast ESPN draft room where multiple picks may happen while the operator is distracted.

The earlier version of this change treated the problem as a generic import/sync layer. That is not precise enough. The actual product question is whether FFBayes can track a live ESPN draft closely enough to reduce board drift without turning the dashboard into a fragile, provider-coupled system that becomes unusable when ESPN changes its UI. This design therefore treats ESPN live sync as optional operational assistance layered on top of the existing manual workflow.

## Goals / Non-Goals

**Goals:**
- Support near-live ESPN draft updates through a desktop browser userscript running in the ESPN draft room.
- Reconcile ESPN pick events into the current dashboard model for `taken`, `mine`, and current pick number.
- Require local identity matching so owned picks can be classified without storing authentication secrets.
- Preserve manual controls, undo/redo, local browser state, and finalize behavior before, during, and after sync problems.
- Surface sync health, freshness, and trust boundaries clearly in the dashboard.

**Non-Goals:**
- Do not build a generic multi-provider sync platform.
- Do not depend on stable official ESPN APIs or claim low-maintenance compatibility.
- Do not replace the current manual dashboard workflow or make sync mandatory.
- Do not let ESPN sync overwrite local queue state in v1.
- Do not publish ESPN identity or sync metadata to `site/` or other public artifacts.

## Decisions

1. **Use an ESPN userscript, not an official API integration.**
   V1 should assume a desktop browser userscript running in the ESPN live draft room. The userscript observes visible draft-room state and emits normalized pick events for FFBayes. This is the only approach that plausibly delivers near-live behavior without inventing unsupported API guarantees.

   Alternatives considered:
   - Official ESPN API integration. Rejected because no stable official live-draft integration surface is assumed.
   - Generic file import only. Rejected because it does not satisfy the true live-sync use case.
   - Browser extension first. Rejected because it adds packaging and lifecycle overhead before the userscript contract is proven.

2. **Treat sync as optional, additive assistance rather than the source of truth.**
   The dashboard must remain fully operable through manual controls even if sync fails, stalls, or becomes unsupported. Incoming sync events should add or upgrade local board state conservatively rather than replace it wholesale.

   Alternatives considered:
   - Make sync authoritative. Rejected because parser drift, stale updates, or identity ambiguity would make the whole dashboard fragile.
   - Disable manual controls while sync is active. Rejected because draft-day recovery speed matters more than purity.

3. **Require local ESPN identity for ownership detection.**
   The sync flow needs a local identity contract so the dashboard can distinguish `mine` from `other`. The preferred identifier is the draft-room team label visible in ESPN. Username is a fallback only when the team label is unavailable or unstable.

   Alternatives considered:
   - Infer ownership with no local identity. Rejected because ambiguity would be too frequent.
   - Store auth tokens or secret cookies. Rejected because the feature only needs ownership classification, not account-level authentication.

4. **Keep queue local-only in v1.**
   ESPN sync should update `taken`, `mine`, and current pick number, but it must not overwrite `queue`. Queue is a FFBayes planning surface, not something ESPN represents reliably.

   Alternatives considered:
   - Drive queue from ESPN. Rejected because ESPN does not encode FFBayes intent.
   - Clear queue when synced picks arrive. Rejected because that would destroy local planning context.

5. **Expose sync health and fallback controls in the dashboard.**
   Users need to know whether ESPN sync is live, stale, disconnected, or unsupported. The dashboard must also offer explicit pause/disconnect behavior and make it obvious that manual actions remain available.

   Alternatives considered:
   - Quiet background sync with no status. Rejected because silent failure during a live draft is unacceptable.

6. **Support seasonal best-effort maintenance only.**
   This capability should be specified as an unsupported ESPN integration that is expected to require preseason verification and occasional upkeep if the ESPN UI changes.

   Alternatives considered:
   - Promise near-stable maintenance-free behavior. Rejected because the integration surface is not controlled by this repo.

## Public Interfaces

### Live event contract

```json
{
  "schema_version": "espn_live_pick_v1",
  "captured_at": "ISO-8601 timestamp",
  "source": "espn_userscript",
  "league_label": "string or null",
  "pick_number": 12,
  "player_name": "string",
  "position": "string or null",
  "team": "string or null",
  "drafted_by": "mine|other|unknown"
}
```

### Local identity contract

```json
{
  "espn_identity": {
    "team_label": "string or null",
    "username": "string or null"
  }
}
```

Rules:
- The userscript should attempt to classify ownership from visible ESPN draft-room labels.
- The dashboard should prefer `team_label` matching over `username`.
- If ownership cannot be matched confidently, the event should degrade to `unknown`.
- Duplicate events should be deduplicated by normalized player name plus pick number when available.

## Reconciliation Rules

- `other` events mark players as `taken`.
- `mine` events mark players as `mine` and therefore also `taken`.
- `unknown` events mark players as `taken` and surface an ownership warning.
- Queue remains unchanged by sync.
- Current pick number may advance only when incoming pick numbers are monotone and valid.
- Each sync batch should be one undoable transaction.
- Manual actions remain available before, during, and after sync failures.

## Risks / Trade-offs

- [ESPN UI drift] -> The userscript may break when ESPN changes the draft-room DOM. Mitigation: seasonal best-effort support, explicit unsupported-page state, and manual fallback.
- [Identity ambiguity] -> Owned picks may be misclassified if team labels or usernames are unclear. Mitigation: require local identity config, prefer team label, degrade to `unknown`, and surface warnings.
- [Stale or disconnected sync] -> The operator may trust stale board state. Mitigation: explicit `live`, `stale`, `disconnected`, and `unsupported page` states with last-update timestamp.
- [Manual/sync conflicts] -> Local edits and incoming events can disagree. Mitigation: additive reconciliation, queue protection, and undoable sync batches.
- [Operational dependency] -> Users may expect sync to “just work” forever. Mitigation: document the feature as optional ESPN assistance with expected preseason verification.
