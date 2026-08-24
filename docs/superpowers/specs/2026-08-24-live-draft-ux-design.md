# Live Draft UX and Analytical Dashboard Design

## Goal

Restore a genuinely live draft workflow and the useful analytical dashboard
surface while retaining one Python calculation engine, one payload contract,
and one local dashboard implementation.

## Architecture

`src/ffbayes/draft_2026/draft_state.py` owns the canonical per-league draft
state: draft slot, current overall pick, ordered draft actions, queue, and
derived Mine/Taken identifiers. A record action consumes exactly one current
pick when the player has not been recorded; changing an existing player between
Mine and Taken is a correction and consumes no pick. Undo removes the latest
action and restores its pick number. Manual clock synchronization remains an
explicit state transition.

`DashboardService` stores one `DraftState` per profile and applies transitions
atomically before calling the existing Python board builder. The browser never
owns authoritative draft state and never calculates rankings, value, scarcity,
timing, or recommendations. It dispatches `record`, `queue`, `undo`, and `sync`
actions and renders the returned canonical payload.

The canonical payload gains an `analytics` object containing recommendation
evidence, roster/queue views, positional-cliff series, timing-frontier rows,
and freshness summaries derived from the same board and runtime state. The
self-contained dashboard renders these values as DOM-safe tables, cards, and
bar visualizations. No old schema, publishing route, React calculation path,
or generated artifact is restored.

## State semantics

- A new state starts at overall pick `1`, independent of draft slot.
- `draft_slot` determines the user's snake turns and next user pick.
- `record(player_id, disposition)` accepts `mine` or `taken`.
- Recording an available player appends an action at `current_pick` and advances
  to `current_pick + 1`.
- Recording an already recorded player changes only its disposition.
- `queue` changes only queue membership.
- `undo` removes the most recent action and sets the clock to that action's pick.
- `sync` changes the manually editable clock without modifying action history.
- The clock may equal one past the final configured pick to represent a complete
  draft; timing fields then report no future pick.

## Payload additions

The existing `draft_2026_v1` contract remains the only schema. Its new
`analytics` members are:

- `recommendation`: best available row plus numeric evidence and alternatives;
- `roster`: Mine rows and position counts;
- `queue`: queued current rows;
- `positional_cliffs`: per-position replacement level, demand, and available
  projected-point series;
- `timing_frontier`: top available candidates with current value and
  next-pick availability;
- `freshness`: generated timestamp, source timestamps, cache modes, and
  coverage summary.

All values are generated in Python and included in provenance validation.

## Security

The restored surface uses `textContent`, explicit DOM elements, and constrained
attributes. Upstream names, recommendations, provenance, and profile text are
never inserted as executable HTML. Hostile-input tests remain required.

## Verification

Deterministic tests cover initial state, record, correction, undo, queue
invariance, snake boundaries, manual synchronization, payload analytics, and
provenance. A browser regression exercises record/undo/queue/sync against the
real local service and asserts visible analytics change without console errors.
