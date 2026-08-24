# Live Draft UX and Analytical Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local 2026 dashboard a live snake-draft assistant with atomic Taken/Mine/Queue/Undo/Sync transitions and restore canonical-data analytical panels.

**Architecture:** `draft_state.py` owns immutable-style per-league action history and derived roster/availability state. `DashboardService` owns one mutable state instance per profile and applies one transition before one Python board/payload recomputation. The self-contained dashboard renders the canonical payload; it does not calculate rankings or analytics.

**Tech Stack:** Python 3.10, pandas, pytest, standard-library HTTP server, browser JavaScript, Node.js 22, Playwright.

**Spec:** `docs/superpowers/specs/2026-08-24-live-draft-ux-design.md`

## Global Constraints

- Preserve one Python calculation engine, one `draft_2026_v1` payload contract, and one dashboard surface.
- New draft state starts at overall pick 1 and never silently uses draft slot as the clock.
- A new record advances exactly once; correction does not advance; Queue does not advance; Undo rewinds the last action.
- Draft slot, team count, scoring, roster, and snake rules come from the selected league profile.
- All analytical values shown in the browser come from the canonical Python payload.
- No login, authentication, database, stale fallback, old schema, publishing route, or generated artifact may be added.
- Preserve DOM-safe rendering and hostile-input coverage.

---

### Task 1: Canonical draft-state model

**Files:**
- Create: `src/ffbayes/draft_2026/draft_state.py`
- Modify: `src/ffbayes/draft_2026/engine.py`
- Test: `tests/test_draft_2026_state.py`
- Test: `tests/test_draft_2026_engine.py`

**Interfaces:**
- `DraftAction(pick: int, player_id: int, disposition: Literal['mine','taken'])` is the ordered recorded selection.
- `DraftState(draft_slot: int | None, current_pick: int, actions: tuple[DraftAction, ...], queue_ids: tuple[int, ...])` exposes `taken_ids`, `your_ids`, and `record`, `toggle_queue`, `undo`, and `sync_clock` transitions.
- `DraftState.record` appends only for an unrecorded player; an existing player changes disposition without advancing.
- `next_snake_pick` accepts a clock at `total_draft_picks() + 1` and returns `None` when no future pick exists.

- [ ] Write failing tests for initial pick 1, record advancement, Mine roster derivation, Taken availability, Queue invariance, correction without advancement, Undo, manual sync, final-pick bounds, and snake round boundaries including consecutive picks.
- [ ] Run `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/ffbayes/bin/python -m pytest tests/test_draft_2026_state.py tests/test_draft_2026_engine.py -q`; confirm the new tests fail because the model does not exist.
- [ ] Implement the smallest coherent state model and engine boundary changes.
- [ ] Rerun the focused tests and confirm they pass.
- [ ] Commit `feat: add canonical live draft state model`.

### Task 2: Atomic dashboard transitions and canonical analytics

**Files:**
- Modify: `src/ffbayes/draft_2026/dashboard_app.py`
- Modify: `src/ffbayes/draft_2026/pipeline.py`
- Test: `tests/test_draft_2026_dashboard_service.py`
- Test: `tests/test_draft_2026_pipeline.py`

**Interfaces:**
- `DashboardService` stores one `DraftState` per profile and never accepts browser-supplied Taken/Mine arrays as authoritative state.
- `POST /api/board` performs an initial read or explicit clock/profile synchronization.
- `POST /api/action` accepts `{profile_id, action: {type: 'record'|'queue'|'undo', player_id?, disposition?}, draft_slot?}` and returns one recomputed payload.
- Payload `runtime_state` includes `actions`, `current_pick`, `draft_slot`, derived IDs, and queue IDs.
- Payload `analytics` includes recommendation evidence, roster, queue, positional cliffs, timing frontier, and freshness.

- [ ] Write failing service tests for Taken, Mine, Queue, correction, Undo, manual synchronization, duplicate action idempotence, and independent league state.
- [ ] Run the focused service tests and confirm the current full-array request model fails the new transition assertions.
- [ ] Implement service transition dispatch with atomic state commit only after board/payload validation.
- [ ] Implement Python analytics helpers from the board and runtime state; include analytics in provenance digests/validation.
- [ ] Rerun focused service/pipeline tests and confirm all pass.
- [ ] Commit `feat: make dashboard transitions atomic and expose live analytics`.

### Task 3: Restored analytical dashboard surface

**Files:**
- Modify: `src/ffbayes/draft_2026/dashboard_app.py`
- Test: `tests/test_draft_2026_dashboard_service.py`
- Test: `tests/test_draft_2026_pipeline.py`

**Interfaces:**
- The single self-contained page renders controls for league, draft slot, current overall pick, Sync clock, Undo last pick, and Export snapshot.
- Panels with stable IDs: `recommendation-panel`, `positional-cliffs`, `timing-frontier`, `roster-panel`, `queue-panel`, `freshness-panel`, `provenance`, and `draft-board`.
- Browser event handlers dispatch server actions and replace all state-dependent content from the returned payload.

- [ ] Write failing static/browser assertions for panel presence, canonical evidence values, and DOM-safe rendering.
- [ ] Implement accessible DOM construction for cards, tables, bars, and action controls using `textContent` and constrained attributes.
- [ ] Add recommendation/evidence, positional cliff, timing frontier, roster, queue, freshness, and provenance rendering from payload analytics.
- [ ] Wire Taken/Mine/Queue/Undo/Sync to `/api/action` or `/api/board` transitions and preserve per-league state.
- [ ] Run static hostile and browser smoke tests; confirm panels render and state changes replace canonical values.
- [ ] Commit `feat: restore live analytical dashboard surface`.

### Task 4: Browser live-draft regression

**Files:**
- Modify: `tests/serve_draft_2026_hostile_fixture.py` or create `tests/serve_draft_2026_live_fixture.py`
- Create or modify: `tests/test_draft_2026_dashboard_live.mjs`
- Modify: `tests/dashboard_smoke.mjs`

**Interfaces:**
- The deterministic fixture exposes the real dashboard service with enough players for recommendations and analytics.
- The browser test records Taken A, Mine B, Queue C, Undo, and manual clock synchronization.

- [ ] Write the browser assertions first and run them to confirm the current page lacks the controls/panels and fails.
- [ ] Implement only fixture wiring needed to exercise the canonical service.
- [ ] Assert current pick 1 → 2 → 3, availability, Mine roster, Queue non-advancement, Undo back to 2, recommendation change, panel updates, no console/page errors, and manual sync.
- [ ] Add snake-boundary browser coverage for a final-slot back-to-back turn.
- [ ] Run the browser tests and commit `test: cover live draft interactions in browser`.

### Task 5: Documentation and review preparation

**Files:**
- Modify: `README.md`
- Modify: `docs/DASHBOARD_OPERATOR_GUIDE.md`
- Modify: `docs/OUTPUT_EXAMPLES.md`
- Modify: `docs/TECHNICAL_DEEP_DIVE.md`

- [ ] Document current pick semantics, automatic action advancement, correction/Undo, Queue behavior, and restored panels.
- [ ] Document the canonical payload analytics fields and one-dashboard architecture.
- [ ] Run Markdown/link and contract tests.
- [ ] Commit `docs: document live draft state and analytics`.

### Task 6: Full verification and whole-branch review

**Files:**
- Review all changes from `3f25ae3` to final HEAD.

- [ ] Run Python tests, Ruff, mypy, npm audit, npm smoke, hostile smoke, static smoke, and live browser regression from the final worktree.
- [ ] Run a fresh 2026 source-backed dashboard smoke when public sources are reachable; report exact external failures otherwise.
- [ ] Inspect the actual live payload for action history, current pick, analytics, and provenance.
- [ ] Search for duplicate schemas, recommendation engines, old frontend routes, unsafe HTML, and stale publishing references.
- [ ] Request a whole-branch code review; resolve all Critical/Important findings and perform scoped re-review.
- [ ] Confirm the worktree is clean. Do not merge or push.
