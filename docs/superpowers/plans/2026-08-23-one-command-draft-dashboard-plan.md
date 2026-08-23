# One-command 2026 draft dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one supported `ffbayes dashboard --year 2026` workflow that fetches and validates fresh public inputs, serves an interactive local dashboard for both named leagues, accepts draft-day slot/state input, recalculates through the Python model, and exports provenance-bound snapshots.

**Architecture:** Keep `ffbayes.draft_2026` as the only valuation path. Add a small standard-library loopback service that owns one validated `FreshInputs` snapshot and exposes status, league metadata, board recalculation, and snapshot endpoints. Serve a focused same-origin dashboard page from that service so the current 2026 path cannot accidentally use the historical React payload contract or client-side valuation defaults.

**Tech Stack:** Python 3.10+, argparse, `http.server`, pandas/numpy, existing ESPN/nflverse public ingestion, pytest, Ruff, mypy, Node/Vite only for the existing frontend regression suite.

**Spec:** `docs/superpowers/specs/2026-08-23-one-command-draft-dashboard-design.md`

## Global Constraints

- The only normal operator command is exactly `ffbayes dashboard --year 2026`.
- No email, login, authentication, session, private-league endpoint, or account credential is used anywhere in the current 2026 path.
- Fresh required sources use cache mode `off`; no stale, neutral, 50% ADP, or ignored-artifact fallback is allowed.
- Both profiles must contain explicit stable settings; draft slot remains runtime state and is not written back to profile JSON.
- The Python engine is the single source of scoring, replacement, VOR, scarcity, availability, and recommendation values.
- Draft-state actions use stable `espn_id` values, not player names.
- Failed validation never returns a previous board as current and never creates a certified board artifact.
- Normal package metadata exposes only the `ffbayes` console script; lower-level modules remain importable only for maintenance/tests.
- Every implementation step has a focused failing test before the production change.

---

### Task 1: Lock explicit league-profile and draft-state contracts

**Files:**
- Modify: `config/leagues/league_1_2026.json`
- Modify: `config/leagues/league_2_2026.json`
- Modify: `src/ffbayes/draft_2026/league.py`
- Modify: `src/ffbayes/draft_2026/engine.py`
- Test: `tests/test_draft_2026_engine.py`
- Test: `tests/test_draft_2026_pipeline.py`

**Interfaces:**
- `LeagueProfile.draft_slot` becomes `int | None`; `None` is valid for slot-neutral valuation only.
- Add `LeagueProfile.total_draft_picks() -> int` and `LeagueProfile.validate_runtime_slot(slot: int) -> None`.
- Extend `build_draft_board(players, profile, *, current_pick=None, taken_ids=(), your_ids=()) -> pandas.DataFrame`.
- The returned frame includes `espn_id`, `is_available`, `roster_status`, and `recommendation`; frame attrs include `current_pick`, optional `next_pick`, and `replacement`.

- [ ] **Step 1: Write failing profile and state tests.** Assert the two checked-in profiles have explicit team count, snake format, half-PPR scoring, flex eligibility, waiver type, verification timestamp, and correct roster counts. Assert a null profile slot parses, `validate_runtime_slot` accepts `1..team_count`, and rejects zero, negative, non-integer, and out-of-range values. Assert slot-neutral boards have `next_pick is None` and no timing recommendation. Assert taken/user IDs are marked and are not actionable.

- [ ] **Step 2: Run the focused tests and verify failure.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_engine.py tests/test_draft_2026_pipeline.py -q`

Expected: failures for unresolved profile fields, required draft slot, missing runtime-state columns, and timing logic with a null slot.

- [ ] **Step 3: Implement the profile contract.** Populate both JSON profiles with the explicit local standard settings used by the existing test fixtures: `team_count: 10`, `draft_format: "snake"`, `scoring_label: "Half PPR"`, stat weights `3:0.04`, `4:4`, `24:0.1`, `25:6`, `42:0.1`, `43:6`, `53:0.5`, `80:1`, `85:1`, empty bonuses/waiver constraints, `waiver_type: "unknown"`, `flex_eligible: ["RB","WR","TE"]`, and an ISO verification timestamp. Remove the unresolved-status fields from the production contract. Permit `draft_slot: null` and validate it only when supplied.

- [ ] **Step 4: Implement runtime state in the engine.** Make `next_snake_pick` require a validated runtime slot rather than reading a profile slot. In `build_draft_board`, keep base valuation anchored to the complete validated frame, omit availability probabilities and timing actions when no slot/current pick is supplied, and otherwise calculate them from the explicit slot/current pick. Validate all taken/user IDs against the frame's `espn_id` set, mark rows with `roster_status` (`available`, `taken`, `mine`), and set taken/mine recommendations to those explicit statuses.

- [ ] **Step 5: Run the focused tests and verify they pass.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_engine.py tests/test_draft_2026_pipeline.py -q`

Expected: PASS for profile parsing, slot validation, state isolation, slot-neutral behavior, and existing scoring/scarcity mutation tests.

### Task 2: Make source coverage and provenance fail closed

**Files:**
- Modify: `src/ffbayes/draft_2026/sources.py`
- Modify: `src/ffbayes/draft_2026/pipeline.py`
- Test: `tests/test_draft_2026_sources.py`
- Test: `tests/test_draft_2026_pipeline.py`

**Interfaces:**
- `validate_source_coverage` remains the canonical coverage gate and returns projection/ADP statistics only on pass.
- Add `validate_fresh_inputs(inputs: FreshInputs, season: int) -> None` for season, eligibility, coverage, and manifest checks.
- `build_dashboard_payload` accepts runtime state and includes stable IDs plus profile/runtime/source/board/state digests.
- `validate_output_provenance` validates all digests and rejects stale/mismatched runtime state.

- [ ] **Step 1: Write failing mutation tests.** Add tests that inject a historical/inactive player, fewer than production projection rows, missing ADP in the market top-N, stale market timestamps, a 2025 manifest, a missing `espn_id`, a tampered state digest, and a board generated after a failed recalculation. Each must raise `SemanticInputError` or `OutputProvenanceError` without producing a certified payload.

- [ ] **Step 2: Run source and pipeline tests to confirm failure.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_sources.py tests/test_draft_2026_pipeline.py -q`

Expected: failures where current payloads lack ID/state digests or permit incomplete coverage.

- [ ] **Step 3: Implement the fail-closed checks.** Require non-null stable IDs, current eligibility, season-matched manifests, adequate position/projection depth, ADP coverage and freshness, and finite scored inputs. Include source row counts, projection counts, ADP fraction, source URLs, fetched timestamps, cache mode, and SHA-256 values in the coverage report. Do not catch and replace semantic errors.

- [ ] **Step 4: Extend payload provenance.** Add `runtime_state` (`draft_slot`, `current_pick`, `taken_ids`, `your_ids`, `queue_ids`), `profile_sha256`, `runtime_sha256`, `state_sha256`, `board_sha256`, source manifest digests, and `generated_at`. Require `espn_id` in every decision row. Validate that all timestamps are ordered and all digests recompute exactly.

- [ ] **Step 5: Run the focused tests and verify they pass.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_sources.py tests/test_draft_2026_pipeline.py -q`

Expected: PASS, with every deliberate mutation rejected and no neutral/default substitution.

### Task 3: Add the loopback dashboard service and interactive page

**Files:**
- Create: `src/ffbayes/draft_2026/dashboard_app.py`
- Modify: `src/ffbayes/draft_2026/pipeline.py`
- Test: `tests/test_draft_2026_dashboard_service.py`
- Test: `tests/test_draft_2026_dashboard_browser.mjs`

**Interfaces:**
- `DashboardService(fresh_inputs, profiles, *, project_root, run_root, code_revision)` owns immutable inputs and mutable per-profile runtime state.
- `DashboardService.handle_board(request: Mapping[str, Any]) -> dict[str, Any]` validates profile ID, slot, current pick, IDs, and provenance before invoking `build_draft_board` and `build_dashboard_payload`.
- `DashboardService.write_snapshot(request) -> Path` writes an atomic JSON snapshot under `run_root/snapshots/`.
- `serve_dashboard(...) -> int` starts `ThreadingHTTPServer` on `127.0.0.1`, serves the dashboard page and `/api/status`, `/api/leagues`, `/api/board`, `/api/snapshot`, and supports `--port`, `--no-browser`, `--fixture-root` only as developer/test options.

- [ ] **Step 1: Write failing service tests.** Use a deterministic `FreshInputs` fixture and assert `/api/status`, `/api/leagues`, valid board recalculation for both profiles, invalid slot rejection, unknown ID rejection, profile-state isolation, atomic snapshot digests, and blocked status after a source failure. Assert `POST /api/board` never returns an old board when the new request fails.

- [ ] **Step 2: Run the service tests to verify failure.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_dashboard_service.py -q`

Expected: import/endpoint failures because the service does not exist.

- [ ] **Step 3: Implement the service boundary.** Use only Python standard-library HTTP classes. Parse and reject malformed JSON and unknown fields with JSON error responses. Keep one immutable validated `FreshInputs` object for the process. Maintain state in a dict keyed by `profile_id` plus source/profile digest. Build a fresh board for every accepted state mutation, then commit state only after payload validation succeeds.

- [ ] **Step 4: Implement the page.** Serve a self-contained same-origin HTML/JavaScript page with a loading state, blocked state, league selector, read-only roster/scoring settings, slot/current-pick controls, taken/mine/queue controls by `espn_id`, board table, provenance panel, and snapshot button. The page displays Python-returned values; it contains no scoring, replacement, VOR, scarcity, ADP, or recommendation calculations.

- [ ] **Step 5: Add browser falsification tests.** Start the service with the fixture bundle and use the existing Playwright/smoke harness to select each league, enter different slots, verify next-pick/recommendation changes, mark a player taken, switch leagues, export a snapshot, and observe visible blocked states for invalid slot and failed source.

- [ ] **Step 6: Run service and browser tests.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_dashboard_service.py -q` and `node tests/test_draft_2026_dashboard_browser.mjs`

Expected: PASS with no requests outside loopback once the fixture-backed service is running.

### Task 4: Make the CLI and package surface one-command only

**Files:**
- Modify: `src/ffbayes/cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_run_pipeline_split.py`
- Create: `tests/test_cli_dashboard.py`

**Interfaces:**
- Add `dashboard` to the public command registry and dispatch it to `ffbayes.draft_2026.dashboard_app.main`.
- `ffbayes dashboard --year 2026` is the only operator workflow shown in top-level help.
- Unsupported dashboard flags return code 2 with an actionable message.
- Retained internal command help returns without importing/running work-producing modules.

- [ ] **Step 1: Write failing CLI tests.** Assert top-level help shows exactly one draft-day command and no overlapping aliases, `dashboard --help` has no filesystem/network side effects, unsupported options fail, `dashboard --year 2026 --no-browser` dispatches, and `pyproject.toml` defines only the `ffbayes` console script.

- [ ] **Step 2: Run CLI tests to verify failure.**

Run: `PYTHONPATH=src pytest tests/test_cli_dashboard.py -q`

Expected: failures because `dashboard` is not registered, help uses `parse_known_args`, and package scripts are duplicated.

- [ ] **Step 3: Implement exact command-surface behavior.** Register `dashboard` as public, hide legacy maintenance commands from normal help, remove direct `ffbayes-*` console scripts, and intercept `--help` before dispatch for every retained command. Parse the dashboard argument vector with strict `parse_args`, never `parse_known_args`; keep compatibility dispatch only for explicitly invoked maintenance commands.

- [ ] **Step 4: Run CLI tests and existing dispatcher tests.**

Run: `PYTHONPATH=src pytest tests/test_cli_dashboard.py tests/test_run_pipeline_split.py -q`

Expected: PASS, including safe help and strict rejection.

### Task 5: Update current documentation and remove contradictory operator paths

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/DASHBOARD_OPERATOR_GUIDE.md`
- Modify: `docs/DASHBOARD_FRONTEND_ARCHITECTURE.md`
- Modify: `docs/DASHBOARD_FRONTEND_CUTOVER.md`
- Modify: `docs/DATA_LINEAGE_AND_PATHS.md`
- Modify: `docs/METRIC_REFERENCE.md`
- Modify: `docs/OUTPUT_EXAMPLES.md`
- Modify: `docs/TECHNICAL_DEEP_DIVE.md`
- Modify: `tests/test_documentation_contracts.py`

**Interfaces:**
- Current-facing docs state the one-command workflow, local loopback boundary, runtime slot entry, two league labels, output path, fail-closed behavior, and no-login policy.
- Historical plans remain historical only when explicitly labeled; no current guide instructs `pre-draft --stage-pages`, static `dashboard/index.html`, or direct dashboard staging.

- [ ] **Step 1: Write failing documentation-contract tests.** Search all current docs for obsolete operator commands and assert they are absent from current sections; assert required command, league names, runtime slot wording, and no-login wording are present. Assert docs mention the public GitHub branch is not changed by an unpushed worktree.

- [ ] **Step 2: Run documentation tests to verify failure.**

Run: `PYTHONPATH=src pytest tests/test_documentation_contracts.py -q`

Expected: failures for stale command examples and missing current workflow language.

- [ ] **Step 3: Update the docs.** Replace operator instructions with `ffbayes dashboard --year 2026`; move maintenance commands into an explicitly labeled developer-only section; document the named leagues and runtime slot flow; document local run-scoped artifacts and provenance; state directly that no accounts/private league access are used.

- [ ] **Step 4: Run documentation tests and grep.**

Run: `PYTHONPATH=src pytest tests/test_documentation_contracts.py -q` and `rg -n "pre-draft --stage-pages|stage-dashboard|ffbayes-draft|dashboard/index.html" README.md docs --glob '*.md'`

Expected: tests pass; any remaining matches are inside labeled historical/developer sections only.

### Task 6: Full falsification and validation gate

**Files:**
- Modify: `tests/test_draft_2026_dashboard_smoke.mjs`
- Create: `tests/test_draft_2026_acceptance.py`
- Create: `tests/test_draft_2026_provenance.py`

**Interfaces:**
- Acceptance tests exercise the canonical command with a deterministic fixture source and validate both league payloads, top-100 shape, provenance, and state isolation.

- [ ] **Step 1: Add acceptance mutations.** Deliberately change FLEX count, team count, scoring, and draft slot and assert replacement levels, scarcity/value signals, or next-pick recommendations change. Remove projections, ADP, current roster status, source digest, or league settings and assert the command blocks.

- [ ] **Step 2: Run acceptance tests to identify remaining failures.**

Run: `PYTHONPATH=src pytest tests/test_draft_2026_acceptance.py tests/test_draft_2026_provenance.py -q`

Expected: any remaining failure identifies an implementation gap rather than being weakened.

- [ ] **Step 3: Fix only the canonical implementation path.** Resolve failures in the profile, source, engine, service, or CLI modules; do not add player-specific blacklists, neutral defaults, compatibility fallbacks, or duplicate client-side math.

- [ ] **Step 4: Run the complete validation suite from the isolated worktree.**

Run:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src mypy src/ffbayes
PYTHONPATH=src ruff check src tests
npm --prefix dashboard_frontend test -- --runInBand
npm --prefix dashboard_frontend run typecheck
npm --prefix dashboard_frontend run build
node tests/test_draft_2026_dashboard_browser.mjs
PYTHONPATH=src python -m ffbayes.cli dashboard --year 2026 --help
```

Expected: all relevant checks pass; dashboard help creates no runtime artifacts.

- [ ] **Step 5: Inspect fresh outputs and provenance manually.** Confirm both league payloads contain current-season IDs, adequate coverage statistics, explicit source timestamps/digests, no Tom Brady or other historical-only player, top-25/top-50/top-100 plausibility, one-QB quarterback range, and explainable market/model outliers. Record any inaccessible ESPN/nflverse source as an external blocker instead of substituting data.

- [ ] **Step 6: Run final worktree checks.**

Run: `git diff --check`, `git status --short`, and `git diff --stat`.

Expected: no whitespace errors, only intentional implementation/doc/test changes, and no merge/push/deploy performed.

## Self-review checklist

- Spec coverage: Tasks 1–2 cover explicit settings, current-player/projection/ADP gates, replacement/VOR/scarcity, provenance, and mutation rejection. Task 3 covers the local service, UI states, two-league isolation, ID-based draft state, snapshots, and browser behavior. Task 4 covers the one-command CLI and safe help. Task 5 covers all current-facing docs. Task 6 covers full falsification, manual top-100 review, external-source reporting, and the complete validation suite.
- Placeholder scan: no task relies on “TBD”, “TODO”, an unnamed helper, a default substitution, or an unspecified error branch; each step names concrete files, functions, tests, and commands.
- Type consistency: the `DashboardService`, `handle_board`, `write_snapshot`, `serve_dashboard`, `LeagueProfile.validate_runtime_slot`, and extended `build_draft_board` interfaces are named once and reused consistently.
