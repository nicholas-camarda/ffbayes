# One-command 2026 draft dashboard

## Status

Draft design for review. This document is intentionally written before implementation.

## Objective

Give the user one normal command that prepares and opens the 2026 fantasy-football dashboard:

```bash
ffbayes dashboard --year 2026
```

The command owns the complete user workflow: fresh public-data collection, semantic validation, league selection, draft-slot entry, draft-state updates, recommendation recalculation, provenance display, and local snapshot export. The user should not need to run collection, validation, board-generation, dashboard-staging, or publishing commands.

The dashboard must support two separate local league profiles:

- Bill's Underbit… — Med School Friends: 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 D/ST, 1 K, 6 bench, 1 IR.
- Nicholas's Nif… — Camarda-Klein Family: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 D/ST, 1 K, 7 bench, 1 IR.

The profile files contain the stable league rules. Draft slot is deliberately not a checked-in value because it is unknown until draft day. The dashboard collects it at runtime separately for each league.

## User experience

### Start

The user runs one command. The command:

1. Loads the two local league profiles.
2. Starts a loopback-only local dashboard server and opens the browser.
3. The server fetches current public 2026 ESPN fantasy player/projection/ADP data and the current nflverse roster feed.
4. The server validates season, eligibility, projection depth, ADP coverage, scoring inputs, replacement demand, and source provenance.

No email, login, private-league identifier, cloud service, or account session is used.

The browser initially shows a loading state. If collection or validation fails, it remains open on a visible blocked state containing the exact failure and affected features. No certified board artifact is created on failure.

### League setup

The initial dashboard screen presents the two named leagues. Selecting a league loads that league's independent state. Stable league settings are displayed read-only so the user can verify the selected league; draft slot and current overall pick are editable runtime inputs.

The user enters a draft slot from 1 through that league's configured team count. Until a valid slot is entered, the dashboard may show slot-independent valuation rankings, but it must not display a guessed availability probability or `draft_now`/`can_wait` recommendation. It must show an explicit `Draft slot required` state.

The user's draft slot is persisted only in local dashboard state and in any exported snapshot. It must not mutate the checked-in profile file.

### Draft operation

After the slot is entered, the dashboard displays:

- current overall pick;
- next snake pick for the selected league;
- ranked available players;
- projected points, replacement level, VOR, scarcity, and ADP;
- availability at the next pick;
- `draft_now`, `fallback`, or `can_wait` action;
- provenance and freshness status.

The user can mark players as taken by another team, mark players as theirs, edit the current overall pick, and maintain a queue. Each state change submits the complete effective state through the same board-recalculation endpoint; the current pick is never inferred from wall-clock time or a hidden default. Recalculation is performed by the canonical Python model through the local service, not by a second JavaScript valuation implementation.

League state is namespaced by `profile_id`. Switching from Bill's league to the Camarda-Klein Family league cannot leak taken players, roster state, slot, or current pick across leagues.

### Export

The dashboard provides a local snapshot action. A snapshot contains the effective league configuration, runtime draft slot/current pick, taken and user-roster state, board rows, source manifests, profile digest, code revision, state digest, and generation timestamp. A snapshot is read-only and does not publish or deploy anything.

## System design

### Single user-facing entrypoint

Add a `dashboard` CLI command. Its implementation may call lower-level collection, validation, model, and rendering functions, but those commands are developer/internal surfaces. The normal README and operator guide document only `ffbayes dashboard --year 2026`.

Useful developer-only options may include an explicit output root, port, or no-browser mode, but they are not required for the user workflow and must not alter validation semantics.

The top-level help must not present overlapping draft commands as separate normal workflows. `pre-draft`, `pipeline`, `split`, `draft-strategy`, `stage-dashboard`, and `refresh-dashboard` remain available only as explicitly labeled developer/compatibility surfaces if tests or maintenance workflows still require them. Their implementation modules may remain, but they must not be duplicated as additional user-facing entrypoints. Redundant console-script aliases such as `ffbayes-cli` and direct module wrappers must either be removed or clearly marked developer-only; the supported operator executable is `ffbayes`.

The supported draft-day command surface is exactly one operator command: `ffbayes dashboard --year 2026`. Post-draft retrospective and research/diagnostic routines are not part of the draft-day surface and must not appear as alternate ways to generate the board. Existing lower-level commands may remain importable for tests and maintenance, but they must be hidden or explicitly labeled internal. The installed package should expose only the deliberate `ffbayes` executable for normal use; direct `ffbayes-*` wrappers are not required for the product workflow.

The dispatcher must reject unsupported options for the dashboard command. It must not use `parse_known_args` or equivalent behavior that silently discards user input on the supported path. Every command retained in the top-level parser must have a safe `--help` path that returns without fetching data, running a backtest, writing runtime artifacts, or opening a browser.

Documentation is part of the product contract. `README.md`, `docs/README.md`, and `docs/DASHBOARD_OPERATOR_GUIDE.md` must agree with `ffbayes --help`, the actual command dispatch, output paths, and the current 2026 payload. Stale instructions for `pre-draft --stage-pages`, `draft-strategy`, static dashboard shortcuts, or historical runtime paths must be moved to an explicitly labeled developer/legacy section or removed from the operator path. The public GitHub `master` documentation is external state and cannot be declared current from an unpushed worktree; after implementation, the local docs must be ready for a deliberate reviewed push, but this task does not push or deploy.

### Local service boundary

The command starts a local-only HTTP service bound to loopback. The service owns one validated `FreshInputs` snapshot and exposes same-origin JSON endpoints:

- `GET /api/status`: source freshness, validation status, and blocking errors;
- `GET /api/leagues`: sanitized stable profile metadata for the two leagues;
- `POST /api/board`: effective profile plus runtime slot/current pick/draft-state input, returning a newly validated dashboard payload;
- `POST /api/snapshot`: writes a provenance-bound snapshot under the run-scoped runtime directory.

The service must reject malformed JSON, unknown profile IDs, invalid slots, impossible current picks, unknown players, season mismatches, and state/profile digest mismatches. It must not return a prior successful board after a failed recalculation.

The valid current-pick range is 1 through the configured total draft picks. A draft-state update is atomic: the dashboard keeps the last known valid state visible while displaying the new validation error, and it never presents a partially recalculated board as current.

The implementation should prefer Python's standard-library loopback server unless an existing dependency is deliberately adopted. No external service is needed.

### Canonical model path

The current `ffbayes.draft_2026` engine remains the only valuation implementation. It must be extended so runtime draft state is an explicit input. Base scoring, replacement levels, VOR, and scarcity remain anchored to the validated full player universe; taken players and the user's roster affect actionable availability and roster need. Draft slot/current pick affect snake timing and availability recommendations. The browser displays the returned validated payload and does not recreate scoring math.

The payload contract must distinguish:

- stable profile settings;
- runtime draft state;
- effective recommendation context;
- source and code provenance.

No slot, scoring value, team count, or market value may be filled by a neutral/default substitution when absent.

### Dashboard integration

Reuse the existing dashboard frontend components where their payload contract is compatible, but adapt them to the current 2026 payload rather than routing current boards through the historical dashboard pipeline. The current static 2026 HTML renderer is replaced for the user-facing command by the interactive frontend template served from the local service.

The dashboard must show an explicit error/blocked screen for:

- unavailable or semantically inadequate ESPN data;
- unavailable or semantically inadequate nflverse roster data;
- unresolved stable league settings;
- missing draft slot when timing recommendations are requested;
- failed board recalculation;
- stale or inconsistent provenance.

The UI must not silently fall back to the old staged site, a prior runtime output, or an ignored generated artifact.

## Data and provenance contract

The initial load creates one run-scoped input snapshot for both leagues. It records:

- source URLs and source names;
- source season;
- fetched timestamps;
- cache mode (`off` for required fresh sources);
- row counts and positional coverage;
- projection and ADP coverage statistics;
- source SHA-256 digests;
- code revision.

Every board response and exported snapshot records:

- base profile digest;
- effective runtime configuration digest;
- draft-state digest;
- source manifest digests;
- board digest;
- generated timestamp;
- current pick and next pick.

An output is valid only if all digest and season relationships validate. A failed source or failed recalculation leaves no new certified board artifact.

## Error and external-data policy

If ESPN's public feed fails, changes schema, is truncated, is missing usable ADP/projections, or is inaccessible, the dashboard reports the exact source and failure mode, identifies dependent features, classifies transience when possible, and does not substitute another source silently.

If the nflverse roster feed fails or does not provide a current-season eligibility cross-check, current-player filtering fails closed. No retired or historical-only player may become actionable.

No authentication or private league endpoint is a valid dependency. The dashboard must contain a regression guard that scans the user-facing command and current 2026 path for email, login, session, private-league, or account requirements.

## Testing and acceptance criteria

### Python tests

- `ffbayes dashboard --year 2026` is the documented and functional user entrypoint.
- The top-level help exposes one draft-day operator command and does not advertise overlapping board/staging aliases.
- `ffbayes dashboard --help` and all retained command help paths have no side effects.
- Unsupported dashboard flags fail with a nonzero status and an actionable error.
- Both named profiles load independently and have the correct roster shapes and display labels.
- A null profile draft slot is allowed for slot-neutral valuation but cannot produce timing recommendations.
- Runtime slot entry accepts 1..team_count and rejects zero, negative, non-integer, and out-of-range values.
- Different runtime slots produce different snake next picks and availability/recommendation outputs.
- Runtime slot/current-pick input does not mutate profile files.
- League state is isolated by profile ID.
- Taken and user-roster players are handled explicitly and cannot be unknown or historical players.
- Source, projection, ADP, replacement, and provenance mutation tests remain fail-closed.
- Snapshot digests and timestamps validate, and tampering is rejected.

### Frontend tests

- Initial loading, blocked-source, unresolved-profile, and slot-required states render clearly.
- The user can select each league and enter an independent draft slot.
- Applying a slot updates current/next pick and recommendation lanes.
- Marking a player taken or yours updates the visible board and persists in the correct league namespace.
- Switching leagues restores the other league's state without contamination.
- Snapshot export contains the effective runtime state and provenance.

### Integration and browser tests

- Start the local dashboard command with a deterministic fixture source bundle.
- Exercise both leagues in a browser.
- Enter different draft slots and prove the payload and displayed recommendations change.
- Mark players taken and prove they are no longer actionable.
- Prove source failure, stale provenance, invalid slot, and profile mismatch produce visible blocked states.
- Verify no account/login or external network request is made by the dashboard after the initial public-source fetch; loopback API traffic is the expected local implementation detail.

### Full validation

Run all Python tests, mypy, Ruff, frontend tests/typecheck/build, Playwright/smoke tests, ingestion/validation tests, current-season model tests, and the full isolated dashboard pipeline. Inspect both league payloads and top-100 outputs before declaring readiness.

## Non-goals

- Connecting to ESPN private leagues or importing draft-room state automatically.
- Publishing or deploying a dashboard.
- Treating old staged pages or historical runtime outputs as current inputs.
- Allowing the UI to edit stable league scoring/roster rules silently.
- Duplicating the Python valuation engine in TypeScript.

## Completion definition

The feature is complete when the user can run one command, choose either named league, enter the draft slot on draft day, operate the draft state in the browser, and export a provenance-validated snapshot, while all source, player-universe, coverage, valuation, and no-login invariants remain enforced.
