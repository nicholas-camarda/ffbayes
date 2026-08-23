# 2026 Data Lineage and Paths

Audience: operators and reviewers who need to trace a board back to inputs.

Scope: the current loopback dashboard run and its provenance-bound snapshots.

Trust boundary: only fresh, validated public inputs and explicit local league
profiles may reach the Python valuation engine.

## What This Is

The supported path is `ffbayes dashboard --year 2026`. It creates one run-scoped
input snapshot shared by both league profiles, then recalculates a board from
explicit runtime state.

## Source lineage

| Stage | Source or artifact | Required contract |
| --- | --- | --- |
| player catalog/projections/ADP | ESPN public 2026 fantasy feed | current season, active players, adequate projections and fresh ADP |
| eligibility cross-check | nflverse 2026 roster release via `nflreadpy` | current roster status and stable identity match |
| stable league rules | `config/leagues/league_1_2026.json`, `league_2_2026.json` | explicit team count, snake format, scoring, roster, FLEX, waiver fields |
| board calculation | `src/ffbayes/draft_2026/engine.py` | scoring, replacement, VOR, scarcity, timing, and state all in Python |
| browser response | local `/api/board` | validated payload with stable IDs and digests |
| export | `runtime/runs/dashboard_2026/<timestamp>/snapshots/*.json` | atomic, provenance-validated snapshot |

Required sources use cache mode `off`. The pipeline does not read `site/`,
`dashboard/`, older runtime folders, June artifacts, or ignored generated files.
There is no alternate source or neutral value when a required source fails.

## Provenance fields

Every board payload records the source URLs, seasons, fetched timestamps, cache
mode, row/coverage statistics, source SHA-256 values, code revision, profile
digest, runtime state digest, and board digest. Runtime state includes draft
slot, current overall pick, taken IDs, your IDs, and queue IDs.

`validate_output_provenance` recomputes these relationships. A source season
mismatch, future fetch timestamp, changed state, changed board, missing stable
ID, or failed coverage status rejects the payload.

## Paths

- stable configuration: `config/leagues/*.json`
- code: `src/ffbayes/draft_2026/`
- run root: `runtime/runs/dashboard_2026/<timestamp>/`
- snapshot directory: `<run-root>/snapshots/`
- source manifests: held in memory for the service and embedded in payloads

The loopback server is the current dashboard surface. Nothing is published or
deployed by the operator command. An unpushed worktree does not update public
GitHub Pages.

## Failure propagation

Transport failures identify the source and affected features. Semantic failures
identify the inadequate field or coverage statistic. The browser shows a
blocked state and the service does not retain a previous successful board as
the current result.
