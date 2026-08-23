# 2026 data lineage and paths

This document shows how a dashboard row is connected to the public inputs,
league profile, runtime draft state, and generated snapshot.

## Supported flow

```bash
ffbayes dashboard --year 2026
```

The command loads one fresh input snapshot for the run. Every configured local
profile is valued against that same snapshot, while each profile keeps separate
draft state in the service.

## Source lineage

| Stage | Source or code | Required contract |
| --- | --- | --- |
| player catalog, projections, ADP | ESPN's public 2026 fantasy endpoint | active current-season players, projection depth, and complete recent ADP |
| identity/status cross-check | nflverse 2026 roster data via `nflreadpy` | stable player identity and current roster status |
| league rules | `config/leagues/example_2026.json` or an ignored `*.local.json` | team count, snake format, scoring, roster, FLEX, bench/IR, and waiver fields |
| board calculation | `src/ffbayes/draft_2026/engine.py` | scoring, demand, replacement, VOR, scarcity, timing, and draft state |
| browser response | local `/api/board` endpoint | validated payload with stable IDs and digests |
| snapshot export | `runtime/runs/dashboard_2026/<timestamp>/snapshots/` | atomic export after provenance validation |

Required sources are fetched with cache mode `off`. The 2026 dashboard path
does not read staged HTML, old runtime folders, ignored generated files, or
historical seasons as a substitute for current inputs.

## What is recorded

Each payload records source URLs, seasons, fetch timestamps, cache mode, row and
coverage statistics, source SHA-256 values, code revision, profile digest,
runtime-state digest, and board digest. Runtime state includes draft slot,
current overall pick, taken IDs, your IDs, and queue IDs.

`validate_output_provenance` recomputes these relationships. A season mismatch,
future timestamp, changed state, failed coverage report, missing stable ID, or
digest mismatch rejects the payload.

## Paths

- public configuration template: `config/leagues/example_2026.json`
- private local profiles: `config/leagues/*.local.json` (ignored by Git)
- 2026 code: `src/ffbayes/draft_2026/`
- run root: `runtime/runs/dashboard_2026/<timestamp>/`
- snapshots: `<run-root>/snapshots/`
- source manifests: embedded in payloads and retained for the run

## Failure propagation

Transport failures identify the source and affected features. Semantic failures
identify the failed field or coverage statistic. The dashboard shows a blocked
state and does not retain a previous successful board as the current result.
