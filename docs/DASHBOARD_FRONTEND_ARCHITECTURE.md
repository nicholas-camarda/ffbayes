# 2026 dashboard architecture

The draft-day surface is a local Python service with a same-origin browser
page. Python is the single implementation of scoring, replacement, VOR,
scarcity, ADP timing, and provenance.

## Service

```bash
ffbayes dashboard --year 2026
```

The command starts a standard-library `ThreadingHTTPServer` on loopback and
serves one self-contained page. Its endpoints are:

- `GET /api/status` — source freshness, coverage, and blocked errors.
- `GET /api/leagues` — sanitized profile metadata.
- `POST /api/board` — strict runtime state and a freshly calculated payload.
- `POST /api/snapshot` — atomic export of the validated payload.

The service loads one `FreshInputs` snapshot for the run. Profile state contains
the draft slot, current overall pick, taken IDs, your IDs, and queue IDs. A
mutation is committed only after board construction, payload creation, and
provenance validation succeed.

## Browser responsibilities

The page handles loading and blocked states, profile selection, read-only league
settings, slot/current-pick inputs, Taken/Mine/Queue actions, table rendering,
provenance display, and snapshot export. It does not recalculate fantasy
points, replacement levels, VOR, scarcity, ADP availability, or recommendations.

## Optional React package

`dashboard_frontend/` contains a tested React/Vite template for frontend
development. It is not a second draft-day calculation path. The operator only
needs the Python command above; Node is required for frontend development and
template regeneration, not for using a generated board.
