# 2026 Dashboard Frontend Architecture

Audience: maintainers reviewing the current draft-day surface.

Scope: the local Python service and same-origin interactive page.

Trust boundary: Python owns all valuation and provenance; browser JavaScript
only collects state and renders the validated response.

## What This Is

`ffbayes dashboard --year 2026` starts a standard-library
`ThreadingHTTPServer` bound to `127.0.0.1`. It serves one self-contained page
and four JSON endpoints:

- `GET /api/status`: source freshness, coverage, and blocking errors;
- `GET /api/leagues`: sanitized stable profile metadata;
- `POST /api/board`: strict runtime state and a fresh Python-calculated payload;
- `POST /api/snapshot`: atomic local export of that validated payload.

## State flow

The service loads one `FreshInputs` snapshot for both leagues. State is keyed by
profile ID and contains draft slot, current overall pick, taken IDs, your IDs,
and queue IDs. A state mutation is committed only after `build_draft_board`,
payload creation, and provenance validation all succeed. A failed request cannot
replace the last valid state with a partial board.

## Browser responsibilities

The page provides loading and blocked states, league selection, read-only league
settings, slot/current-pick inputs, Taken/Mine/Queue actions, a board table,
provenance display, and snapshot export. It does not calculate scoring,
replacement levels, VOR, scarcity, ADP availability, or recommendations.

## Current path versus historical notes

The older React dashboard and staged `site/` surface are retained as research and
maintenance material only. They are not used by the current 2026 command and
must not be treated as a second valuation implementation or a fallback.
