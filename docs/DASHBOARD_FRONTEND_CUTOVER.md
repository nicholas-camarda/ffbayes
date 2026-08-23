# Dashboard Frontend Cutover Notes

Audience: maintainers of historical dashboard artifacts.

Scope: the transition to the current loopback 2026 dashboard.

Trust boundary: the current service is authoritative; staged pages and prior
payloads are not inputs.

## What This Is

The current operator surface is the interactive page served by:

```bash
ffbayes dashboard --year 2026
```

It is intentionally self-contained and served by Python's standard library.
Node is not required at draft time. The page calls only same-origin loopback
endpoints and displays Python-generated values.

## Historical material

Earlier frontend work produced a React single-file template and staged `site/`
pages. Those artifacts remain useful for frontend tests and research history,
but they are not the current 2026 board, are not a fallback, and are not opened
by the operator command.

## Verification

The current browser smoke test starts a fixture-backed loopback service, enters
different slots for both profiles, marks a player taken, switches leagues,
rejects an invalid slot, and exports a snapshot. The Python service tests cover
the same boundary without a browser.
