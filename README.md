# FFBayes 2026 fantasy-football draft dashboard

FFBayes provides one draft-day workflow:

```bash
ffbayes dashboard --year 2026
```

That command fetches fresh public ESPN fantasy projections/ADP and the current
nflverse roster feed, validates them, starts a loopback-only dashboard, and
opens it in your browser. There is no account, session, private-league, or
credential dependency.

## Draft-day workflow

1. Run the command above.
2. Choose either league in the dashboard:
   - **Bill's Underbit… — Med School Friends:** 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX,
     1 D/ST, 1 K, 6 bench, 1 IR.
   - **Nicholas's Nif… — Camarda-Klein Family:** 1 QB, 2 RB, 2 WR, 1 TE,
     1 FLEX, 1 D/ST, 1 K, 7 bench, 1 IR.
3. Enter that league's draft slot when the draft room reveals it. Enter the
   current overall pick explicitly; the initial pick defaults to the entered
   slot (round one) and can be edited at any time.
4. Use the row controls to mark players taken, mark your players, and maintain
   a queue. Every change is sent back to the Python model and recalculated.
5. Export a local snapshot when you want a durable record of the board and
   state.

League settings are explicit, read-only profile data. The draft slot and draft
state live only in the dashboard run and exported snapshots; profile JSON is not
mutated.

## Trust boundary

The board is fail-closed. The run stops with a visible blocked state when the
public sources are unavailable, the season is wrong, the player universe is
not current, projections or ADP are too shallow/stale, replacement demand
cannot be filled, league settings are incomplete, or provenance digests do not
match. There is no guessed ADP availability, neutral market value, stale-cache
rescue, or silent reuse of an older output.

The current-player universe is reconciled against nflverse and includes stable
ESPN IDs. Historical-only players cannot become actionable through the normal
data flow. The Python engine is the only implementation of scoring,
replacement levels, VOR, scarcity, availability, and recommendations; the
browser renders returned values and does not recalculate them.

## Fresh inputs and outputs

Each run uses cache mode `off` for required sources and creates a timestamped
directory under `runtime/runs/dashboard_2026/` containing source manifests,
coverage statistics, and any exported snapshots. A snapshot includes the
effective league settings, runtime slot/current pick, taken/your/queue IDs,
board rows, source digests, profile/state/board digests, code revision, and
generation time.

If ESPN or nflverse cannot be accessed or fails semantic validation, the exact
source and failure mode are shown in the dashboard. No alternate source is
silently substituted.

## Setup

```bash
conda env create -f environment.yml
conda activate ffbayes
pip install -e .
```

The package exposes only the `ffbayes` operator executable. Lower-level Python
modules remain available for tests and maintenance, but they are not alternate
draft-day workflows.

## Validation

Relevant checks include:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src mypy src/ffbayes
PYTHONPATH=src ruff check src tests
npm --prefix dashboard_frontend test
npm --prefix dashboard_frontend run typecheck
npm --prefix dashboard_frontend run build
node tests/test_draft_2026_dashboard_browser.mjs
```

See [docs/DASHBOARD_OPERATOR_GUIDE.md](docs/DASHBOARD_OPERATOR_GUIDE.md) for
the operational runbook, [docs/DATA_LINEAGE_AND_PATHS.md](docs/DATA_LINEAGE_AND_PATHS.md)
for provenance, and [docs/METRIC_REFERENCE.md](docs/METRIC_REFERENCE.md) for
metric definitions.

## License

MIT License.
