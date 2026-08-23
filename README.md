# FFBayes

FFBayes is a reproducible fantasy-football draft board. It combines public
current-season projections and market data with explicit league settings to
produce a local, interactive board.

## Quick start

```bash
conda env create -f environment.yml
conda activate ffbayes
pip install -e .
ffbayes dashboard --year 2026
```

The command fetches the required public inputs, validates them, starts a local
dashboard, and opens it in the browser. It does not connect to a fantasy
platform, require an account, or use private-league credentials.

## Configure a league

The repository includes [`config/leagues/example_2026.json`](config/leagues/example_2026.json)
as a complete, portable example. It is intentionally not tied to a real league.
Copy it to a local-only file whose name ends in `.local.json`, then edit the
league name, team count, draft format, scoring, roster slots, FLEX eligibility,
bench, IR, and waiver settings. Local profile files are ignored by Git.

```bash
cp config/leagues/example_2026.json config/leagues/my-league.local.json
ffbayes dashboard --year 2026
```

When local profiles are present, the command loads them automatically. You can
also pass one or more profiles explicitly:

```bash
ffbayes dashboard --year 2026 \
  --profile config/leagues/my-league.local.json
```

Draft slot and current overall pick are entered in the dashboard at draft time;
they are not stored in the public example profile.

## What the command does

1. Fetches the current-season ESPN fantasy player, projection, and ADP feed.
2. Fetches the matching nflverse roster release for current-player checks.
3. Rejects wrong-season, inactive, duplicate, shallow, stale, or incomplete
   inputs before valuation.
4. Applies the selected league's scoring and roster rules.
5. Calculates replacement levels, VOR, scarcity, market blend, and snake-draft
   timing.
6. Serves the board locally. Browser actions are sent back to the Python model.

If a required source or league setting is unavailable, the dashboard reports the
failure and does not substitute stale data, neutral market values, or guessed
availability.

## Model summary

For each player, the engine first converts projected stat lines into league
points:

\[
P_i = \sum_s x_{i,s} w_s + \text{configured bonuses}
\]

Roster demand is built from required starters, an optimization of FLEX slots
across eligible positions, and bench demand allocated in current ADP order. The
replacement level for position \(p\) is the projected score at the last roster
slot required for that position:

\[
R_p = P_{p,(D_p)}
\]

where \(D_p\) is league-wide demand. Value over replacement is:

\[
\operatorname{VOR}_i = P_i - R_{p(i)}
\]

The board combines standardized VOR and local positional drop-off, then blends
that model signal with ADP rank. If a draft slot is supplied, next-pick
availability uses an ADP-centered normal approximation with a minimum standard
deviation; without a slot, timing is explicitly marked `slot_required`.

These are decision-support quantities, not guarantees of player performance.
The complete definitions and assumptions are in
[`docs/METRIC_REFERENCE.md`](docs/METRIC_REFERENCE.md) and
[`docs/TECHNICAL_DEEP_DIVE.md`](docs/TECHNICAL_DEEP_DIVE.md).

## Inputs and outputs

Required sources are public:

- ESPN current-season fantasy player/projection/ADP feed
- nflverse current-season roster release, accessed through `nflreadpy`

Required fetches use cache mode `off`. Each run writes timestamped artifacts
under `runtime/runs/dashboard_2026/`, including source manifests, coverage
statistics, board payloads, and optional snapshots. Runtime artifacts are local
and ignored by Git.

## Development checks

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src mypy src/ffbayes
PYTHONPATH=src ruff check src tests
npm ci
npx playwright install chromium
PYTHON=/path/to/ffbayes/bin/python node tests/test_draft_2026_dashboard_browser.mjs
PYTHON=/path/to/ffbayes/bin/python npm test
```

The root `npm test` and `npm run smoke` commands run the deterministic browser
smoke against the Python fixture service. Set `PYTHON` to the executable from
the environment where the project dependencies are installed when `python` is
not the desired interpreter.

The public operator surface is the single `ffbayes dashboard` command. Lower-
level modules remain available for development and testing.

## Documentation

- [`docs/DASHBOARD_OPERATOR_GUIDE.md`](docs/DASHBOARD_OPERATOR_GUIDE.md):
  running and operating the dashboard.
- [`docs/DATA_LINEAGE_AND_PATHS.md`](docs/DATA_LINEAGE_AND_PATHS.md): inputs,
  artifacts, and reproducibility.
- [`docs/METRIC_REFERENCE.md`](docs/METRIC_REFERENCE.md): field definitions and
  formulas.
- [`docs/TECHNICAL_DEEP_DIVE.md`](docs/TECHNICAL_DEEP_DIVE.md): implemented
  pipeline and mathematical details.
- [`docs/OUTPUT_EXAMPLES.md`](docs/OUTPUT_EXAMPLES.md): generic payload shape.

## License

MIT License.
