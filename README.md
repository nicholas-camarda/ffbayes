# FFBayes: Fantasy Football Draft Engine

A fantasy football draft engine that builds season-total player forecasts, converts them into draft-day recommendations, and renders a live draft dashboard plus workbook.

## See It Live

The clearest example of what this repo produces is the live draft war room dashboard:

- Public GitHub Pages dashboard: https://nicholas-camarda.github.io/ffbayes/site/
- Project root redirect: https://nicholas-camarda.github.io/ffbayes/

If you are trying to understand the product before reading the workflow docs, start there.

## End-to-End Usage

1. Run the pre-draft pipeline.
2. Open the local dashboard shortcut.
3. Use the workbook and dashboard during the draft.
4. Use one command to refresh and stage the public dashboard when needed.
5. Import the finalized draft bundle after the draft.
6. Run the retrospective once realized outcomes are available.

If you remember only one thing, use:

```bash
ffbayes pre-draft
```

That command refreshes the full pre-draft analysis stack and regenerates the live draft artifacts.

## Quick Start

### 1. Set Up The Environment

```bash
git clone https://github.com/nicholas-camarda/ffbayes.git
cd ffbayes

conda env create -f environment.yml
conda activate ffbayes
pip install -e .
```

All supported Python commands should run inside the `ffbayes` conda environment.

### 2. Configure Your League

Edit `config/user_config.json`:

```json
{
  "league_settings": {
    "draft_position": 10,
    "league_size": 10,
    "scoring_type": "PPR",
    "ppr_value": 0.5,
    "risk_tolerance": "medium"
  }
}
```

You can also override some settings from the CLI:

```bash
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
```

### 3. Run The Pre-Draft Pipeline

```bash
ffbayes pre-draft
```

`ffbayes pre-draft` is the main entry point for preparing the board. It runs:

- collect raw season data
- validate freshness and completeness
- preprocess the analysis-ready dataset
- build the VOR snapshot and the canonical player forecast inputs
- generate the draft board workbook, dashboard payload, and dashboard HTML
- run the internal draft-decision backtest used by the dashboard evidence surface

The draft-decision backtest is a required dashboard component. Production
dashboard generation fails closed unless decision evidence is available and fresh.

Freshness policy:

- by default, the workflow fails closed if the latest expected season is missing
- dashboard generation also fails if the draft-decision evidence is unavailable
  or degraded
- `FFBAYES_ALLOW_STALE_SEASON=true` is not a production dashboard path; use it
  only for non-production investigation where degraded evidence is explicitly
  acceptable

### 4. Open The Correct Dashboard

After `ffbayes pre-draft`, `ffbayes draft-strategy`, or `ffbayes refresh-dashboard`,
use the repo-local shortcut:

- local shortcut: `dashboard/index.html`
- paired shortcut payload: `dashboard/dashboard_payload.json`

This is a derived local draft surface regenerated from the canonical runtime
payload. Do not use it if it is stale relative to the authoritative payload.

Do not treat `site/index.html` as your draft-day working dashboard. `site/` is the staged GitHub Pages copy, not the authoritative local surface.

### 5. Use The Draft Artifacts During The Draft

Primary artifacts:

- workbook: `seasons/<year>/draft_strategy/draft_board_<year>.xlsx`
- authoritative payload: `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- authoritative HTML: `seasons/<year>/draft_strategy/draft_board_<year>.html`
- repo-local shortcut: `dashboard/index.html`

During the draft:

- keep the local dashboard open
- use the workbook as a tabular companion and fallback
- inspect `Decision evidence` before over-trusting the ranking
- treat `n/a` or `not estimable` validation entries as slices the board could not judge cleanly, not as measured zero relationships
- inspect `Freshness and provenance` before assuming the board is current
- if the current dashboard build includes war-room visuals, use them as decision aids, not as a separate model
- when you click Finalize, keep the downloaded finalized bundle

### 6. Refresh And Stage The Public Dashboard In One Step

If you want the repo’s public Pages site updated, use the one-step staging command:

```bash
ffbayes stage-dashboard --year 2026
```

That command:

- rebuilds dashboard HTML from the authoritative payload
- refreshes the repo-local shortcut surfaces
- stages repo-tracked `site/`
- writes publish provenance to `site/publish_provenance.json`

Important distinction:

- `dashboard/index.html` is the local working shortcut
- `site/index.html` is the staged Pages copy

GitHub Pages only updates after the staged `site/` files are committed and pushed.

### 7. Refresh HTML Without Rerunning Models

If the payload is still authoritative and only the dashboard template changed:

```bash
ffbayes refresh-dashboard --year 2026
```

If you also want the public Pages copy updated, prefer:

```bash
ffbayes stage-dashboard --year 2026
```

`ffbayes publish-pages --year 2026` still exists as a lower-level staging command, but it only copies the current runtime dashboard into `site/`. It does not rerender the dashboard first.

To verify whether an HTML surface is stale relative to its payload:

```bash
ffbayes refresh-dashboard --check --json \
  --payload-path /path/to/dashboard_payload.json \
  --output-html /path/to/index.html
```

### 8. Import The Finalized Bundle After The Draft

The post-draft path is to import finalized files downloaded from the local dashboard into the canonical runtime folder:

```bash
ffbayes draft-retrospective \
  --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* \
  --ingest-only \
  --year 2026
```

Imported finalized artifacts live under:

```text
seasons/<year>/draft_strategy/finalized_drafts/
```

### 9. Run The Retrospective When Real Outcomes Exist

Once realized season outcomes are available in the unified dataset:

```bash
ffbayes draft-retrospective --year 2026
```

That command auto-discovers imported finalized JSON and writes retrospective JSON and HTML next to the other draft-strategy artifacts.

## Command Guide

Use these commands by intent:

- `ffbayes pre-draft`: full pre-draft workflow
- `ffbayes draft-strategy`: regenerate the board and dashboard from current processed inputs and league settings
- `ffbayes stage-dashboard`: one-step refresh plus GitHub Pages staging for `site/`
- `ffbayes refresh-dashboard`: rebuild HTML from an existing authoritative payload
- `ffbayes publish-pages`: lower-level staging helper that copies the current dashboard into repo-tracked `site/`
- `ffbayes draft-retrospective`: import finalized draft bundles and later evaluate them against realized outcomes
- `ffbayes collect`, `ffbayes validate`, `ffbayes preprocess`: lower-level debugging or recovery steps

The top-level `ffbayes` CLI is the main interface. Module-level scripts still exist, but the unified CLI is the intended operator surface.

The repository uses the hierarchical empirical-Bayes estimator for player forecasts. There is no active sampled-Bayes workflow in the repository.

## Output Organization

### Authoritative Runtime Artifacts

Runtime outputs are written under the configured runtime root.

- runtime root: `<runtime-root>`
- override only when needed: `FFBAYES_RUNTIME_ROOT=/path/to/runtime`

Main pre-draft runtime tree:

- `seasons/<year>/vor_strategy/`
- `seasons/<year>/draft_strategy/`
- `seasons/<year>/diagnostics/`

Key draft artifacts:

- `seasons/<year>/draft_strategy/draft_board_<year>.xlsx`
- `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- `seasons/<year>/draft_strategy/draft_board_<year>.html`
- `seasons/<year>/draft_strategy/draft_decision_backtest_<year_range>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_<year>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_diagnostics_<year>.json`
- `seasons/<year>/draft_strategy/model_outputs/player_forecast/player_forecast_validation_<year_range>.json`
- `seasons/<year>/diagnostics/validation/player_forecast_validation_summary_<year_range>.json`
- `seasons/<year>/draft_strategy/finalized_drafts/`
- `seasons/<year>/draft_strategy/draft_retrospective_<year>.json`
- `seasons/<year>/draft_strategy/draft_retrospective_<year>.html`

### Derived Local And Published Surfaces

- runtime-root dashboard shortcut: `<runtime-root>/dashboard/index.html`
- runtime-root shortcut payload: `<runtime-root>/dashboard/dashboard_payload.json`
- repo-local dashboard shortcut: `dashboard/index.html`
- repo-local shortcut payload: `dashboard/dashboard_payload.json`
- staged Pages root: `site/index.html`
- staged Pages payload: `site/dashboard_payload.json`
- staged Pages provenance: `site/publish_provenance.json`

Authority levels:

- authoritative: canonical runtime payload plus canonical runtime HTML
- derived local shortcut: runtime-root `dashboard/` plus repo `dashboard/`
- staged Pages copy: repo `site/`

### Optional Cloud Publish Surfaces

`ffbayes publish --year <year>` mirrors selected runtime outputs into the configured cloud root:

- stable mirrored data under `data/`
- dated snapshots under `Analysis/<date>/`

This is optional and separate from GitHub Pages staging.

## Model At A Glance

The supported draft board is not just a plain Monte Carlo ranking and not just a simple VOR list.

At a high level:

- the player layer builds season-total posterior projections and uncertainty estimates from historical player performance, availability, and draft-time-safe features
- the board layer converts those projections into starter edge, replacement edge, fragility, upside, and market-gap signals
- the recommendation layer then decides whether to pick now or wait based on roster urgency, next-pick survival, and expected regret
- the dashboard keeps `Simple VOR proxy` as an explicit baseline comparison instead of pretending the contextual board is the only view

The exact technical details live in [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md).

## Trust Surfaces

The dashboard is designed as a decision aid, not an oracle.

Primary trust surfaces:

- `Decision evidence`: internal holdout backtest summaries, season-level deltas, and interpretation limits
- `Freshness and provenance`: runtime freshness state, warnings, and degraded-run disclosure
- `Publish provenance`: staged Pages metadata showing when `site/` was built and from which dashboard artifacts
- `refresh-dashboard --check --json`: machine-readable stale/fresh check for derived HTML surfaces

Interpretation limits:

- backtests are internal holdout evidence, not external validation
- board value ordering is a decision-support heuristic, not a causal claim about what will happen
- Pages staging is a derived copy, not the authoritative local working surface

## Additional Commands

These commands are available outside the default `ffbayes pre-draft` operator path:

- `ffbayes bayesian-vor`: generate rolling holdout forecast validation summaries
- `ffbayes mc`: run the standalone Monte Carlo roster analysis
- `ffbayes publish --year <year>`: mirror selected runtime artifacts into cloud storage

## Documentation

Start with the durable docs guide suite:

- [docs/README.md](docs/README.md)
- [docs/DASHBOARD_OPERATOR_GUIDE.md](docs/DASHBOARD_OPERATOR_GUIDE.md)
- [docs/LAYPERSON_GUIDE.md](docs/LAYPERSON_GUIDE.md)
- [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md)
- [docs/METRIC_REFERENCE.md](docs/METRIC_REFERENCE.md)
- [docs/DATA_LINEAGE_AND_PATHS.md](docs/DATA_LINEAGE_AND_PATHS.md)
- [docs/OUTPUT_EXAMPLES.md](docs/OUTPUT_EXAMPLES.md)

## Contributing

Follow the repository guidance in [AGENTS.md](AGENTS.md).

Before submitting changes, run:

```bash
ruff check .
ruff format .
conda run -n ffbayes pytest -q
```

## License

MIT License.
