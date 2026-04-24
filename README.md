# FFBayes: Fantasy Football Draft Engine

A fantasy football draft engine that builds season-total player forecasts, converts them into draft-day recommendations, and renders a live draft dashboard plus workbook.

## See It Live

The clearest example of what this repo produces is the live draft war room dashboard:

- Live dashboard: https://nicholas-camarda.github.io/ffbayes/

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

### 2. Set The Starting League Defaults

Set default league settings in `config/user_config.json`:

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

These values seed the generated artifacts. League size, draft position, scoring
preset, and risk tolerance are editable from the dashboard controls while you
draft. Current pick advances from the players marked drafted; it is not a
manual setup field.

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

After `ffbayes pre-draft` or `ffbayes draft-strategy`, open:

- `dashboard/index.html`

That is the local draft-day dashboard. `site/` is only the staged public Pages
copy.

### 5. Use The Draft Artifacts During The Draft

Draft-day artifacts:

- dashboard: `dashboard/index.html`
- workbook: `seasons/<year>/draft_strategy/draft_board_<year>.xlsx`

During the draft:

- keep the local dashboard open
- use the workbook as a tabular companion and fallback
- inspect `Decision evidence` before over-trusting the ranking
- treat `n/a` or `not estimable` validation entries as slices the board could not judge cleanly, not as measured zero relationships
- inspect `Freshness and provenance` before assuming the board is current
- if the current dashboard build includes war-room visuals, use them as decision aids, not as a separate model
- when you click Finalize, keep the downloaded finalized bundle

### 6. Import The Finalized Bundle After The Draft

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

### 7. Run The Retrospective When Real Outcomes Exist

Once realized season outcomes are available in the unified dataset:

```bash
ffbayes draft-retrospective --year 2026
```

That command auto-discovers imported finalized JSON and writes retrospective JSON and HTML next to the other draft-strategy artifacts.

## Command Guide

Use these commands by intent:

- `ffbayes pre-draft`: full pre-draft workflow
- `ffbayes draft-strategy`: regenerate the board and dashboard from current processed inputs and league settings
- `ffbayes draft-retrospective`: import finalized draft bundles and later evaluate them against realized outcomes
- `ffbayes collect`, `ffbayes validate`, `ffbayes preprocess`: lower-level debugging or recovery steps

Internal developer dashboard refresh, Pages staging, and publish-surface checks are
covered in [docs/DASHBOARD_OPERATOR_GUIDE.md](docs/DASHBOARD_OPERATOR_GUIDE.md)
and [docs/DATA_LINEAGE_AND_PATHS.md](docs/DATA_LINEAGE_AND_PATHS.md).

The top-level `ffbayes` CLI is the main interface. Module-level scripts still exist, but the unified CLI is the intended operator surface.

The repository uses the hierarchical empirical-Bayes estimator for player forecasts.

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

The full shortcut, Pages, and cloud publish path contract lives in
[docs/DATA_LINEAGE_AND_PATHS.md](docs/DATA_LINEAGE_AND_PATHS.md).

## Model At A Glance

The supported draft board has three layers: player projection, board construction, and draft-action policy.

At a high level:

- the player layer builds season-total posterior projections and uncertainty estimates from historical player performance, availability, and draft-time-safe features
- the board layer converts those projections into starter edge, replacement edge, fragility, upside, and market-gap signals
- the recommendation layer then decides whether to pick now or wait based on roster urgency, next-pick survival, and expected regret
- the dashboard keeps `Simple VOR proxy` as an explicit baseline comparison beside the contextual board score

The exact technical details live in [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md).

## Trust Surfaces

The dashboard is designed as a decision aid, not an oracle.

Primary trust surfaces:

- `Decision evidence`: internal holdout backtest summaries, season-level deltas, and interpretation limits
- `Freshness and provenance`: runtime freshness state, warnings, and degraded-run disclosure

Interpretation limits:

- backtests are internal holdout evidence, not external validation
- board value ordering is a decision-support heuristic, not a causal claim about what will happen

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
