# FFBayes: Fantasy Football Analytics Pipeline

A fantasy football analytics system that combines Monte Carlo simulations, Bayesian uncertainty modeling, and draft-utility decision modeling to generate draft-board recommendations.

## Dashboard Paths

### [Open the live draft dashboard](https://nicholas-camarda.github.io/ffbayes/)

- This is the GitHub Pages view of the dashboard for repo browsers.
- Local draft users should run `ffbayes draft-strategy` and open the generated `dashboard/index.html` shortcut instead.
- The local dashboard shortcut is created by `ffbayes draft-strategy`; it will not exist in a fresh clone until that command runs.
- Local runtime outputs live under `~/ProjectsRuntime/ffbayes/` by default.
- If you need a different runtime root, set `FFBAYES_RUNTIME_ROOT` explicitly before running the CLI.

## Quick Start

### Step 1: Setup
```bash
# Clone
git clone https://github.com/nicholas-camarda/ffbayes.git
cd ffbayes

# Create/activate conda environment (first time only)
conda env create -f environment.yml
conda activate ffbayes

# Install
pip install -e .
```

### Step 2: Configure Your League
Edit `config/user_config.json` to set your preferences:

```json
{
  "league_settings": {
    "draft_position": 10,      // Your draft position (1-16)
    "league_size": 10,         // League size (8, 10, 12, 14, 16)
    "scoring_type": "PPR",     // Scoring type
    "ppr_value": 0.5,         // Points per reception
    "risk_tolerance": "medium" // low/medium/high
  }
}
```

### Step 3: Run the Draft Helper
The unified CLI exposes the current commands through `ffbayes` and `ffbayes-cli`.

```bash
# Draft-day workflow
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
```

### Command Reference
- `ffbayes collect`: downloads raw season data and refreshes the runtime season datasets.
- `ffbayes validate`: checks the collected data for completeness and freshness, and fails closed if the latest expected season is missing unless you explicitly opt into degraded execution.
- `ffbayes preprocess`: builds the analysis-ready combined dataset used by downstream models.
- `ffbayes pre-draft`: runs the full pre-draft workflow, including collection, validation, preprocessing, VOR, hybrid modeling, draft-board generation, backtesting, and visualizations.
- `ffbayes draft-strategy`: generates the draft board workbook, dashboard payload, HTML fallback, and compatibility JSON for the current league settings.
- `ffbayes refresh-dashboard`: regenerates dashboard HTML from an existing payload without rerunning draft modeling, and supports `--check --json` for machine-readable drift validation.
- `ffbayes draft-backtest`: backtests draft decision strategies against historical season data.
- `ffbayes compare-strategies`: compares draft strategy variants and summarizes how they differ.
- `ffbayes bayesian-vor`: compares the Bayesian outputs with traditional VOR rankings.
- `ffbayes split`: runs the supported split pipeline directly if you want to bypass the convenience shortcuts.
- `ffbayes pipeline`: runs the full end-to-end pipeline in one command.

The module-level commands still work too, but the top-level CLI is the supported entry point.

The collection step uses `nflreadpy` through a thin adapter that converts Polars frames to pandas before they reach the rest of the pipeline, so the collector output schema stays the same. The collector refreshes season files on each run instead of reusing older CSV output, and the latest-season freshness check is schema-aware rather than depending on backend-specific error text.

Freshness policy: the supported `pre_draft` workflow now fails closed by default when the latest expected season is missing. If you intentionally want a degraded analysis window, set `FFBAYES_ALLOW_STALE_SEASON=true` for that run and expect the resulting manifests, dashboard payload, and Pages staging provenance to report the degraded state explicitly.

The split pipeline runs collection in summary mode, so it shows the yearly collection summary without printing per-row progress. If you want the more verbose progress output, run `ffbayes collect` directly.

Maintenance note: if nflverse changes the underlying Python loaders again, keep the adaptation logic inside `src/ffbayes/data_pipeline/nflverse_backend.py` so the rest of the pipeline stays pandas-first.

### Outputs
- Draft board workbook: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.xlsx`
- Dashboard payload: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- HTML dashboard: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html`
- Convenience dashboard shortcut (repo root): `dashboard/index.html`
- Convenience dashboard shortcut (runtime): `~/ProjectsRuntime/ffbayes/dashboard/index.html`
- Live dashboard: [nicholas-camarda.github.io/ffbayes](https://nicholas-camarda.github.io/ffbayes/) serving the staged dashboard from `site/index.html`
- Decision backtest: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/draft_decision_backtest_<year_range>.json`
- Pages publish provenance: `site/publish_provenance.json` after `ffbayes publish-pages`

### Optional Publishing
- `ffbayes publish --year <year>` copies selected runtime artifacts into the cloud mirror for long-term storage.
- `ffbayes refresh-dashboard --year <year>` refreshes the runtime dashboard HTML from the current payload when template code changes but the payload is still authoritative.
- `ffbayes refresh-dashboard --check --json --payload-path <payload> --output-html <html>` reports whether a dashboard HTML target is fresh or stale relative to regeneration from the specified payload.
- `ffbayes publish-pages --year <year>` stages the current dashboard HTML and payload into `site/` for GitHub Pages and records publish-time provenance for the staged dashboard.

## Dashboard Paths

### GitHub Repo Viewers (No Clone)

- Open the live dashboard here: [nicholas-camarda.github.io/ffbayes](https://nicholas-camarda.github.io/ffbayes/)
- The intended experience is the repo's **GitHub Pages** site, which serves the most recently staged `site/index.html`.
- Pages only updates when someone runs `ffbayes publish-pages` and commits/pushes the updated `site/` directory to `master` (the workflow deploys `./site`).

### Local Users (Repo Cloned)

- Run `ffbayes draft-strategy ...` and then open the repo-root shortcut: `dashboard/index.html`
- Also available is the shallow, stable runtime dashboard: `~/ProjectsRuntime/ffbayes/dashboard/index.html`
- The local dashboard shortcut is generated by `ffbayes draft-strategy`; it will not exist in a fresh clone until that command runs.
- If the dashboard template changes but the runtime payload is still current, run `ffbayes refresh-dashboard --year <year>` instead of rerunning the full draft strategy.
- If you want the GitHub Pages version, stage it with `ffbayes publish-pages --year <year>` and then open `site/index.html`.

### Step 4: Use During Draft
- Open `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.xlsx`
- Use `dashboard/index.html` or `~/ProjectsRuntime/ffbayes/dashboard/index.html` for the live draft helper
- Follow the pick-by-pick recommendations in the workbook
- Use backup options if primary targets are gone
- Review the dashboard's `Decision evidence` and `Freshness and provenance` panels before treating the board as trustworthy

### Step 5: After the Draft
The supported CLI surface now stops at the pre-draft command center. Keep the runtime artifacts if you want a local audit trail, or mirror the supported pre-draft outputs with `ffbayes publish`.

---

## Modeling Approach

### Hybrid Monte Carlo and Bayesian Model
- Monte Carlo: uses 5 years of NFL performance data with 5000 simulations
- Bayesian: adds uncertainty modeling with confidence intervals
- Generalization: handles new or previously unseen players through pattern learning
- Data integrity: uses position-aware fuzzy matching for name resolution

### Generalization for New or Sparse Players
Compared with models that depend heavily on historical player-level volume, FFBayes can:
- Evaluate rookies using position patterns and team context
- Handle limited data by sampling from comparable historical cases
- Adapt to team changes using broader historical patterns
- Quantify uncertainty even when direct player history is limited

### VOR and Draft Utility
- Traditional VOR: value over replacement as a baseline ranking method
- Additional signals: position scarcity, team construction, and risk management
- Result: draft recommendations that incorporate more than rank order alone

---

## Output Organization

### Pre-Draft Outputs
Runtime working tree: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/`

- `artifacts/vor_strategy/` - VOR rankings and draft guide outputs
- `artifacts/draft_strategy/` - Draft board workbook, dashboard payload, HTML dashboard, and decision backtest
- `artifacts/hybrid_mc_bayesian/` - Monte Carlo + Bayesian model outputs
- `diagnostics/` - Rendered plots and diagnostics (strategy comparison, slot sensitivity, etc.)
- Repo-local `site/` - GitHub Pages dashboard root copied from the canonical HTML artifact

Published mirror: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/results/<year>/pre_draft/` after `ffbayes publish`

### Visualizations
Runtime plots: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/diagnostics/`

- `pre_draft/` - Strategy comparison, draft-score diagnostics, freshness views, and slot sensitivity

Published plots: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/plots/<year>/pre_draft/` after `ffbayes publish`
Published preview images: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/docs/images/` after `ffbayes publish`

---

## Configuration

### Main Configuration: `config/user_config.json`
```json
{
  "league_settings": {
    "draft_position": 10,
    "league_size": 10,
    "scoring_type": "PPR",
    "ppr_value": 0.5,
    "risk_tolerance": "medium"
  },
  "vor_settings": {
    "ppr": 0.5,
    "top_rank": 120
  },
  "model_settings": {
    "monte_carlo_simulations": 5000,
    "historical_years": 5,
    "confidence_level": 0.95
  }
}
```

### Pipeline Configuration
- `config/pipeline_config.json` defines the full end-to-end pipeline.
- `config/pipeline_pre_draft.json` drives the split runner.

The current pipeline steps are:
1. Data collection
2. Data validation
3. Data preprocessing
4. Traditional VOR draft strategy
5. Unified dataset creation
6. Hybrid Monte Carlo + Bayesian analysis
7. Draft decision strategy
8. Draft strategy comparison
9. Pre-draft visualizations

---

## Draft Outputs

### Primary Draft Outputs
- Draft board workbook: `draft_board_<year>.xlsx` with board, by-position, my picks, tier cliffs, availability, targets by round, roster scenarios, player notes, diagnostics, freshness, and backtest summary
- Dashboard payload: `dashboard_payload_<year>.json` for the local interactive dashboard
- HTML fallback: `draft_board_<year>.html` for browser access without a notebook or app shell
- Decision backtest: `draft_decision_backtest_<year_range>.json` comparing draft strategies on the same targets

### Trust Surfaces
- Decision evidence panel: the dashboard now summarizes internal holdout backtest evidence, season-level deltas, and interpretation limits as a dedicated decision-support surface
- Freshness and provenance: runtime manifests and staged Pages payloads expose freshness status, missing years, and whether a degraded override was explicitly used
- Publish-time provenance: `ffbayes publish-pages` records when the staged Pages site was generated and from which dashboard artifacts it was built
- Dashboard lifecycle checks: `ffbayes refresh-dashboard --check --json` emits a machine-readable stale/fresh result so CI and local operators can verify whether a target HTML file still matches regeneration from its authoritative payload

### Dashboard Lifecycle Ownership
- `ffbayes draft-strategy` owns canonical runtime payload generation and the primary runtime HTML artifact.
- `ffbayes refresh-dashboard` owns cheap HTML regeneration and non-mutating drift checks from an existing authoritative payload.
- `ffbayes publish-pages` owns staging the validated dashboard bundle into repo-tracked `site/` for GitHub Pages.
- The dedicated dashboard-sync validation workflow checks `site/` before deployment; `.github/workflows/pages.yml` remains deployment-focused.

### How the Draft Score Works

The main decision number is a weighted utility score, not just a projection rank:

```text
draft_score =
  0.34 * z(starter_delta)
  + 0.20 * z(replacement_delta)
  + 0.16 * z(proj_points_mean)
  + 0.12 * z(availability_at_pick)
  + 0.10 * z(upside_score)
  + 0.08 * z(starter_need)
  + 0.08 * z(position_scarcity)
  - 0.25 * z(fragility_score)
  + 0.06 * z(market_gap)
```

Plain English:
- `starter_delta` tells you how much better the player is than a likely starter at that position.
- `availability_at_pick` estimates whether the player survives to your next turn.
- `upside_score` rewards ceiling and lineup leverage.
- `fragility_score` penalizes injury, role, and uncertainty risk.
- `market_gap` catches players the market is underrating relative to the model.

Interpretation guide:
- High `draft_score` means "best overall draft decision now."
- High `availability_at_pick` means "safe to wait."
- High `upside_score` with high `fragility_score` means "boom-bust upside."
- High `why_flags` density means the player is unusual enough to inspect manually.

### Publication
- Runtime outputs stay local by default.
- To mirror selected results into the cloud workspace, run `ffbayes publish --year <year>`.

---

## Usage Examples

### Basic 10-Team League (Position 10)
```bash
ffbayes pre-draft
```

### 12-Team League (Position 5, Full PPR)
```bash
# Update config/user_config.json:
# "league_size": 12, "draft_position": 5, "ppr_value": 1.0
ffbayes pre-draft
```

### After-Draft Analysis
```bash
# The supported CLI surface stops at the pre-draft command center.
```

### Other Common Commands
```bash
ffbayes collect --years 2021,2022,2023 --allow-stale-season
ffbayes preprocess
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
ffbayes draft-backtest
ffbayes compare-strategies
ffbayes bayesian-vor
ffbayes publish --year 2025
ffbayes publish-pages --year 2025
```

---

## Troubleshooting

### Common Issues

#### "No players found in database"
- Problem: player names do not match the database format
- Solution: use full names, for example `Patrick Mahomes` instead of `P. Mahomes`
- Name resolution support includes initials, suffix normalization, and position-aware fuzzy matching

#### "Pipeline failed with errors"
- Problem: a critical step failed
- Solution: check logs in `~/ProjectsRuntime/ffbayes/logs/` for detailed error information
- Verify draft-board inputs and league settings before rerunning the pre-draft workflow

#### "Missing required columns"
- Problem: the team file has incorrect column names
- Solution: use `Name`, `Position`, `Team` columns, or `POS`, `PLAYER`, `BYE` for the legacy format
- Default path: `~/ProjectsRuntime/ffbayes/data/raw/my_ff_teams/drafted_team_<year>.tsv`

### Team File Format
Your team file should have these columns:
```tsv
Name	Position	Team
Patrick Mahomes	QB	KC
James Cook	RB	BUF
# ... your full team
```

Legacy format also supported:
```tsv
POS	PLAYER	BYE
QB	Patrick Mahomes	10
RB	James Cook	7
# ... your full team
```

---

## Technical Documentation

For detailed technical information:
- [Documentation Index](docs/README.md) - Navigation for the docs folder
- [Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE.md) - How the models work
- [Output Examples](docs/OUTPUT_EXAMPLES.md) - Usage examples

---

## Contributing

We welcome contributions. Follow the repository guidance in [AGENTS.md](AGENTS.md) and keep changes aligned with the current CLI, path constants, and test coverage.

Before submitting changes, run `ruff check .`, `ruff format .`, and `conda run -n ffbayes pytest -q`.

---

## Visualizations

The visualization layer produces the draft board HTML dashboard, workbook-backed audit tables, and the supporting validation plots.

Current outputs:
- Draft board workbook and dashboard payload in the pre-draft runtime tree
- GitHub Pages dashboard root in `site/index.html`
- Published plots and preview images only after `ffbayes publish`

The dashboard is designed to answer four draft-day questions quickly:
- Who are the best players right now?
- Who is likely to survive to my next pick?
- What position run risk should I care about?
- Which strategy has been working best in the backtest?

## License

MIT License.
