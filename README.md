# FFBayes: Fantasy Football Analytics Pipeline

A fantasy football analytics system that combines Monte Carlo simulations, Bayesian uncertainty modeling, and draft-utility decision modeling to generate draft-board recommendations.

## See It Live

The clearest example of what this repo produces is the live draft war room dashboard:

- Public GitHub Pages dashboard: https://nicholas-camarda.github.io/ffbayes/site/
- Project root redirect: https://nicholas-camarda.github.io/ffbayes/

If you are trying to understand the product before reading the workflow docs, start there.

## End-to-End Usage

This repo has one supported draft workflow:

1. Run the pre-draft pipeline.
2. Open the local dashboard shortcut.
3. Use the dashboard and workbook during the draft.
4. Optionally stage `site/` for GitHub Pages.
5. Import the finalized draft bundle after the draft.
6. Run the retrospective once real outcomes are available.

If you remember only one thing, it is this:

```bash
ffbayes pre-draft
```

That is the supported command that refreshes the full pre-draft analysis stack and produces the live draft artifacts.

## Quick Start

### 1. Set Up the Environment

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

You can also override some settings from the CLI, for example:

```bash
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
```

### 3. Run the Pre-Draft Pipeline

```bash
ffbayes pre-draft
```

`ffbayes pre-draft` is the main entry point for preparing the board. It runs the supported pre-draft workflow:

- collect raw season data
- validate freshness and completeness
- preprocess the analysis-ready dataset
- build VOR and hybrid model artifacts
- generate the draft board, dashboard payload, and dashboard HTML
- run the draft decision backtest used by the decision-evidence surface

Freshness policy:

- By default, the workflow fails closed if the latest expected season is missing.
- If you intentionally want a degraded run, set `FFBAYES_ALLOW_STALE_SEASON=true` for that invocation.
- When degraded execution is allowed, the manifests and dashboard provenance should say so explicitly.

### 4. Open the Right Dashboard

After `ffbayes pre-draft` or `ffbayes draft-strategy`, use the repo-local shortcut:

- Local dashboard shortcut: `dashboard/index.html`
- Paired local payload: `dashboard/dashboard_payload.json`

This is the supported local draft surface.

Do not treat `site/index.html` as your draft-day local surface. `site/` is the staged GitHub Pages copy, not the authoritative working dashboard.

### 5. Use the Draft Artifacts During the Draft

The main outputs are:

- Workbook: `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.xlsx`
- Canonical payload: `runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- Canonical HTML: `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html`
- Repo-local shortcut: `dashboard/index.html`

During the draft:

- keep the local dashboard open
- use the workbook for tabular review and backup planning
- check the `Decision evidence` panel before over-trusting recommendations
- check `Freshness and provenance` before assuming the board is current
- when you click Finalize, keep the downloaded finalized bundle

### 6. Publish the Dashboard Only If You Need GitHub Pages

If you want the repo’s public Pages site updated, stage `site/` explicitly:

```bash
ffbayes publish-pages --year 2026
```

This copies the validated dashboard bundle into repo-tracked `site/` and writes publish provenance to `site/publish_provenance.json`.

Important distinction:

- `dashboard/index.html` is the local working shortcut
- `site/index.html` is the staged Pages copy

GitHub Pages only updates after the staged `site/` files are committed and pushed.

### 7. Refresh HTML Without Rerunning the Models

If the payload is still authoritative and only the dashboard template changed, regenerate HTML without rerunning the full analysis:

```bash
ffbayes refresh-dashboard --year 2026
```

If you want to restage Pages immediately after refreshing:

```bash
ffbayes refresh-dashboard --year 2026 --stage-pages
```

If you want to verify whether an HTML surface is stale relative to its payload:

```bash
ffbayes refresh-dashboard --check --json \
  --payload /path/to/dashboard_payload.json \
  --output-html /path/to/index.html
```

### 8. After the Draft: Import the Finalized Bundle

The supported post-draft path is to import the finalized files downloaded from the local dashboard into the canonical runtime folder:

```bash
ffbayes draft-retrospective \
  --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* \
  --ingest-only \
  --year 2026
```

Those imported artifacts live under:

```text
runs/<year>/pre_draft/artifacts/draft_strategy/finalized_drafts/
```

### 9. Run the Retrospective When Outcomes Exist

Once the season outcomes are available in the unified dataset:

```bash
ffbayes draft-retrospective --year 2026
```

That command auto-discovers the imported finalized JSON and writes retrospective JSON and HTML next to the other draft-strategy artifacts.

## Command Guide

Use these commands by intent:

- `ffbayes pre-draft`: full supported pre-draft workflow
- `ffbayes draft-strategy`: regenerate the board and dashboard from current processed inputs and current league settings
- `ffbayes refresh-dashboard`: rebuild HTML from an existing authoritative payload
- `ffbayes publish-pages`: stage the current dashboard into repo-tracked `site/`
- `ffbayes draft-retrospective`: import finalized draft bundles and later evaluate them against realized outcomes
- `ffbayes collect`, `validate`, `preprocess`: lower-level pipeline steps for debugging or manual recovery

The top-level `ffbayes` CLI is the supported interface. Module-level commands still exist, but the unified CLI is the intended operator surface.

## Output Paths

### Local and Runtime Artifacts

- Draft board workbook: `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.xlsx`
- Dashboard payload: `runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- HTML dashboard: `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html`
- Decision backtest: `runs/<year>/pre_draft/artifacts/draft_strategy/draft_decision_backtest_<year_range>.json`
- Finalized draft bundle folder: `runs/<year>/pre_draft/artifacts/draft_strategy/finalized_drafts/`
- Repo-local dashboard shortcut: `dashboard/index.html`
- Runtime dashboard shortcut: `dashboard/index.html` under the configured runtime root

### Published Surfaces

- GitHub Pages site: [nicholas-camarda.github.io/ffbayes](https://nicholas-camarda.github.io/ffbayes/)
- Staged Pages root in the repo: `site/index.html`
- Staged Pages payload in the repo: `site/dashboard_payload.json`
- Pages publish provenance: `site/publish_provenance.json`

### Runtime Root

Runtime outputs are written under the configured runtime root.

- Default runtime tree: `~/ProjectsRuntime/ffbayes`
- Override only when you need to: `FFBAYES_RUNTIME_ROOT=/path/to/runtime`

If you override the runtime root, do it before running the CLI so collection, manifests, dashboard artifacts, and post-draft imports all agree on the same path base.

## Draft-Day Notes

- Open `dashboard/index.html`, not `site/index.html`
- Treat the local dashboard shortcut as the primary decision surface
- Use the workbook as a companion artifact, not the source of freshness truth
- Review freshness before relying on the board
- Review evidence before treating the score ordering as gospel
- Keep the finalized downloads after the draft so you can import them cleanly

## Common Patterns

### First-time setup and full run

```bash
conda activate ffbayes
ffbayes pre-draft
open dashboard/index.html
```

### League settings changed, but processed data is still current

```bash
conda activate ffbayes
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
open dashboard/index.html
```

### Payload is current, HTML template changed

```bash
conda activate ffbayes
ffbayes refresh-dashboard --year 2026
open dashboard/index.html
```

### Update GitHub Pages after a fresh run

```bash
conda activate ffbayes
ffbayes pre-draft
ffbayes publish-pages --year 2026
```

### Import finalized draft files after the draft

```bash
conda activate ffbayes
ffbayes draft-retrospective \
  --import-finalized ~/Downloads/ffbayes_finalized_*_2026_* \
  --ingest-only \
  --year 2026
```

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
Runtime working tree: `runs/<year>/pre_draft/` under the configured runtime root

- `artifacts/vor_strategy/` - VOR rankings and draft guide outputs
- `artifacts/draft_strategy/` - Draft board workbook, dashboard payload, HTML dashboard, and decision backtest
- `artifacts/hybrid_mc_bayesian/` - Monte Carlo + Bayesian model outputs
- `diagnostics/` - Supplemental diagnostics that do not replace the canonical dashboard artifacts
- Repo-local `site/` - GitHub Pages dashboard root copied from the canonical HTML artifact

Published cloud snapshot: `Analysis/<date>/` under the configured synced project home after `ffbayes publish`
Published stable data: `data/` under the configured synced project home after `ffbayes publish`

### Visualizations
The supported visualization product is the live draft dashboard plus its staged Pages copy. Supplemental diagnostics under `runs/<year>/pre_draft/diagnostics/` may exist, but they are not a separate supported decision surface.

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
8. Draft decision backtest

---

## Draft Outputs

### Primary Draft Outputs
- Draft board workbook: `draft_board_<year>.xlsx` with board, by-position, my picks, tier cliffs, availability, targets by round, roster scenarios, player notes, diagnostics, freshness, and backtest summary
- Dashboard payload: `dashboard_payload_<year>.json` for the local interactive dashboard
- HTML fallback: `draft_board_<year>.html` for browser access without a notebook or app shell
- Decision backtest: `draft_decision_backtest_<year_range>.json` comparing draft strategies on the same targets
- Imported finalized bundles: `finalized_drafts/ffbayes_finalized_*` under the year-scoped `draft_strategy/` runtime directory
- Draft retrospective: `draft_retrospective_<year>.json` and `draft_retrospective_<year>.html` comparing finalized draft artifacts against realized season outcomes when those outcomes are available

### Trust Surfaces
- Decision evidence panel: the dashboard now summarizes internal holdout backtest evidence, season-level deltas, and interpretation limits as a dedicated decision-support surface
- Freshness and provenance: runtime manifests and staged Pages payloads expose freshness status, missing years, and whether a degraded override was explicitly used
- Publish-time provenance: `ffbayes publish-pages` records when the staged Pages site was generated and from which dashboard artifacts it was built
- Dashboard lifecycle checks: `ffbayes refresh-dashboard --check --json` emits a machine-readable stale/fresh result so CI and local operators can verify whether a target HTML file still matches regeneration from its authoritative payload
- Draft retrospective evaluation: `ffbayes draft-retrospective` imports finalized bundles into a canonical runtime folder and uses those finalized artifacts plus realized season outcomes to evaluate expected-versus-realized roster performance, while treating follow/pivot audit metrics as secondary context

### Dashboard Lifecycle Ownership
- `ffbayes draft-strategy` owns canonical runtime payload generation and the primary runtime HTML artifact.
- `ffbayes refresh-dashboard` owns cheap HTML regeneration and non-mutating drift checks from an existing authoritative payload.
- `ffbayes publish-pages` owns staging the validated dashboard bundle into repo-tracked `site/` for GitHub Pages.
- `ffbayes draft-retrospective` owns the canonical runtime `finalized_drafts/` ingest path plus runtime-local post-draft evaluation artifacts, and it does not publish to `site/`.
- The dedicated dashboard-sync validation workflow checks `site/` before deployment; `.github/workflows/pages.yml` remains deployment-focused.

### How the Board Value Score Works

The main board value number is the internal `draft_score` utility field, not just a projection rank:

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
- High board value score means "best overall draft decision now."
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
ffbayes collect
ffbayes preprocess
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
ffbayes draft-backtest
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
- Solution: check the configured runtime `logs/` directory for detailed error information
- Verify draft-board inputs and league settings before rerunning the pre-draft workflow

#### "Missing required columns"
- Problem: the team file has incorrect column names
- Solution: use `Name`, `Position`, `Team` columns, or `POS`, `PLAYER`, `BYE` for the legacy format
- Explicit input required: pass `--team-file <path>` or set `TEAM_FILE`; legacy implicit `my_ff_teams` defaults are deprecated

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

The local war room dashboard is designed as a decision aid, not just a ranked table. In addition to the recommendation lanes and player inspector, it now includes:
- a `Wait vs Pick Frontier` for timing tradeoffs between current value and next-pick survival
- a `Positional Cliffs` panel that highlights where a position group is about to fall off
- an inspector-first `Contextual vs baseline` explainer so disagreement with the simple VOR proxy stays attached to the selected player instead of living in a detached analytics view

The dashboard is designed to answer these draft-day questions quickly:
- Who are the best players right now?
- Who is likely to survive to my next pick?
- What position run risk or positional cliff should I care about?
- Which strategy has been working best in the backtest?

## License

MIT License.
