# FFBayes: Advanced Fantasy Football Analytics Pipeline

A sophisticated fantasy football analytics system that combines Monte Carlo simulations, Bayesian uncertainty modeling, and draft-utility decision modeling to generate draft-board recommendations and post-draft team analysis.

## Workspace Contract

- Source code lives in `~/Projects/ffbayes`
- Runtime data, logs, and per-run artifacts live in `~/ProjectsRuntime/ffbayes` unless you override `FFBAYES_RUNTIME_ROOT`
- Raw scraped inputs and processed analysis data stay in the runtime tree by default
- Published deliverables are mirrored into `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes` only when you run `ffbayes publish`
- `FFBAYES_PROJECT_ROOT`, `FFBAYES_RUNTIME_ROOT`, and `FFBAYES_CLOUD_ROOT` let you relocate the source, runtime, or publish trees

## 🚀 **Quick Start**

### **Step 1: Setup**
```bash
# Clone and install
git clone https://github.com/nicholas-camarda/ffbayes.git
cd ffbayes
pip install -e .

# Activate conda environment
conda activate ffbayes
```

### **Step 2: Configure Your League**
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

### **Step 3: Run the pipeline**
The unified CLI exposes the current commands through `ffbayes` and `ffbayes-cli`.

```bash
# Data collection and preprocessing
ffbayes collect --years 2021,2022,2023 --allow-stale-season
ffbayes validate
ffbayes preprocess

# Draft-day workflow
ffbayes pre-draft
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
ffbayes draft-backtest
ffbayes compare-strategies
ffbayes bayesian-vor

# Post-draft workflow
ffbayes validate-team
ffbayes post-draft

# Publish runtime outputs to the cloud mirror
ffbayes publish --year 2025 --phase pre_draft
```

### **Command Reference**
- `ffbayes collect`: downloads raw season data and refreshes the runtime season datasets.
- `ffbayes validate`: checks the collected data for completeness and freshness.
- `ffbayes preprocess`: builds the analysis-ready combined dataset used by downstream models.
- `ffbayes pre-draft`: runs the full pre-draft workflow, including collection, validation, preprocessing, VOR, hybrid modeling, draft-board generation, backtesting, and visualizations.
- `ffbayes draft-strategy`: generates the draft board workbook, dashboard payload, HTML fallback, and compatibility JSON for the current league settings.
- `ffbayes draft-backtest`: backtests draft decision strategies against historical season data.
- `ffbayes compare-strategies`: compares draft strategy variants and summarizes how they differ.
- `ffbayes bayesian-vor`: compares the Bayesian outputs with traditional VOR rankings.
- `ffbayes validate-team`: validates your drafted team file before post-draft analysis.
- `ffbayes post-draft`: runs the post-draft workflow, including team aggregation, Monte Carlo validation, and post-draft plots.
- `ffbayes publish`: copies selected runtime artifacts into the cloud mirror for long-term storage.
- `ffbayes split pre_draft` or `ffbayes split post_draft`: runs one phase of the split pipeline directly if you want to bypass the convenience shortcuts.
- `ffbayes pipeline`: runs the full end-to-end pipeline in one command.

The module-level commands still work too, but the top-level CLI is the supported entry point.

The collection step uses `nflreadpy` through a thin adapter that converts Polars frames to pandas before they reach the rest of the pipeline, so the collector output schema stays the same. The collector refreshes season files on each run instead of reusing older CSV output, and the latest-season freshness check is schema-aware rather than depending on backend-specific error text.

The split pipeline runs collection in summary mode, so it shows the yearly collection summary without printing per-row progress. If you want the more verbose progress output, run `ffbayes collect` directly.

Maintenance note: if nflverse changes the underlying Python loaders again, keep the adaptation logic inside `src/ffbayes/data_pipeline/nflverse_backend.py` so the rest of the pipeline stays pandas-first.

**What You Get:**
- 📊 **Draft Board Workbook**: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/results/draft_strategy/draft_board_<year>.xlsx`
- 🧠 **Dashboard Payload**: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/dashboard/dashboard_payload_<year>.json`
- 🌐 **HTML Fallback**: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/dashboard/draft_board_<year>.html`
- 📋 **Decision Backtest**: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/results/draft_strategy/draft_decision_backtest_<year_range>.json`
- 🎯 **Team Analysis**: `~/ProjectsRuntime/ffbayes/runs/<year>/post_draft/results/team_aggregation/team_analysis_results.json`
- 📈 **Season Projections**: `~/ProjectsRuntime/ffbayes/runs/<year>/post_draft/results/montecarlo_results/mc_projections_<year>_trained_on_<range>.tsv`

If you want the published mirror, run:
```bash
ffbayes publish --year <year> --phase pre_draft
```

### **Step 4: Use During Draft**
- Open `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/results/draft_strategy/draft_board_<year>.xlsx`
- Use `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/dashboard/draft_board_<year>.html` if you want a browser view
- Follow the pick-by-pick recommendations in the workbook
- Use backup options if primary targets are gone

### **Step 5: Post-Draft Analysis**
After drafting, save your team to `~/ProjectsRuntime/ffbayes/data/raw/my_ff_teams/drafted_team_<year>.tsv`:

```tsv
Name	Position	Team
Patrick Mahomes	QB	KC
James Cook	RB	BUF
# ... your full team
```

Then run:
```bash
ffbayes validate-team
ffbayes post-draft
```

If you want to mirror the runtime outputs into the cloud workspace, run `ffbayes publish --year <year> --phase post_draft`.

**What You Get:**
- 🎯 **Team Analysis**: Player contributions, team totals, and reliability
- 📊 **Season Projections**: Weekly score expectations with confidence intervals
- 🔍 **Monte Carlo Validation**: Validation artifacts and plots for the drafted roster
- 📋 **Team Summary**: Comprehensive analysis and recommendations

---

## 📊 **What Makes FFBayes Different**

### **🎲 Hybrid Monte Carlo + Bayesian Model**
- **Monte Carlo**: Uses 5 years of actual NFL performance data with 5000 simulations
- **Bayesian**: Adds intelligent uncertainty modeling with confidence intervals
- **Generalization**: Handles new/unknown players through pattern learning
- **Data Integrity**: Robust name resolution with position-aware fuzzy matching

### **🚀 Intelligent Generalization**
Unlike traditional models that fail with new players, FFBayes can:
- **Evaluate Rookies**: Project performance based on position patterns and team context
- **Handle Limited Data**: Use intelligent sampling for players with few games
- **Adapt to Changes**: Project veterans on new teams using historical patterns
- **Quantify Uncertainty**: Always provide confidence bounds, even for unknown players

### **📈 VOR + Advanced Analytics**
- **Traditional VOR**: Industry-standard value over replacement
- **Enhanced Analysis**: Position scarcity, team construction, risk management
- **Result**: Better draft decisions than rankings alone

---

## 📁 **Output Organization**

### **Pre-Draft Outputs** (Use During Draft)
Runtime working tree: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/`

- `results/vor_strategy/` - VOR rankings and draft guide outputs
- `results/draft_strategy/` - Canonical draft board workbook, compatibility JSON, and draft decision backtest
- `dashboard/` - Interactive dashboard payload and HTML fallback
- `results/hybrid_mc_bayesian/` - Monte Carlo + Bayesian model outputs

Published mirror: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/results/<year>/pre_draft/` after `ffbayes publish`

### **Post-Draft Outputs** (Use During Season)
Runtime working tree: `~/ProjectsRuntime/ffbayes/runs/<year>/post_draft/`

- `results/team_aggregation/` - Team analysis JSON and derived summaries
- `results/montecarlo_results/` - Season projections TSV files
- `plots/` - Post-draft charts and validation visuals

Published mirror: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/results/<year>/post_draft/` after `ffbayes publish`

### **Visualizations** (Analysis & Strategy)
Runtime plots: `~/ProjectsRuntime/ffbayes/runs/<year>/pre_draft/plots/` and `~/ProjectsRuntime/ffbayes/runs/<year>/post_draft/plots/`

- `pre_draft/` - Strategy comparison, draft-score diagnostics, freshness views, and slot sensitivity
- `post_draft/` - Team analysis charts

Published plots: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/plots/<year>/` after `ffbayes publish`
Published preview images: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/docs/images/` after `ffbayes publish`

---

## 🔧 **Configuration**

### **Main Configuration: `config/user_config.json`**
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

### **Pipeline Configuration**
- `config/pipeline_config.json` defines the full end-to-end pipeline.
- `config/pipeline_pre_draft.json` and `config/pipeline_post_draft.json` drive the split runner.
- The split runner forwards the active phase through `FFBAYES_PIPELINE_PHASE`.

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
10. Team aggregation
11. Monte Carlo validation
12. Post-draft visualizations

---

## 📊 **Visualizations**

### **Primary Draft Outputs**
- **Draft Board Workbook** - `draft_board_<year>.xlsx` with board, by-position, my picks, tier cliffs, availability, targets by round, roster scenarios, player notes, diagnostics, freshness, and backtest summary
- **Dashboard Payload** - `dashboard_payload_<year>.json` for the local interactive dashboard
- **HTML Fallback** - `draft_board_<year>.html` for browser access without a notebook or app shell
- **Decision Backtest** - `draft_decision_backtest_<year_range>.json` comparing draft strategies on the same targets

### **How the Draft Score Works**

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

### **Publication**
- Runtime outputs stay local by default.
- To mirror selected results into the cloud workspace, run `ffbayes publish --year <year> --phase pre_draft` or `--phase post_draft`.
- Use `--phase both` only when you want both run phases mirrored together.

### **Post-Draft Visualizations**
- **Team Composition Analysis** - Roster balance and depth assessment
- **Performance Projections** - Weekly scoring expectations with confidence intervals
- **Monte Carlo Validation** - Team performance distribution across simulations
- **Trade Analysis Tools** - Player value comparison and trade evaluation

---

## 🎯 **Usage Examples**

### **Basic 10-Team League (Position 10)**
```bash
ffbayes pre-draft
```

### **12-Team League (Position 5, Full PPR)**
```bash
# Update config/user_config.json:
# "league_size": 12, "draft_position": 5, "ppr_value": 1.0
ffbayes pre-draft
```

### **Post-Draft Analysis**
```bash
# After saving your team to the runtime raw data tree:
ffbayes validate-team
ffbayes post-draft
```

### **Other Common Commands**
```bash
ffbayes collect --years 2021,2022,2023 --allow-stale-season
ffbayes preprocess
ffbayes draft-strategy --draft-position 10 --league-size 10 --risk-tolerance medium
ffbayes draft-backtest
ffbayes compare-strategies
ffbayes bayesian-vor
ffbayes publish --year 2025 --phase both
```

---

## 🆘 **Troubleshooting**

### **Common Issues**

#### **"No players found in database"**
- **Problem**: Player names don't match database format
- **Solution**: Use full names (e.g., "Patrick Mahomes" not "P. Mahomes")
- **Smart Resolution**: The pipeline includes enhanced name resolution for:
  - Initials (e.g., "P. Mahomes" → "Patrick Mahomes")
  - Suffixes (e.g., "Michael Pittman Jr." → "Michael Pittman")
  - Position-aware fuzzy matching

#### **"Pipeline failed with errors"**
- **Problem**: Critical step failed
- **Solution**: Check logs in `~/ProjectsRuntime/ffbayes/logs/` for detailed error information
- **Check**: Verify your team file format matches requirements and that `ffbayes validate-team` passes before post-draft analysis

#### **"ffbayes split requires a phase argument"**
- **Problem**: The explicit split runner needs a phase name
- **Solution**: Run `ffbayes split pre_draft` or `ffbayes split post_draft`

#### **"Missing required columns"**
- **Problem**: Team file has wrong column names
- **Solution**: Use `Name`, `Position`, `Team` columns (or `POS`, `PLAYER`, `BYE` for legacy format)
- **Default path**: `~/ProjectsRuntime/ffbayes/data/raw/my_ff_teams/drafted_team_<year>.tsv`

### **Team File Format**
Your team file should have these columns:
```tsv
Name	Position	Team
Patrick Mahomes	QB	KC
James Cook	RB	BUF
# ... your full team
```

**Legacy format also supported:**
```tsv
POS	PLAYER	BYE
QB	Patrick Mahomes	10
RB	James Cook	7
# ... your full team
```

---

## 📚 **Technical Documentation**

For detailed technical information:
- [Documentation Index](docs/README.md) - Navigation for the docs folder
- [Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE.md) - How the models work
- [Pre/Post Draft Examples](docs/PRE_POST_DRAFT_EXAMPLES.md) - Usage examples

---

## 🎉 **Success Stories**

> "FFBayes gave me a complete draft strategy in minutes. The Excel output was perfect for draft day." - *10-team league winner*

> "The uncertainty analysis helped me balance safe picks with high-upside players." - *12-team league finalist*

> "Finally, a fantasy tool that doesn't require a PhD to use!" - *8-team league player*

---

## 🤝 **Contributing**

We welcome contributions. Follow the repository guidance in [AGENTS.md](AGENTS.md) and keep changes aligned with the current CLI, path constants, and test coverage.

Before submitting changes, run `ruff check .`, `ruff format .`, and `conda run -n ffbayes pytest -q`.

---

## 📊 Visualizations

The visualization layer produces the draft board HTML dashboard, workbook-backed audit tables, and the supporting validation plots.

Current outputs:
- Draft board workbook and dashboard payload in the pre-draft runtime tree
- Team analysis charts in the post-draft runtime tree
- Published plots and preview images only after `ffbayes publish`

The dashboard is designed to answer four draft-day questions quickly:
- Who are the best players right now?
- Who is likely to survive to my next pick?
- What position run risk should I care about?
- Which strategy has been working best in the backtest?

## 📄 **License**

MIT License.

---

**FFBayes**: Where Monte Carlo meets Bayesian intelligence for fantasy football dominance.
