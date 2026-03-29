# FFBayes: Advanced Fantasy Football Analytics Pipeline

A sophisticated fantasy football analytics system that combines Monte Carlo simulations, Bayesian uncertainty modeling, and draft-utility decision modeling to generate draft-board recommendations and post-draft team analysis.

## Workspace Contract

- Source code lives in `~/Projects/ffbayes`
- Runtime artifacts live in `~/ProjectsRuntime/ffbayes`
- Raw scraped inputs and published deliverables live in `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes`
- Successful runs mirror results and plots from runtime into the cloud tree

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

### **Step 3: Run Pre-Draft Analysis**
```bash
# Generate complete draft strategy
python -m ffbayes.run_pipeline_split pre_draft
```

**What You Get:**
- 📊 **Draft Board Workbook**: `draft_board_<year>.xlsx` with board, tiers, availability, scenarios, and diagnostics
- 🧠 **Dashboard Payload**: `dashboard_payload_<year>.json` for the local interactive draft dashboard
- 📋 **Decision Backtest**: `draft_decision_backtest_<year_range>.json` comparing market, VOR, consensus, and draft-score strategies

### **Step 4: Use During Draft**
- Open `~/ProjectsRuntime/ffbayes/results/<year>/pre_draft/vor_strategy/DRAFTING STRATEGY -- snake-draft_ppr-<ppr>_vor_top-<top_rank>_<year>.xlsx`
- Follow the pick-by-pick recommendations
- Use backup options if primary targets are gone

### **Step 5: Post-Draft Analysis**
After drafting, save your team to `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/data/raw/my_ff_teams/drafted_team_<year>.tsv`:

```tsv
Name	Position	Team
Patrick Mahomes	QB	KC
James Cook	RB	BUF
# ... your full team
```

Then run:
```bash
python -m ffbayes.run_pipeline_split post_draft
```

**What You Get:**
- 🎯 **Team Analysis**: Player contributions and reliability
- 📊 **Season Projections**: Weekly score expectations with confidence intervals
- 🔍 **Monte Carlo Validation**: 5000 simulations of your team's performance
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
Runtime working tree: `~/ProjectsRuntime/ffbayes/results/<year>/pre_draft/`

- `vor_strategy/` - Excel draft guide and raw VOR data
- `draft_strategy/` - Draft board workbook, dashboard payload, HTML dashboard, and compatibility JSON
- `hybrid_mc_bayesian/` - Monte Carlo + Bayesian model outputs

Published mirror: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/results/<year>/pre_draft/`

### **Post-Draft Outputs** (Use During Season)
Runtime working tree: `~/ProjectsRuntime/ffbayes/results/<year>/post_draft/`

- `team_aggregation/` - Team analysis JSON
- `montecarlo_results/` - Season projections TSV
- `monte_carlo_validation/` - Validation JSON

Published mirror: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/results/<year>/post_draft/`

### **Visualizations** (Analysis & Strategy)
Runtime plots: `~/ProjectsRuntime/ffbayes/plots/<year>/`

- `pre_draft/` - Strategy comparison, analysis, and draft decision support
- `post_draft/` - Team analysis charts

Published plots: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/plots/<year>/`
Published preview images: `~/Library/CloudStorage/OneDrive-Personal/SideProjects/ffbayes/docs/images/`

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

### **Pipeline Configuration: `config/pipeline_config.json`**
Defines the 12-step pipeline process:
1. **Data Collection** - Gather NFL fantasy data
2. **Data Validation** - Verify data quality
3. **Data Preprocessing** - Prepare for analysis
4. **VOR Strategy** - Generate traditional rankings
5. **Unified Dataset** - Combine all data sources
6. **Hybrid MC Analysis** - Run Monte Carlo + Bayesian model
7. **Draft Decision Strategy** - Generate the draft utility table and board
8. **Strategy Comparison** - Compare market, VOR, consensus, and draft-score strategies
9. **Pre-Draft Analysis** - Generate draft board and dashboard outputs
10. **Team Aggregation** - Analyze drafted team
11. **Monte Carlo Validation** - Validate team performance
12. **Team Summary Export** - Save analysis to files

---

## 📊 **Visualizations**

🚧 **UNDER CONSTRUCTION** 🚧

The visualization system is currently being updated to provide more comprehensive and actionable insights. New visualizations will include:

### **Pre-Draft Visualizations** (Draft-Day Ready)
- **Draft Board Dashboard** - Interactive local board with filters, targets, and regret view
- **Model Performance Dashboard** - Market vs VOR vs consensus vs draft-score diagnostics
- **Draft Value Heatmap** - Positional value by draft round
- **Risk-Reward Analysis** - Player uncertainty and upside potential
- **Strategy Success Rates** - Historical performance of different approaches

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

### **Post-Draft Visualizations** (Coming Soon)
- **Team Composition Analysis** - Roster balance and depth assessment
- **Performance Projections** - Weekly scoring expectations with confidence intervals
- **Monte Carlo Validation** - Team performance distribution across simulations
- **Trade Analysis Tools** - Player value comparison and trade evaluation

---

## 🎯 **Usage Examples**

### **Basic 10-Team League (Position 10)**
```bash
# Edit config/user_config.json first, then:
python -m ffbayes.run_pipeline_split pre_draft
```

### **12-Team League (Position 5, Full PPR)**
```bash
# Update config/user_config.json:
# "league_size": 12, "draft_position": 5, "ppr_value": 1.0
python -m ffbayes.run_pipeline_split pre_draft
```

### **Post-Draft Analysis**
```bash
# After saving your team to the cloud raw data tree:
python -m ffbayes.run_pipeline_split post_draft
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
- **Check**: Verify your team file format matches requirements

#### **"Missing required columns"**
- **Problem**: Team file has wrong column names
- **Solution**: Use `Name`, `Position`, `Team` columns (or `POS`, `PLAYER`, `BYE` for legacy format)

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
- [Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE.md) - How the models work
- [Pre/Post Draft Examples](docs/PRE_POST_DRAFT_EXAMPLES.md) - Usage examples
- [API Reference](docs/API_REFERENCE.md) - Complete function documentation

---

## 🎉 **Success Stories**

> "FFBayes gave me a complete draft strategy in minutes. The Excel output was perfect for draft day." - *10-team league winner*

> "The uncertainty analysis helped me balance safe picks with high-upside players." - *12-team league finalist*

> "Finally, a fantasy tool that doesn't require a PhD to use!" - *8-team league player*

---

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code standards
- Testing guidelines
- Pull request process

---

## 📊 Visualizations

The visualization layer now produces the draft board HTML dashboard, workbook-backed audit tables, and the supporting validation plots.

Current outputs:
- `draft_board_<year>.html` - interactive local draft board
- `draft_board_<year>.xlsx` - workbook for draft-day review
- `dashboard_payload_<year>.json` - machine-readable dashboard data
- `draft_decision_backtest_<year_range>.json` - strategy comparison summary

The dashboard is designed to answer four draft-day questions quickly:
- Who are the best players right now?
- Who is likely to survive to my next pick?
- What position run risk should I care about?
- Which strategy has been working best in the backtest?

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

**FFBayes**: Where Monte Carlo meets Bayesian intelligence for fantasy football dominance.
