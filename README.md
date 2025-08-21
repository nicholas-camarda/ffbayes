# FFBayes: Fantasy Football Analytics Pipeline

A complete fantasy football analytics pipeline that generates tiered draft strategies using Bayesian modeling and Monte Carlo simulations.

## 🚀 Quick Start: Generate Your Draft Strategy

Want to get your tiered fantasy football draft list right now? Run this:

```bash
# Install the package
pip install -e .

# Preferred: one command to generate a complete draft strategy (Phase A)
ffbayes-pipeline --phase draft
```

This will give you a complete tiered draft strategy with:
- **Primary targets** for each pick
- **Backup options** if your targets are gone
- **Fallback options** for late-round steals
- **Position scarcity analysis**
- **Uncertainty quantification**

## 📊 Complete Pipeline: From Data to Draft Strategy

### Preferred: One-command runs via CLI
```bash
# Phase A: Draft-only
ffbayes-pipeline --phase draft

# Phase B: Post-draft validation (requires your team TSV)
ffbayes-pipeline --phase validate --team-file my_ff_teams/my_actual_2025.tsv

# Full pipeline (uses enhanced orchestrator when available)
ffbayes-pipeline --phase full
```

## 🧪 Post-Draft Validation (Phase B)
After your real draft, validate your team via Monte Carlo:
```bash
ffbayes-pipeline --phase validate --team-file my_ff_teams/my_actual_2025.tsv
```

## 🎯 How the Models Work

### Bayesian Hierarchical Model (Step 1: Propose)
- **Purpose:** Player-level predictions with uncertainty
- **Method:** PyMC4 hierarchical model accounting for team effects, position, and player history
- **Input:** Player performance, team matchups, position data
- **Output:** Individual player projections with confidence intervals
- **Visualization:** `plots/bayesian_model/` - Model diagnostics and predictions

### Draft Strategy Generation (Step 2: Optimize)
- **Purpose:** Tiered draft recommendations
- **Method:** Uses Bayesian predictions to create optimal team construction
- **Input:** Bayesian player projections + position scarcity + uncertainty analysis
- **Output:** Multiple options per pick with reasoning and uncertainty analysis

### Monte Carlo Validation (Step 3: Evaluate - Adversarial!)
- **Purpose:** Adversarial validation of draft strategies
- **Method:** Simulates thousands of outcomes for drafted teams to test strategy performance
- **Input:** Drafted team compositions from strategy
- **Output:** Team performance validation and strategy effectiveness
- **Visualization:** `plots/monte_carlo/` - Team performance distributions

## 📈 Available Visualizations

### Example Outputs (from your latest run)

- Team Score Breakdown
  ![Team Score Breakdown](plots/team_aggregation/team_score_breakdown_latest.png)

- Position Analysis
  ![Position Analysis](plots/team_aggregation/position_analysis_latest.png)

- Uncertainty Analysis
  ![Uncertainty Analysis](plots/team_aggregation/uncertainty_analysis_latest.png)

- Comparison Insights
  ![Comparison Insights](plots/team_aggregation/comparison_insights_latest.png)

## 📋 Example Draft Strategy Output

```json
{
  "strategy": {
    "Pick 3": {
      "primary_targets": ["Lamar Jackson", "Ja'Marr Chase", "Josh Allen"],
      "backup_options": ["Saquon Barkley", "Joe Burrow", "Jahmyr Gibbs"],
      "fallback_options": ["Rashee Rice", "Jayden Daniels", "Baker Mayfield"],
      "position_priority": "QB > TE > RB",
      "reasoning": "TE depth available, early pick - target elite players",
      "uncertainty_analysis": {
        "risk_tolerance": "medium",
        "primary_avg_uncertainty": 0.101,
        "overall_uncertainty": 0.096
      }
    }
  }
}
```

## 🔧 Configuration Options

### Draft Strategy Parameters
```bash
ffbayes-draft-strategy \
  --draft-position 3 \        # Your draft position (1 - [LEAGUE_SIZE])
  --league-size 12 \          # League size (8, 10, 12, 14, 16)
  --risk-tolerance low \      # Risk level (low, medium, high)
  --output-file strategy.json # Save to file
```

### Pipeline Options
```bash
# Quick test mode (faster, less data)
ffbayes-collect --quick-test
ffbayes-mc --quick-test

# Force refresh (reprocess existing data)
ffbayes-collect --force-refresh

# Custom data directory
ffbayes-collect --data-dir /path/to/data
```

## 📁 File Structure

```
ffbayes/
├── datasets/
│   ├── season_datasets/          # Raw NFL data by year
│   └── combined_datasets/        # Processed 5-year datasets
├── results/
│   ├── montecarlo_results/       # Team simulations
│   ├── bayesian-hierarchical-results/  # Player predictions
│   ├── team_aggregation/         # Combined analysis
│   └── draft_strategy/           # Draft strategy outputs
├── plots/
│   ├── bayesian_model/           # Model diagnostics
│   ├── team_aggregation/         # Team analysis
│   └── monte_carlo/              # Simulation results
├── config/
│   └── pipeline_config.json      # Pipeline configuration
└── my_ff_teams/                  # Your team configurations
```

## 🧪 Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Test individual components:
```bash
python -m pytest tests/test_draft_strategy.py -v
python -m pytest tests/test_monte_carlo.py -v
```

## 🐛 Troubleshooting

### Common Issues

**"No combined datasets found"**
```bash
# Regenerate the combined dataset
ffbayes-preprocess
```

**"Monte Carlo results not found"**
```bash
# Run Monte Carlo simulation
ffbayes-mc --team-file my_ff_teams/my_actual_2025.tsv
```

**"Bayesian results not found"**
```bash
# Run Bayesian modeling
ffbayes-bayes
```

### Data Issues
- Check `datasets/season_datasets/` for raw data files
- Verify internet connection for data collection
- Ensure sufficient disk space for results

## 📊 Performance

- **Data Collection:** ~5 minutes for 5 years of data
- **Monte Carlo:** ~10-15 minutes for 70,000 simulations
- **Bayesian Modeling:** ~20-30 minutes for full model
- **Draft Strategy:** ~30 seconds for complete tiered list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📝 Appendix: Advanced/Manual Commands
If you prefer granular control, you can run individual stages:

- Draft strategy only (assumes data/model already prepared):
```bash
ffbayes-draft-strategy --draft-position 3 --league-size 12 --risk-tolerance medium
```

- Individual stages manual flow:
```bash
ffbayes-collect
ffbayes-validate
ffbayes-preprocess
ffbayes-bayes
ffbayes-draft-strategy --draft-position 3 --league-size 12 --risk-tolerance medium
ffbayes-mc --team-file my_ff_teams/my_actual_2025.tsv
```
