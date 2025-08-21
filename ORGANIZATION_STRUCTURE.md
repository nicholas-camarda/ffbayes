# 📁 FFBayes Project Organization Structure

## 🎯 **Purpose**
This document defines the organized folder structure for plots and results to ensure all outputs are properly categorized and easy to find.

## 📊 **Plots Directory Structure**

```
plots/
├── team_aggregation/           # Team aggregation visualizations
│   ├── team_score_distribution_*.png
│   ├── uncertainty_analysis_latest.png
│   └── team_score_breakdown_latest.png
├── monte_carlo/               # Monte Carlo simulation visualizations
│   └── [TSV files moved here for organization]
├── draft_strategy_comparison/  # Draft strategy comparison charts
├── bayesian_model/            # Bayesian model visualizations
├── test_runs/                 # Test run outputs and debugging
└── [other organized subfolders]
```

## 📁 **Results Directory Structure**

```
results/
├── team_aggregation/          # Team aggregation results and analysis
│   └── team_aggregation_results_*.json
├── montecarlo_results/        # Monte Carlo simulation outputs
│   └── 2025_projections_from_years*.tsv
├── bayesian-hierarchical-results/  # Bayesian model results and traces
│   ├── unified_model_results.json
│   └── unified_trace_*.pkl
├── draft_strategy/            # Draft strategy outputs and configurations
│   ├── draft_strategy_pos*.json
│   └── team_for_monte_carlo_*.tsv
├── draft_strategy_comparison/ # Draft strategy comparison reports
│   └── draft_strategy_comparison_report_*.txt
├── model_comparison/          # Model comparison and evaluation results
└── [other organized subfolders]
```

## 🔧 **Implementation Details**

### **Fixed Issues:**
1. ✅ **Monte Carlo TSV files**: Now saved to `results/montecarlo_results/` instead of `plots/`
2. ✅ **Team aggregation plots**: Now saved to `plots/team_aggregation/` instead of generic `plots/`
3. ✅ **Draft strategy comparison reports**: Now saved to `results/draft_strategy_comparison/` instead of root `results/`
4. ✅ **Pipeline directory creation**: Now creates all organized subfolders automatically

### **Code Changes Made:**
- `src/ffbayes/analysis/montecarlo_historical_ff.py`: Fixed output directory to `results/montecarlo_results/`
- `src/ffbayes/analysis/bayesian_team_aggregation.py`: Fixed visualization output to `plots/team_aggregation/`
- `src/ffbayes/analysis/draft_strategy_comparison.py`: Fixed report output to `results/draft_strategy_comparison/`
- `src/ffbayes/run_pipeline.py`: Added creation of all organized subfolders

### **Benefits:**
1. **🎯 Clear Organization**: Each type of output has its dedicated subfolder
2. **🔍 Easy Navigation**: Users can quickly find specific types of results
3. **📈 Scalability**: New output types can be added to appropriate subfolders
4. **🧹 Clean Root Directories**: No more mixed file types in root folders
5. **📋 Consistent Structure**: All scripts follow the same organization pattern

## 🚀 **Usage**

### **For Developers:**
- All new outputs should go to the appropriate organized subfolder
- Use the established naming conventions for consistency
- Update this document when adding new output types

### **For Users:**
- **Team Analysis**: Check `plots/team_aggregation/` and `results/team_aggregation/`
- **Monte Carlo Results**: Check `results/montecarlo_results/`
- **Bayesian Model**: Check `results/bayesian-hierarchical-results/`
- **Draft Strategy**: Check `results/draft_strategy/` and `results/draft_strategy_comparison/`

## 📝 **Maintenance**

This structure is automatically maintained by:
1. **Pipeline execution**: Creates all required directories
2. **Script outputs**: Save to correct organized subfolders
3. **File organization**: Existing files moved to appropriate locations

Last updated: August 21, 2025
