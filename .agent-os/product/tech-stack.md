# Technical Stack & Implementation Guide

## Overview
This document provides comprehensive technical documentation for the fantasy football analytics pipeline, including current implementation, enhancement plans, and research findings.

## Current Status
- **Baseline MAE**: 3.71 (7-game average)
- **Current Bayesian MAE**: 3.69 (0.4% improvement)
- **Target**: 5-15% improvement (MAE ≤ 3.15-3.53)

## TESTING PROTOCOL - MANDATORY ENFORCEMENT

### **CRITICAL: ALWAYS USE TEST MODE DURING TESTING**

**ENFORCEMENT RULE**: During ANY testing, development, or validation work, you MUST use QUICK_TEST mode to prevent:
- ❌ **Production mode execution** (slow, resource-intensive)
- ❌ **Full dataset processing** (unnecessary during testing)
- ❌ **Resource waste** (CPU, memory, time)
- ❌ **Unreliable MAE metrics** (test mode results are not production metrics)

### **Required Test Mode Usage**

**1. Environment Variable Setup (MANDATORY)**
```bash
# ALWAYS set this before testing
export QUICK_TEST=true

# OR use inline for single commands
QUICK_TEST=true python -m ffbayes.analysis.bayesian_hierarchical_ff_unified
```

**2. Test Mode Validation (REQUIRED CHECK)**
```bash
# Verify test mode is active
echo $QUICK_TEST
# Should output: true

# Check for test mode indicators in output
QUICK_TEST=true python -m ffbayes.analysis.bayesian_hierarchical_ff_unified | grep "QUICK TEST"
# Should show: "QUICK TEST MODE ENABLED for Unified Bayesian model"
```

**3. Test Mode Parameters (AUTOMATIC)**
When `QUICK_TEST=true`:
- **Cores**: 1 (vs 7 production)
- **Draws**: 20 (vs 1000 production)
- **Tune**: 20 (vs 1000 production)
- **Chains**: 1 (vs 4 production)
- **Execution Time**: ~7 seconds (vs hours production)

**4. Test Mode Warnings (AUTOMATIC)**
The system automatically displays:
```
WARNING: QUICK_TEST mode detected — MAE and improvement metrics are not reliable and should not be trusted for evaluation.
```

### **Testing Workflow (MANDATORY)**

**Step 1: Set Test Mode**
```bash
export QUICK_TEST=true
```

**Step 2: Verify Test Mode**
```bash
echo $QUICK_TEST
# Must show: true
```

**Step 3: Execute Tests**
```bash
# Fast test execution
python -m ffbayes.analysis.bayesian_hierarchical_ff_unified
```

**Step 4: Validate Test Mode**
- Check output for "QUICK TEST MODE ENABLED"
- Verify execution time is ~7 seconds (not hours)
- Confirm MAE warning is displayed

### **Production Mode (ONLY for Final Validation)**

**Production mode should ONLY be used for:**
- ✅ Final model validation after testing is complete
- ✅ Production pipeline execution
- ✅ Performance benchmarking
- ✅ User-facing results

**NEVER use production mode for:**
- ❌ Development testing
- ❌ Code validation
- ❌ Feature testing
- ❌ Debugging

### **Violation Consequences**

**If you run production mode during testing:**
- ⚠️ **Wasted Resources**: Hours of CPU time, memory usage
- ⚠️ **Unreliable Results**: MAE metrics from test runs are meaningless
- ⚠️ **Development Delays**: Waiting for unnecessary computations

## PROJECT ORGANIZATION & FILE STRUCTURE

### **Organized Output Structure**

The pipeline automatically organizes all outputs into dedicated subfolders for easy navigation and management. **Files are named by draft year instead of timestamps** to prevent clutter and ensure easy access to the most recent results:

**📊 Plots Directory (`plots/`)**
```
plots/
├── team_aggregation/           # Team aggregation visualizations
│   ├── team_score_distribution_2025.png    # Current year (updates with each run)
│   ├── uncertainty_analysis_latest.png
│   └── team_score_breakdown_latest.png
├── monte_carlo/               # Monte Carlo simulation visualizations  
├── draft_strategy_comparison/  # Draft strategy comparison charts
│   └── draft_strategy_comparison_2025.png  # Current year (updates with each run)
├── bayesian_model/            # Bayesian model visualizations
└── test_runs/                 # Test run outputs and debugging
```

**📁 Results Directory (`results/`)**
```
results/
├── team_aggregation/          # Team aggregation results and analysis
│   └── team_aggregation_results_2025.json  # Current year (updates with each run)
├── montecarlo_results/        # Monte Carlo simulation outputs
│   └── 2025_projections_from_years*.tsv    # Current year (updates with each run)
├── bayesian-hierarchical-results/  # Bayesian model results and traces
│   ├── unified_model_results.json
│   └── unified_trace_2025.pkl     # Current year (updates with each run)
├── draft_strategy/            # Draft strategy outputs and configurations
│   ├── draft_strategy_pos*_2025.json       # Current year (updates with each run)
│   └── team_for_monte_carlo_2025.tsv       # Current year (updates with each run)
├── draft_strategy_comparison/ # Draft strategy comparison reports
└── model_comparison/          # Model comparison and evaluation results
```

### **Key Benefits**
- **🎯 Clear Organization**: Each output type has its dedicated subfolder
- **🔍 Easy Navigation**: Users quickly find specific types of results
- **📈 Scalability**: New output types automatically go to appropriate subfolders
- **🧹 Clean Structure**: No mixed file types in root directories
- **📅 Draft Year Naming**: Files named by year instead of timestamps (prevents clutter)
- **🔄 File Updates**: Each run updates the same year file instead of creating new ones

### **Automatic Maintenance**
- **Pipeline Execution**: Creates all required organized subfolders automatically
- **Script Outputs**: All scripts save to correct organized subfolders
- **File Organization**: Existing files automatically moved to appropriate locations

### **Usage Quick Reference**
- **Team Analysis**: `plots/team_aggregation/` + `results/team_aggregation/`
- **Monte Carlo Results**: `results/montecarlo_results/`
- **Bayesian Model**: `results/bayesian-hierarchical-results/`
- **Draft Strategy**: `results/draft_strategy/` + `results/draft_strategy_comparison/`

## Tech Stack

### Languages
- Python 3.8+ (for PyMC4 compatibility)

### Core Python Libraries
- **Data Processing**: pandas (1.4+), numpy (1.21+), scipy (1.7+)
- **Visualization**: matplotlib (3.5+)
- **Bayesian Modeling**: PyMC3 (3.11.x), PyMC4 (5.x), Theano-PyMC, ArviZ
- **NFL Data**: `nfl_data_py`
- **Web Scraping**: requests, BeautifulSoup4
- **Progress Monitoring**: alive-progress
- **Data Validation**: pyarrow (for efficient CSV handling)

### Script Organization
```
scripts/
├── data_pipeline/          # Data collection and preprocessing
│   ├── 01_collect_data.py      # Primary data collection
│   ├── 02_validate_data.py     # Data validation
│   ├── get_ff_data.py          # Legacy script (to be consolidated)
│   ├── get_ff_data_improved.py # Enhanced data collection
│   └── snake_draft_VOR.py      # Draft strategy generation
├── analysis/               # Statistical modeling and analysis
│   ├── montecarlo-historical-ff.py      # Monte Carlo simulations
│   ├── bayesian-hierarchical-ff.py      # PyMC3 Bayesian model
│   └── bayesian-hierarchical-ff-modern.py # PyMC4 Bayesian model
├── utils/                  # Utility functions and helpers
│   ├── progress_monitor.py      # Progress monitoring utilities
│   └── quick_*.py              # Testing scripts (to be evaluated)
├── run_pipeline.py         # Master pipeline orchestrator
└── run_with_conda.sh       # Conda environment helper
```

## Bayesian Model Implementation

### Current Model: `bayesian_hierarchical_ff_modern.py`
**Status**: Modern PyMC4 implementation (primary)
**Purpose**: Bayesian hierarchical model for individual player predictions

### Core Architecture

#### 1. Data Preprocessing
- **Position Encoding**: One-hot encoding for QB, WR, RB, TE
- **Team Encoding**: Integer encoding for teams and opponents
- **Home/Away Indicator**: Binary indicator for home/away games
- **Rolling Averages**: 7-game rolling average of fantasy points
- **Ranking System**: Quartile-based ranking (1-4) based on rolling average

#### 2. Model Components

**Observables**:
```python
# Degrees of freedom for Student's t distributions
nu = pm.Exponential('nu_minus_one', 1 / 29.0, shape=2) + 1

# Standard deviations based on rank
err = pm.Uniform('std_dev_rank', 0, 100, shape=ranks)
err_b = pm.Uniform('std_dev_rank_b', 0, 100, shape=ranks)
```

**Defensive Effects**:
```python
# Global defensive priors for each position
opp_def = pm.Normal('opp_team_prior', 0, 100**2, shape=num_positions)

# Team-specific defensive effects by position
opp_qb = pm.Normal('defensive_differential_qb', opp_def[0], 100**2, shape=team_number)
# ... similar for WR, RB, TE
```

**Home/Away Effects**:
```python
# Home/away advantages by position and rank
home_adv = pm.Normal('home_additive_prior', 0, 100**2, shape=num_positions)
away_adv = pm.Normal('away_additive_prior', 0, 100**2, shape=num_positions)
```