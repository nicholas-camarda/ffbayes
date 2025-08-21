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
- ⚠️ **Resource Conflicts**: Blocking other development work

**ENFORCEMENT**: Always verify `QUICK_TEST=true` before any testing execution.

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

**Likelihood Models**:
1. **Difference from Average Model**: Predicts deviation from rolling average
2. **Total Score Prediction Model**: Predicts absolute fantasy points

### Current Performance
- **MAE**: 3.69 (0.4% improvement over baseline)
- **Strengths**: Sophisticated hierarchical structure, uncertainty quantification
- **Weaknesses**: Only marginal improvement over simple baseline

## Enhanced Model Features

### Current Features (Implemented)
- 7-game average baseline
- Opponent defensive effects
- Home/away advantages
- Position-specific effects
- Snap counts (offensive percentage)
- Injury status (Out/Doubtful)
- Practice status (Limited/DNP)

### Planned Features
- Weather data (temperature, wind, precipitation)
- Vegas odds (spreads, over/under)
- Advanced stats (target share, red zone usage)
- Time series features (momentum, trends)

## Data Sources and APIs

### Weather Data
- **API**: OpenWeatherMap Historical Weather API
- **Features**: Temperature, wind speed, precipitation, humidity
- **Integration**: Merge by game date and stadium location

### Vegas Odds
- **API**: ESPN API or SportsData.io
- **Features**: Point spreads, over/under, implied totals
- **Integration**: Merge by game ID and date

### Advanced Stats
- **API**: ESPN API, Pro Football Reference
- **Features**: Target share, snap counts, red zone usage, air yards
- **Integration**: Merge by player ID, season, week

### Enhanced Model Architecture
```python
# Enhanced mean calculation
mu = (
    intercept + 
    (avg_multiplier * player_avg) + 
    defensive_effects + 
    home_away_effects +
    snap_effect * offense_pct +
    injury_penalty * injury_status +
    weather_effects +
    vegas_effects +
    time_series_effects
)
```



## Pipeline Architecture

### Pipeline Dependencies
- **Data Collection**: `nfl_data_py` → Weekly player stats, schedules, injuries
- **Data Validation**: `pyarrow`, `pandas` → Quality checks and completeness validation
- **Monte Carlo**: `numpy`, `pandas` → Team projection simulations
- **Bayesian Modeling**: `pymc`, `arviz` → Player prediction models
- **Draft Strategy**: `requests`, `beautifulsoup4` → FantasyPros scraping and VOR calculations

### Pipeline Orchestration
- **Master Script**: `run_pipeline.py` orchestrates complete workflow
- **Progress Monitoring**: `alive-progress` with comprehensive logging
- **Error Handling**: Graceful degradation with detailed error reporting
- **Step Execution**: Sequential execution with dependency management

### Data Collection Implementation
- **Primary Source**: `nfl_data_py` for NFL statistics and schedule data
- **Secondary Sources**: FantasyPros for player rankings and projections
- **Data Types**: Player stats, schedules, injuries, snap counts
- **Quality Validation**: Completeness checks, statistical validation, outlier detection

### Monte Carlo Simulation
- **Purpose**: Team-level projections using historical data
- **Method**: Random sampling from historical player performance
- **Features**: Weighted year distribution, recursive retry mechanism
- **Output**: Team score projections with uncertainty quantification

## Environment Management
- **Primary**: `ffbayes` conda environment with PyMC4 and modern packages
- **Fallback**: PyMC3 environment for legacy compatibility
- **Helper Scripts**: `run_with_conda.sh`, `Makefile` for easy execution
- **Package Management**: `conda` with `pip` for specific package versions

## Output Artifacts
- **Datasets**: `datasets/*.csv`, `combined_datasets/*.csv`
- **Results**: `results/montecarlo_results/*.tsv`
- **Plots**: `plots/*.png`
- **Draft Strategy**: `snake_draft_datasets/*.csv`, `*.xlsx`
- **Pipeline Logs**: Comprehensive logging and progress monitoring

## Development Tools
- **Linting/Formatting**: Ruff with `pyproject.toml` configuration
- **Version Control**: Git with organized `.gitignore`
- **Documentation**: Agent OS product documentation and technical specs
- **Testing**: Incremental testing approach with quick validation scripts




