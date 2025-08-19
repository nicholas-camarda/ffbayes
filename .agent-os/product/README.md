## ffbayes — Agent OS Product Overview

**Main Idea**: Data-driven fantasy football toolkit that bridges the gap between football experts and data-savvy novices. Built to help someone with strong statistical skills but no football knowledge compete against experienced fantasy players using Bayesian modeling and advanced analytics.

**Target Users**: 
- Primary: Data-savvy individuals (like medical students) who want to compete in fantasy football against experienced players
- Secondary: Anyone looking for more sophisticated fantasy football analytics beyond basic VOR

### Product Vision
The repository solves the problem of competing in fantasy football leagues where opponents have deep football knowledge and experience. It levels the playing field by:
- Using data science to optimize snake draft strategy (currently VOR-based)
- Applying Bayesian modeling for weekly matchup predictions with uncertainty quantification
- Providing reproducible, evidence-based decision making instead of relying on football expertise

### Key Features
- **Unified Data Pipeline**: Organized script structure with clear dependencies and master orchestration
  - `scripts/data_pipeline/01_collect_data.py`: Primary data collection using `nfl_data_py` for weekly player stats, schedules, and injuries
  - `scripts/data_pipeline/02_validate_data.py`: Comprehensive data quality checks and validation
- **Analysis Pipeline**: Statistical modeling and simulation components
  - `scripts/analysis/montecarlo-historical-ff.py`: Monte Carlo simulation for team outcome projections
  - `scripts/analysis/bayesian-hierarchical-ff.py`: PyMC3-based Bayesian modeling for player predictions
  - `scripts/analysis/bayesian-hierarchical-ff-modern.py`: PyMC4-based modern Bayesian modeling
  - `scripts/analysis/bayesian-team-aggregation.py`: Team projections from individual Bayesian predictions
- **Draft Strategy Pipeline**: **Parallel execution of both traditional and advanced approaches**
  - `scripts/draft_strategy/traditional_vor_draft.py`: Traditional VOR-based draft strategy (FantasyPros)
  - `scripts/draft_strategy/advanced_bayesian_draft.py`: Advanced Bayesian-based draft strategy
  - `scripts/draft_strategy/parallel_draft_executor.py`: Execute both strategies simultaneously
  - `scripts/draft_strategy/comparison_framework.py`: Real-time comparison and analysis
- **Pipeline Orchestration**: `scripts/run_pipeline.py` coordinates all stages with parallel execution support
- **Utility Functions**: `scripts/utils/progress_monitor.py` provides consistent progress tracking across all components

### Current State
- **Script Organization**: ✅ Complete - All scripts organized into logical directories with clear dependencies
- **Data Pipeline**: ✅ Functional - Data collection and validation working with `nfl_data_py`
- **Monte Carlo Simulation**: ⚠️ Needs Fixes - Functional but has recursion issues that need resolution
- **Bayesian Modeling**: ⚠️ Compatibility Issues - PyMC3 has numpy/Theano conflicts, PyMC4 version available
- **VOR Implementation**: ✅ Functional - Draft strategy generation working with recent improvements
- **Pipeline Orchestration**: ✅ Basic Structure - Master pipeline script exists but needs enhancement

### Script Organization
```
scripts/
├── data_pipeline/          # Data collection and preprocessing
│   ├── 01_collect_data.py      # Primary data collection (✅ Dynamic 10-year collection)
│   ├── 02_validate_data.py     # Data quality validation (✅ Updated for current pipeline)
│   └── snake_draft_VOR.py      # Draft strategy generation
├── analysis/               # Statistical modeling and analysis
│   ├── montecarlo-historical-ff.py      # Monte Carlo simulations
│   └── bayesian-hierarchical-ff-modern.py # PyMC4 Bayesian model
├── utils/                  # Utility functions and helpers
│   └── progress_monitor.py      # Progress monitoring utilities
├── run_pipeline.py         # Master pipeline orchestrator
└── run_with_conda.sh       # Conda environment helper
```

### Pipeline Flow
1. **Data Collection**: `01_collect_data.py` imports weekly NFL data using `nfl_data_py`
2. **Data Validation**: `02_validate_data.py` checks data quality and completeness
3. **Monte Carlo Analysis**: `montecarlo-historical-ff.py` generates team projections
4. **Bayesian Modeling**: `bayesian-hierarchical-ff.py` (or modern version) creates individual player predictions
5. **Team Aggregation**: `bayesian-team-aggregation.py` combines individual predictions into team totals
6. **Parallel Draft Strategy**: `parallel_draft_executor.py` runs both VOR and Bayesian strategies simultaneously
7. **Comparison Analysis**: `comparison_framework.py` provides side-by-side analysis of both approaches
8. **Pipeline Orchestration**: `run_pipeline.py` coordinates all stages with proper sequencing and parallel execution

### Next Steps
Review `roadmap.md` and `tech-stack.md`. The immediate focus should be:
1. **Script Consolidation**: Merge legacy scripts into the new organized structure
2. **Compatibility Resolution**: Fix PyMC3 issues or complete PyMC4 migration
3. **Pipeline Enhancement**: Improve error handling, progress monitoring, and orchestration
4. **Parallel Execution**: Implement simultaneous execution of both draft strategies
5. **Testing and Validation**: Ensure all components work together seamlessly

### Environment Management
- **Primary Environment**: `ffbayes` conda environment with PyMC4 and modern packages
- **Fallback Environment**: PyMC3 environment if compatibility issues persist
- **Helper Scripts**: `run_with_conda.sh` and `Makefile` for easy environment management
- **Working Directory**: All operations should be performed from the project root directory (`/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/Desktop/coding/ffbayes`)

### Development Standards
- **Working Environment**: The current working directory should always be the project root (`ffbayes/`)
- **Script Execution**: All scripts should be run from the project root, not from subdirectories
- **Path References**: Use relative paths from the project root for all file operations
- **Environment Setup**: Always run `conda init && conda activate ffbayes` before executing any scripts

### Critical Package Compatibility Issues (2025-01-19) - ✅ RESOLVED
- **nfl_data_py Version**: Current version still uses `season_type` column (not `game_type` as initially suspected)
- **Data Structure**: Weekly data and schedules maintain consistent column names across versions
- **Root Cause Identified**: The "Series is ambiguous" error was caused by improper pandas row iteration using `iterrows()`
- **Solution Implemented**: Replace `iterrows()` with `itertuples()` for proper scalar value extraction
- **Status**: ✅ RESOLVED - Script now processes 11,250+ rows successfully across 2023-2024 seasons
- **Performance**: Data collection completed in 3.0 seconds with no errors
- **Output**: Successfully generated season datasets and combined dataset with proper opponent/home-away indicators
- **Key Insight**: This was a pandas best practices issue, not a package compatibility issue