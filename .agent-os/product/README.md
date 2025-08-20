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
  - `scripts/analysis/bayesian-hierarchical-ff-modern.py`: PyMC4-based Bayesian modeling for player predictions
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
- **Data Pipeline**: ✅ Complete - Data collection, validation, and preprocessing working with `nfl_data_py`
- **Monte Carlo Simulation**: ✅ Complete - Functional with 70,000+ simulations, no recursion issues
- **Bayesian Modeling**: ✅ Complete - PyMC4 working perfectly with smart trace management and reuse
- **VOR Implementation**: ✅ Functional - Draft strategy generation working with recent improvements
- **Pipeline Orchestration**: ✅ Complete - Enhanced master pipeline with comprehensive error handling and progress monitoring

### Script Organization
```
scripts/
├── data_pipeline/          # Data collection and preprocessing
│   ├── 01_collect_data.py          # Primary data collection (✅ Dynamic 10-year collection)
│   ├── 02_validate_data.py         # Data quality validation (✅ Enhanced error handling)
│   ├── 03_preprocess_analysis_data.py # NEW: Data preprocessing for analysis (✅ Separated concerns)
│   └── snake_draft_VOR.py          # Draft strategy generation
├── analysis/               # Statistical modeling and analysis
│   ├── montecarlo-historical-ff.py      # Monte Carlo simulations (✅ Working perfectly)
│   └── bayesian-hierarchical-ff-modern.py # PyMC4 Bayesian model (✅ Smart trace management)
├── utils/                  # Utility functions and helpers
│   └── progress_monitor.py      # Progress monitoring utilities (✅ Integrated across pipeline)
├── run_pipeline.py         # Master pipeline orchestrator (✅ Enhanced error handling)
└── run_with_conda.sh       # Conda environment helper

tests/                      # Testing framework
└── test_bayesian_quick.py  # Quick test with fast parameters (✅ Configurable testing)
```

### Pipeline Flow
1. **Data Collection**: `01_collect_data.py` imports weekly NFL data using `nfl_data_py`
2. **Data Validation**: `02_validate_data.py` checks data quality and completeness
3. **Data Preprocessing**: `03_preprocess_analysis_data.py` prepares analysis-ready datasets
4. **Monte Carlo Analysis**: `montecarlo-historical-ff.py` generates team projections
5. **Bayesian Modeling**: `bayesian-hierarchical-ff-modern.py` creates individual player predictions with PyMC4
6. **Team Aggregation**: `bayesian-team-aggregation.py` combines individual predictions into team totals
7. **Parallel Draft Strategy**: `parallel_draft_executor.py` runs both VOR and Bayesian strategies simultaneously
8. **Comparison Analysis**: `comparison_framework.py` provides side-by-side analysis of both approaches
9. **Pipeline Orchestration**: `run_pipeline.py` coordinates all stages with proper sequencing and parallel execution

### Next Steps
Review `roadmap.md` and `tech-stack.md`. The immediate focus should be:
1. **✅ Script Consolidation**: COMPLETE - All legacy scripts successfully consolidated
2. **✅ Compatibility Resolution**: COMPLETE - PyMC4 working perfectly with smart trace management
3. **✅ Pipeline Enhancement**: COMPLETE - Comprehensive error handling and progress monitoring
4. **Parallel Execution**: Implement simultaneous execution of both draft strategies
5. **Testing and Validation**: Ensure all components work together seamlessly

**Current Status**: Phase 1 & 2 COMPLETE - Ready for Phase 3: Testing and Validation

### Environment Management
- **Primary Environment**: PyMC4 environment for modern Bayesian modeling
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

### Major Architectural Improvements (2025-08-19) - ✅ COMPLETE
- **Separation of Concerns**: Data preprocessing separated from analysis scripts
- **Configurable Analysis**: Function parameters for cores, draws, tune, chains (no hardcoded config)
- **Smart Trace Management**: Automatic saving and reuse of expensive PyMC4 sampling results
- **Performance Optimization**: 4.5 minutes → 10 seconds for subsequent Bayesian analysis runs
- **Enhanced Error Handling**: Comprehensive error reporting and graceful degradation across all scripts
- **Progress Monitoring**: Integrated `ProgressMonitor` with fallback to basic progress bars
- **Proper Testing Framework**: Tests organized in `tests/` folder with configurable parameters
- **Pipeline Robustness**: Enhanced orchestration with better error handling and debugging