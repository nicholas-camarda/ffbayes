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
- **Unified Data Pipeline**: Organized package structure with clear dependencies and master orchestration
  - `src/ffbayes/data_pipeline/collect_data.py`: Primary data collection using `nfl_data_py` for weekly player stats, schedules, and injuries
  - `src/ffbayes/data_pipeline/validate_data.py`: Comprehensive data quality checks and validation
  - `src/ffbayes/data_pipeline/preprocess_analysis_data.py`: Data preprocessing for analysis
- **Analysis Pipeline**: Statistical modeling and simulation components
  - `src/ffbayes/analysis/montecarlo_historical_ff.py`: Monte Carlo simulation for team outcome projections
  - `src/ffbayes/analysis/bayesian_hierarchical_ff_modern.py`: PyMC4-based Bayesian modeling for player predictions
  - `src/ffbayes/analysis/bayesian_team_aggregation.py`: Team projections from individual Bayesian predictions
  - `src/ffbayes/analysis/model_comparison_framework.py`: Model comparison and validation
  - `src/ffbayes/analysis/create_team_aggregation_visualizations.py`: Comprehensive visualizations
- **Draft Strategy Pipeline**: **Tier-based Bayesian approach for optimal team construction**
  - `src/ffbayes/draft_strategy/snake_draft_VOR.py`: Traditional VOR-based draft strategy (FantasyPros)
  - `src/ffbayes/draft_strategy/bayesian_draft_strategy.py`: Tier-based Bayesian draft strategy (✅ COMPLETE)
  - **Tier-based approach**: Multiple options per pick (10+ options) for practical draft use
  - **Team construction optimization**: Focus on best possible team given draft position
  - **Uncertainty-aware decisions**: Uses Bayesian predictions with confidence intervals
  - **Position scarcity management**: Accounts for position runs and scarcity
  - **Pre-generated strategy**: Run once before draft, not real-time optimization
- **Pipeline Orchestration**: `src/ffbayes/run_pipeline.py` coordinates all stages with parallel execution support
- **Utility Functions**: Comprehensive utility modules in `src/ffbayes/utils/`
  - `interface_standards.py`: Standard interfaces and environment handling
  - `progress_monitor.py`: Progress tracking across all components
  - `script_interface.py`: Standardized script interfaces
  - `model_validation.py`: Model validation and convergence checking
  - `enhanced_pipeline_orchestrator.py`: Advanced pipeline orchestration
- **Console Scripts**: Standardized command-line interface with 9 console scripts
  - `ffbayes-pipeline`, `ffbayes-collect`, `ffbayes-validate`, `ffbayes-preprocess`
  - `ffbayes-mc`, `ffbayes-bayes`, `ffbayes-agg`, `ffbayes-compare`, `ffbayes-viz`

### Current State
- **Package Structure**: ✅ Complete - Converted to `src/ffbayes` package structure with proper organization
- **Console Scripts**: ✅ Complete - 9 standardized console scripts for all major operations
- **Standardized Interfaces**: ✅ Complete - Consistent argument parsing, logging, and error handling across all scripts
- **Data Pipeline**: ✅ Complete - Data collection, validation, and preprocessing working with `nfl_data_py`
- **Monte Carlo Simulation**: ✅ Complete - Functional with 70,000+ simulations, no recursion issues
- **Bayesian Modeling**: ✅ Complete - PyMC4 working perfectly with smart trace management and reuse
- **Team Aggregation**: ✅ Complete - Individual predictions aggregated to team totals with uncertainty propagation
- **Model Comparison**: ✅ Complete - Framework for comparing Monte Carlo and Bayesian results
- **Visualization System**: ✅ Complete - Comprehensive visualization generation with test/production organization
- **VOR Implementation**: ✅ Functional - Draft strategy generation working with recent improvements
- **Bayesian Draft Strategy**: 🔄 Planned - Tier-based approach for optimal team construction
- **Pipeline Orchestration**: ✅ Complete - Enhanced master pipeline with comprehensive error handling and progress monitoring
- **Utility Integration**: ✅ Complete - All utility functionality properly integrated into main pipeline
- **Testing**: ✅ Complete - 142+ comprehensive tests covering all functionality

### Package Organization
```
src/ffbayes/
├── data_pipeline/          # Data collection and preprocessing
│   ├── collect_data.py              # Primary data collection (✅ Dynamic 10-year collection)
│   ├── validate_data.py             # Data quality validation (✅ Enhanced error handling)
│   └── preprocess_analysis_data.py  # Data preprocessing for analysis (✅ Separated concerns)
├── draft_strategy/         # Draft strategy modules
│   ├── snake_draft_VOR.py           # Traditional VOR-based draft strategy
│   └── bayesian_draft_strategy.py   # Tier-based Bayesian draft strategy (✅ COMPLETE)
├── analysis/               # Statistical modeling and analysis
│   ├── montecarlo_historical_ff.py      # Monte Carlo simulations (✅ Working perfectly)
│   ├── bayesian_hierarchical_ff_modern.py # PyMC4 Bayesian model (✅ Smart trace management)
│   ├── bayesian_team_aggregation.py     # Team aggregation (✅ Individual to team projections)
│   ├── model_comparison_framework.py    # Model comparison (✅ Monte Carlo vs Bayesian)
│   └── create_team_aggregation_visualizations.py # Visualizations (✅ Comprehensive plots)
├── utils/                  # Utility functions and helpers
│   ├── interface_standards.py      # Standard interfaces (✅ Environment handling)
│   ├── progress_monitor.py         # Progress monitoring (✅ Integrated across pipeline)
│   ├── script_interface.py         # Script standardization (✅ Consistent interfaces)
│   ├── model_validation.py         # Model validation (✅ Convergence checking)
│   └── enhanced_pipeline_orchestrator.py # Pipeline orchestration (✅ Advanced orchestration)
└── run_pipeline.py         # Master pipeline orchestrator (✅ Enhanced error handling)

tests/                      # Comprehensive testing framework
├── test_utility_integration.py     # Utility integration tests (✅ 12 tests)
├── test_script_standardization.py  # Script standardization tests (✅ 18 tests)
├── test_standardized_interfaces.py # Interface standardization tests (✅ 19 tests)
└── [other test files]              # Additional comprehensive tests (✅ 142 total)

config/                     # Configuration files
└── pipeline_config.json    # Pipeline configuration (✅ Enhanced orchestration)
```

### Pipeline Flow
1. **Data Collection**: `collect_data.py` imports weekly NFL data using `nfl_data_py`
2. **Data Validation**: `validate_data.py` checks data quality and completeness
3. **Data Preprocessing**: `preprocess_analysis_data.py` prepares analysis-ready datasets
4. **Monte Carlo Analysis**: `montecarlo_historical_ff.py` generates team projections
5. **Bayesian Modeling**: `bayesian_hierarchical_ff_modern.py` creates individual player predictions with PyMC4
6. **Team Aggregation**: `bayesian_team_aggregation.py` combines individual predictions into team totals
7. **Model Comparison**: `model_comparison_framework.py` compares Monte Carlo and Bayesian results
8. **Visualization**: `create_team_aggregation_visualizations.py` generates comprehensive plots
9. **Draft Strategy**: `bayesian_draft_strategy.py` generates tier-based draft strategy for optimal team construction with uncertainty-aware decision making
10. **Pipeline Orchestration**: `run_pipeline.py` coordinates all stages with proper sequencing and parallel execution

### Console Scripts Usage
- **Complete Pipeline**: `ffbayes-pipeline` - Run the entire pipeline from data collection to draft strategy
- **Individual Stages**: Use specific console scripts for targeted operations
  - `ffbayes-collect` - Data collection only
  - `ffbayes-validate` - Data validation only
  - `ffbayes-mc` - Monte Carlo simulation only
  - `ffbayes-bayes` - Bayesian analysis only
  - `ffbayes-agg` - Team aggregation only
  - `ffbayes-compare` - Model comparison only
  - `ffbayes-viz` - Visualization generation only
  - `ffbayes-draft-strategy` - Generate tier-based draft strategy for optimal team construction

### Next Steps
Review `roadmap.md` and `tech-stack.md`. The immediate focus should be:
1. **✅ Script Consolidation**: COMPLETE - All legacy scripts successfully consolidated
2. **✅ Package Migration**: COMPLETE - Converted to `src/ffbayes` package structure
3. **✅ Console Scripts**: COMPLETE - 9 standardized console scripts implemented
4. **✅ Standardized Interfaces**: COMPLETE - Consistent argument parsing and error handling
5. **✅ Utility Integration**: COMPLETE - All utility functionality properly integrated
6. **Advanced Draft Strategy**: Implement Bayesian-based draft strategy as alternative to VOR
7. **Parallel Execution**: Implement simultaneous execution of both draft strategies
8. **Testing and Validation**: Ensure all components work together seamlessly

**Current Status**: Phase 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15 COMPLETE - Advanced Draft Strategy Implemented

### Additional Documentation
- `plots-organization.md`: Documentation of visualization outputs and plots directory structure

### Environment Management
- **Primary Environment**: PyMC4 environment for modern Bayesian modeling
- **Working Directory**: All operations should be performed from the project root directory (`/Users/ncamarda/Library/CloudStorage/OneDrive-Personal/Desktop/coding/ffbayes`)

### Development Standards
- **Working Environment**: The current working directory should always be the project root (`ffbayes/`)
- **Package Execution**: Use console scripts (`ffbayes-*`) or module imports (`python -m ffbayes.*`)
- **Path References**: Use relative paths from the project root for all file operations
- **Environment Setup**: Always run `conda activate ffbayes` before executing any scripts
- **Package Installation**: Install in development mode with `pip install -e .`
- **Testing**: Run tests with `pytest` from the project root

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