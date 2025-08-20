# Fantasy Football Bayesian Analysis Pipeline

A comprehensive data science pipeline for fantasy football analysis using modern Bayesian modeling and Monte Carlo simulation techniques. This project provides tools for data collection, validation, analysis, and prediction to help make data-driven fantasy football decisions.

## What This Is

This pipeline combines statistical modeling with fantasy football data to:

- **Collect and validate NFL player data** from multiple seasons
- **Apply Bayesian hierarchical modeling** to understand opponent effects and player performance
- **Run Monte Carlo simulations** for team projection and uncertainty quantification
- **Generate actionable insights** for draft strategy and player selection

## Key Features

### Data Pipeline
- **Automated data collection** from `nfl_data_py` library
- **Comprehensive data validation** with quality checks and outlier detection
- **Data preprocessing** for analysis-ready datasets
- **Robust error handling** and progress monitoring

### Statistical Analysis
- **Bayesian Hierarchical Model** (PyMC4) for opponent-position effects
- **Monte Carlo Simulation** for team projection with uncertainty
- **Model comparison framework** for validation and selection
- **Advanced draft strategy** based on statistical insights

### Technical Features
- **Comprehensive testing** with 142+ unit tests
- **Progress monitoring** for long-running operations
- **Flexible configuration** for different analysis scenarios
- **Reproducible results** with proper seed management
- **Standardized script interfaces** with consistent argument parsing
- **Console scripts** for easy command-line usage
- **Model validation** with convergence checking and quality assessment
- **Enhanced pipeline orchestration** with dependency management

## Installation

### Prerequisites
- Python 3.10+
- 4GB+ RAM (for Bayesian modeling)
- 1GB+ disk space (for data storage)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ffbayes
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ffbayes
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

That's it! The pipeline will automatically create all required directories when you first run it.

## Configuration Options

### Performance Optimization

#### Quick Test Mode (Recommended for Development)
For fast testing and development, use quick test mode:
```bash
QUICK_TEST=true ffbayes-pipeline
# Or for just Monte Carlo:
QUICK_TEST=true ffbayes-mc
```

Quick test mode automatically:
- Reduces simulations to 200 (vs 5,000)
- Uses only last 2 years of data (vs 5 years)
- Completes in ~3 minutes (vs ~20 minutes)

#### Multiprocessing (Default: Enabled)
The pipeline automatically uses all CPU cores for faster execution:
```bash
# Use all cores (default)
ffbayes-pipeline

# Disable multiprocessing (single-threaded)
USE_MULTIPROCESSING=false ffbayes-pipeline

# Limit to specific number of cores
MAX_CORES=4 ffbayes-pipeline
```

#### Full Configuration Example
```bash
# Fast testing with 4 cores
QUICK_TEST=true MAX_CORES=4 ffbayes-pipeline

# Production run with all cores
USE_MULTIPROCESSING=true ffbayes-pipeline
```

### Monte Carlo Simulation
Manual simulation configuration (if not using environment variables):

- **Default**: 5,000 simulations (high accuracy, ~20 minutes)
- **Quick Test**: 200 simulations (fast, ~3 minutes)
- **Custom**: Use command-line arguments:
  ```bash
  ffbayes-mc --simulations 1000
  ```

### Data Years
By default, the pipeline uses the last 5 years of data (2020-2024). You can specify custom years:
```bash
ffbayes-collect --years 2020,2021,2022,2023,2024
```

## 🚀 Quick Start

The pipeline automatically creates all necessary directories. Simply run:

```bash
# Activate the conda environment
conda activate ffbayes

# Run the complete pipeline
ffbayes-pipeline
```

## 📦 Environment Setup

This project uses a conda environment with PyMC4 for modern Bayesian modeling:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate ffbayes

# Install the package in development mode
pip install -e .

# Verify PyMC4 is installed
python -c "import pymc; print(f'PyMC version: {pymc.__version__}')"
```

### Key Dependencies
- **PyMC4**: Modern Bayesian modeling framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **nfl-data-py**: NFL data collection
- **matplotlib**: Plotting and visualization
- **pytest**: Testing framework
- **jupyter**: Interactive development

## Quick Start

### 1. Run the Complete Pipeline

The simplest way to get started:

```bash
ffbayes-pipeline
```

Or using the module directly:

```bash
python -m ffbayes.run_pipeline
```

This will:
- **Automatically create all required directories**
- Collect current NFL data
- Validate and preprocess it
- Run both Monte Carlo and Bayesian analysis
- Generate results and visualizations

### 2. Step-by-Step Usage

For more control, run individual pipeline stages:

#### Collect Data
```bash
ffbayes-collect
```

#### Validate Data
```bash
ffbayes-validate
```

#### Preprocess for Analysis
```bash
ffbayes-preprocess
```

#### Run Monte Carlo Simulation
```bash
ffbayes-mc
```

#### Run Bayesian Analysis
```bash
ffbayes-bayes
```

#### Run Team Aggregation
```bash
ffbayes-agg
```

#### Run Model Comparison
```bash
ffbayes-compare
```

#### Generate Visualizations
```bash
ffbayes-viz
```

#### Generate Advanced Draft Strategy
```bash
ffbayes-draft-strategy --draft-position 3 --league-size 12 --risk-tolerance medium
```

### 3. Console Scripts Overview

The package provides standardized console scripts for all major operations:

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `ffbayes-pipeline` | Run complete pipeline | `--quick-test`, `--verbose` |
| `ffbayes-collect` | Collect NFL data | `--years`, `--force-refresh` |
| `ffbayes-validate` | Validate data quality | `--data-dir`, `--output-dir` |
| `ffbayes-preprocess` | Preprocess data | `--data-dir`, `--quick-test` |
| `ffbayes-mc` | Monte Carlo simulation | `--simulations`, `--cores` |
| `ffbayes-bayes` | Bayesian analysis | `--draws`, `--tune`, `--chains` |
| `ffbayes-agg` | Team aggregation | `--data-dir`, `--output-dir` |
| `ffbayes-compare` | Model comparison | `--data-dir`, `--output-dir` |
| `ffbayes-viz` | Generate visualizations | `--data-dir`, `--output-dir` |
| `ffbayes-draft-strategy` | Advanced draft strategy | `--draft-position`, `--league-size`, `--risk-tolerance` |

All scripts support standardized arguments:
- `--help`: Show help information
- `--verbose`: Enable verbose logging
- `--quiet`: Suppress output
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--quick-test`: Enable quick test mode
- `--config`: Specify configuration file
- `--output-dir`: Specify output directory

## Using Your Own Team

### Create Team File

Create a team file in `my_ff_teams/my_team_YYYY.tsv`:

```tsv
Name	Position	Tm
Josh Allen	QB	BUF
Christian McCaffrey	RB	SF
Tyreek Hill	WR	MIA
Travis Kelce	TE	KC
```

### Supported Positions
- `QB` - Quarterback
- `RB` - Running Back  
- `WR` - Wide Receiver
- `TE` - Tight End
- `K` - Kicker
- `DST` - Defense/Special Teams

The pipeline will automatically find and use your most recent team file.

## Configuration Options

### Monte Carlo Simulation

Key parameters in `ffbayes-mc`:

```python
# Simulation settings
number_of_simulations = 5000  # Number of simulations to run
my_years = [2019, 2020, 2021, 2022, 2023]  # Years for historical sampling

# Run with custom settings
ffbayes-mc --draws 2000 --cores 8
```

### Bayesian Model

Key parameters in `ffbayes-bayes`:

```python
# MCMC settings
DEFAULT_DRAWS = 1000     # Posterior samples
DEFAULT_TUNE = 500       # Tuning samples  
DEFAULT_CHAINS = 4       # Parallel chains
DEFAULT_CORES = 4        # CPU cores to use

# Run with custom settings
ffbayes-bayes --draws 2000 --tune 1000 --chains 6 --cores 8
```

## Understanding the Results

### Monte Carlo Simulation Output

```
🎯 Starting Monte Carlo Simulation
   Team size: 4 players
   Simulations: 5,000
   
📈 Simulation Results:
   Team projection: 85.32 points
   Standard deviation: 12.45 points
   95% confidence interval: [62.1, 108.7]
   
🏆 Player Performance Summary:
   Josh Allen: 22.1 ± 8.2 points
   Christian McCaffrey: 24.8 ± 9.1 points
```

### Bayesian Model Output

```
Bayesian Model MAE: 8.23
Baseline (7-game avg) MAE: 9.87
Improvement: 16.6%
```

- **MAE (Mean Absolute Error)**: Average prediction error in fantasy points
- **Improvement**: How much better the model is than simple averaging
- **Defensive Effects**: Team-specific impact on opponent scoring

## File Structure

```
ffbayes/
├── src/ffbayes/                    # Main package source
│   ├── run_pipeline.py             # Main pipeline orchestrator
│   ├── data_pipeline/              # Data processing modules
│   │   ├── collect_data.py         # Data collection from nfl_data_py
│   │   ├── validate_data.py        # Data quality validation
│   │   └── preprocess_analysis_data.py  # Analysis preprocessing
│   ├── draft_strategy/             # Draft strategy modules
│   │   ├── bayesian_draft_strategy.py   # Advanced Bayesian draft strategy
│   │   └── snake_draft_VOR.py      # Traditional VOR-based draft strategy
│   ├── analysis/                   # Analysis modules
│   │   ├── montecarlo_historical_ff.py     # Monte Carlo simulation
│   │   ├── bayesian_hierarchical_ff_modern.py  # Bayesian modeling
│   │   ├── bayesian_team_aggregation.py    # Team aggregation
│   │   ├── model_comparison_framework.py   # Model comparison
│   │   └── create_team_aggregation_visualizations.py  # Visualizations
│   └── utils/                      # Utility modules
│       ├── interface_standards.py  # Standard interfaces
│       ├── progress_monitor.py     # Progress monitoring
│       ├── script_interface.py     # Script standardization
│       ├── model_validation.py     # Model validation
│       └── enhanced_pipeline_orchestrator.py  # Pipeline orchestration
├── tests/                          # Comprehensive test suite
├── config/                         # Configuration files
├── datasets/                       # Raw and processed data
├── results/                        # Analysis outputs
├── plots/                          # Generated visualizations
└── my_ff_teams/                    # Your team configurations
```

## Running Tests

Verify everything works correctly:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_utility_integration.py
pytest tests/test_script_standardization.py
pytest tests/test_standardized_interfaces.py
```

## Common Issues

### Memory Issues
If you encounter memory errors during Bayesian modeling:
- Reduce `draws` parameter (e.g., 500 instead of 1000)
- Reduce `chains` parameter (e.g., 2 instead of 4)
- Close other applications to free RAM

### Missing Players
If your team players aren't found in historical data:
- Check player name spelling matches NFL records
- Verify player was active in the analysis years
- Check the console output for "Missing team members"

### Data Collection Errors
If data collection fails:
- Check internet connection
- Verify `nfl_data_py` is properly installed
- Try running collection for individual years

## Contributing

This project uses:
- **Code formatting**: Ruff with 200-character line length
- **Testing**: pytest with comprehensive test coverage
- **Documentation**: Inline docstrings and type hints

Run tests before submitting changes:
```bash
pytest tests/
```

## Data Sources

- **NFL Data**: Provided by `nfl_data_py` library
- **Historical Range**: 2017-2023 (configurable)
- **Update Frequency**: Weekly during NFL season

## License

This project is for educational and personal use. NFL data is subject to NFL terms of service.
