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

### Pipeline Dependencies
- **Data Collection**: `nfl_data_py` → Weekly player stats, schedules, injuries
- **Data Validation**: `pyarrow`, `pandas` → Quality checks and completeness validation
- **Monte Carlo**: `numpy`, `pandas` → Team projection simulations
- **Bayesian Modeling**: `pymc`, `arviz` → Player prediction models
- **Draft Strategy**: `requests`, `beautifulsoup4` → FantasyPros scraping and VOR calculations

### Environment Management
- **Primary**: `ffbayes` conda environment with PyMC4 and modern packages
- **Fallback**: PyMC3 environment for legacy compatibility
- **Helper Scripts**: `run_with_conda.sh`, `Makefile` for easy execution
- **Package Management**: `conda` with `pip` for specific package versions

### Output Artifacts
- **Datasets**: `datasets/*.csv`, `combined_datasets/*.csv`
- **Results**: `results/montecarlo_results/*.tsv`
- **Plots**: `plots/*.png`
- **Draft Strategy**: `snake_draft_datasets/*.csv`, `*.xlsx`
- **Pipeline Logs**: Comprehensive logging and progress monitoring

### Development Tools
- **Linting/Formatting**: Ruff with `pyproject.toml` configuration
- **Version Control**: Git with organized `.gitignore`
- **Documentation**: Agent OS product documentation and technical specs
- **Testing**: Incremental testing approach with quick validation scripts


