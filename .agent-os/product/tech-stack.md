## Tech Stack

### Languages
- Python 3.x
- R (optional; exploratory scripts in `old code/main.R`)

### Core Python Libraries
- Data: pandas (1.4.x), numpy (1.22.x), scipy (1.7.x)
- Visualization: matplotlib (3.5.x)
- Bayesian Modeling: PyMC3, Theano-PyMC, ArviZ
- NFL Data: `nfl_data_py`
- Web: requests, BeautifulSoup
- CLI UX: alive-progress

### Artifacts and Outputs
- Datasets: `datasets/*.csv`, `combined_datasets/*.csv`
- Results: `results/montecarlo_results/*.tsv`
- Plots: `plots/*.png`
- Draft strategy: `snake_draft_datasets/*.csv`, `*.xlsx`

### Environment
- See `pymc3_env_requirements*.txt` for explicit pinned versions. Model scripts expect a conda env with Theano-compatible versions.


