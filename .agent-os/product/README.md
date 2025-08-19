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
- **Data Pipeline**: Historical NFL data ingestion using `nfl_data_py` for weekly player stats, schedules, and injuries; merged and exported per-season to `datasets/` and consolidated in `combined_datasets/` (see `get_ff_data.py`).
- **Monte Carlo Simulation**: Team outcome projections from historical distributions with configurable years and iterations; outputs TSV to `results/montecarlo_results/` (see `montecarlo-historical-ff.py`).
- **Bayesian Hierarchical Modeling**: PyMC3/Theano implementation to estimate opponent-position effects and generate projections with uncertainty; plots saved to `plots/` (see `bayesian-hierarchical-ff.py`).
- **Draft Strategy Generator**: Scrapes FantasyPros projections, computes VOR, and exports ranked CSV and Excel strategy guide (see `snake_draft_VOR.py`).

### Current State
- Data processing pipeline is working and generating datasets
- Monte Carlo simulation runs but needs optimization for edge cases
- Bayesian model has PyMC3 compatibility issues that need resolution
- VOR-based draft strategy is functional but could be improved

### Next Steps
Review `roadmap.md` and `tech-stack.md`. The immediate focus should be:
1. Fix PyMC3 compatibility issues for Bayesian modeling
2. Improve VOR implementation with more robust methods
3. Add weekly matchup optimization features
4. Enhance uncertainty quantification for better decision making


