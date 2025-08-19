## Phase 0: Already Completed

- [x] Historical weekly data ingestion with schedules and injuries; seasonal CSV exports and combined dataset generation (`get_ff_data.py`).
- [x] Monte Carlo simulator for team projections across selected seasons; configurable iterations; TSV outputs (`montecarlo-historical-ff.py`).
- [x] PyMC3 Bayesian hierarchical model estimating opponent-position effects and generating diagnostics and plots (`bayesian-hierarchical-ff.py`).
- [x] Draft strategy generator using VOR from FantasyPros projections; CSV and Excel outputs (`snake_draft_VOR.py`).

## Phase 1: Current Development (Immediate Priorities)

- [ ] **Fix PyMC3 Compatibility**: Resolve Theano/numpy compatibility issues preventing Bayesian model execution
- [ ] **Optimize Monte Carlo**: Fix recursion errors and improve player lookup efficiency
- [ ] **Improve VOR Implementation**: Research and implement more robust alternatives to basic VOR
- [ ] **Add Weekly Matchup Optimization**: Bayesian predictions for weekly lineup decisions

## Phase 2: Enhanced Analytics

- [ ] **Advanced Draft Strategy**: Implement more sophisticated draft algorithms beyond VOR
- [ ] **Uncertainty Quantification**: Better confidence intervals and risk assessment for decisions
- [ ] **Bench Player Optimization**: Limited bench management for small leagues
- [ ] **Injury/Practice Status Integration**: Incorporate injury reports into predictions

## Phase 3: Production Features

- [ ] **CLI Interface**: Command-line tools for easy weekly decision making
- [ ] **Configuration Management**: YAML/JSON configs for league settings and preferences
- [ ] **Automated Updates**: Daily/weekly data refresh and projection updates
- [ ] **Performance Tracking**: Track prediction accuracy and model performance over time

## Phase 4: Advanced Features

- [ ] **Multi-League Support**: Handle different league formats and scoring systems
- [ ] **Trade Analysis**: Bayesian evaluation of potential trades
- [ ] **Waiver Wire Optimization**: Data-driven waiver wire decisions
- [ ] **Playoff Strategy**: Specialized modeling for playoff scenarios


