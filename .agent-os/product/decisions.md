## Architectural Decisions (Initial)

1. **Data Source**: `nfl_data_py` for historical stats, schedules, and injuries - chosen for reliability and comprehensive coverage
2. **Modeling Approach**: PyMC3 with Theano for Bayesian hierarchical modeling - provides uncertainty quantification crucial for decision making
3. **Simulation Method**: Monte Carlo using empirical resampling from historical weeks - simple but effective for team projections
4. **Draft Strategy**: FantasyPros scraping with VOR computation - industry standard but room for improvement
5. **Output Strategy**: File-based artifacts in versioned folders - simple and reproducible

## Key Design Principles

- **Data-Driven Over Expertise**: Prioritize statistical evidence over football knowledge
- **Uncertainty Quantification**: Always provide confidence intervals and risk assessment
- **Reproducibility**: All decisions should be traceable and repeatable
- **Simplicity**: Focus on actionable insights rather than complex models

## Current Challenges & Solutions

### PyMC3 Compatibility Issues
- **Problem**: Theano/numpy version conflicts preventing Bayesian model execution
- **Solution**: Either fix compatibility or migrate to modern PyMC v4/PyTensor
- **Decision**: Prioritize getting Bayesian modeling working over modernization

### VOR Limitations
- **Problem**: Basic VOR may not capture all relevant factors for optimal drafting
- **Solution**: Research alternatives like Expected Value, Risk-Adjusted VOR, or machine learning approaches
- **Decision**: Keep current VOR as baseline, implement improvements incrementally

### Weekly Optimization Scope
- **Problem**: Limited bench players reduce need for complex lineup optimization
- **Solution**: Focus on start/sit decisions and waiver wire optimization
- **Decision**: Prioritize draft strategy over weekly lineup management initially

## Open Questions

1. **PyMC3 vs Modern Alternatives**: Should we fix PyMC3 compatibility or migrate to PyMC v4?
2. **VOR Alternatives**: What are the most promising alternatives to basic VOR for draft strategy?
3. **Bench Management**: How much optimization is needed for limited bench scenarios?
4. **Performance Metrics**: What metrics should we track to validate model performance?

## Team Preferences

- **Learning Focus**: Willing to learn Python best practices during development
- **Statistical Rigor**: Prefer evidence-based approaches over heuristics
- **Practical Application**: Focus on actionable insights for weekly decision making
- **Documentation**: Clear documentation important for reproducibility and learning


