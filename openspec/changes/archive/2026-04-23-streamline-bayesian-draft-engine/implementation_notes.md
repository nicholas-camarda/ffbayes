## Implementation Notes

This file records the concrete implementation inventory and the resolved model
governance decisions for the `streamline-bayesian-draft-engine` change.

### Active supported surfaces at implementation start

- Primary player forecast:
  - `src/ffbayes/analysis/bayesian_player_model.py`
  - `src/ffbayes/analysis/bayesian_vor_comparison.py`
- Draft decision and dashboard path:
  - `src/ffbayes/draft_strategy/draft_decision_system.py`
  - `src/ffbayes/draft_strategy/draft_decision_strategy.py`
  - `src/ffbayes/refresh_dashboard.py`
  - `src/ffbayes/publish_pages.py`
  - `src/ffbayes/publish_artifacts.py`
  - `src/ffbayes/utils/visualization_manager.py`
- Pipeline and path contract:
  - `src/ffbayes/run_pipeline_split.py`
  - `config/pipeline_pre_draft.json`
  - `config/pipeline_config.json`
  - `src/ffbayes/utils/path_constants.py`
- Unified dataset and ingestion path:
  - `src/ffbayes/data_pipeline/create_unified_dataset.py`
  - `src/ffbayes/data_pipeline/nflverse_backend.py`

### Legacy or retirement-candidate surfaces at implementation start

- Hybrid Monte Carlo plus random-forest path:
  - `src/ffbayes/analysis/hybrid_mc_bayesian.py`
  - `src/ffbayes/analysis/hybrid_evaluation.py`
  - `src/ffbayes/data_pipeline/hybrid_data_integration.py`
  - `src/ffbayes/draft_strategy/hybrid_draft_strategy.py`
  - `src/ffbayes/draft_strategy/hybrid_excel_generation.py`
  - `src/ffbayes/draft_strategy/risk_adjusted_rankings.py`
  - `tests/test_hybrid_mc_bayesian.py`
- Hybrid-dependent secondary consumers:
  - `src/ffbayes/draft_strategy/pick_by_pick_strategy.py`
  - `src/ffbayes/analysis/bayesian_team_aggregation.py`
  - `src/ffbayes/analysis/model_comparison_framework.py`
  - `src/ffbayes/utils/strategy_path_generator.py`
  - `src/ffbayes/utils/file_naming.py`

### Runtime path consumers at implementation start

- Runtime input paths still referencing `data/`:
  - `src/ffbayes/utils/path_constants.py`
  - `src/ffbayes/publish_artifacts.py`
  - `src/ffbayes/utils/visualization_manager.py`
  - `tests/test_collect_data_paths.py`
  - `tests/test_directory_management.py`
  - `tests/test_documentation_contracts.py`
- Season output paths still referencing `runs/<year>/pre_draft/artifacts/...`:
  - `src/ffbayes/utils/path_constants.py`
  - `config/pipeline_config.json`
  - `README.md`
  - `docs/DATA_LINEAGE_AND_PATHS.md`
  - `docs/OUTPUT_EXAMPLES.md`
  - `docs/TECHNICAL_DEEP_DIVE.md`
  - `docs/LAYPERSON_GUIDE.md`
  - `docs/METRIC_REFERENCE.md`
  - `tests/test_path_constants.py`
  - `tests/test_directory_management.py`
  - `tests/test_refresh_dashboard.py`

### Forecast-target and dashboard-contract drift identified at implementation start

- `src/ffbayes/analysis/bayesian_player_model.py` aggregates `FantPt` to a
  per-player-season mean (`fantasy_points`) and uses that as the direct target.
- `docs/TECHNICAL_DEEP_DIVE.md` and dashboard-facing docs describe the current
  production model more loosely as player-season fantasy points, which is less
  precise than the implementation.
- `src/ffbayes/draft_strategy/draft_decision_system.py` currently treats
  `proj_points_mean` as the primary player-value quantity and derives starter
  and replacement deltas from that field.

### Resolved design decisions to enforce during implementation

- The supported production dashboard target is season-total fantasy value.
- Season-total posterior summaries are composed through posterior predictive
  simulation from scoring-rate and availability components.
- Empirical Bayes remains the production default estimator unless the sampled
  hierarchical estimator materially improves the declared promotion criteria.
- The sampled hierarchical estimator is an evaluation lane on the same contract,
  not a co-primary runtime path.
- If the sampled hierarchical estimator does not win on the supported contract,
  it does not remain wired into supported CLI, pipeline, dashboard, or docs.
- The first production contextual structure beyond player and position is:
  - team-season effects
  - team-change indicators
  - depth-chart or roster-competition context
  - schedule-strength aggregate only if validated
- College-conference effects are out of scope for this NFL production change.
- The dashboard main board remains lean; rate-versus-availability detail belongs
  in the selected-player inspector and evidence-linked surfaces.
