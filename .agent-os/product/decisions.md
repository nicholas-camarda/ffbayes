## Product Decisions

### Script Organization and Structure

**Decision**: Organize all scripts into logical directories with clear dependencies and a master orchestrator.

**Rationale**: The existing scripts were scattered across the root directory, making it difficult to understand dependencies and maintain the codebase. The new structure provides clear separation of concerns and makes the pipeline flow obvious.

**Implementation**: 
- `scripts/data_pipeline/`: Data collection, validation, and preprocessing
- `scripts/analysis/`: Statistical modeling and analysis components
- `scripts/utils/`: Utility functions and helper scripts
- `run_pipeline.py`: Master pipeline orchestrator

**Status**: ✅ **COMPLETED** - All scripts organized and basic structure in place

### Script Consolidation Strategy

**Decision**: Merge legacy scripts into the new organized structure while preserving all existing functionality.

**Rationale**: The user has sophisticated implementations in their legacy scripts that should be enhanced, not replaced. Consolidation will create a unified pipeline while maintaining the sophisticated data processing logic they've developed.

**Implementation**: 
- Analyze `get_ff_data.py` functionality and merge into `01_collect_data.py`
- Fix recursion issues in `montecarlo-historical-ff.py` and integrate into analysis pipeline
- Resolve PyMC3 compatibility issues or complete PyMC4 migration
- Standardize interfaces and error handling across all components

**Status**: 🔄 **IN PROGRESS** - Analysis phase beginning

### PyMC3 vs PyMC4 Migration

**Decision**: Evaluate both PyMC3 and PyMC4 versions to determine the best path forward.

**Rationale**: PyMC3 has compatibility issues with newer numpy versions, but the existing model logic is valuable. PyMC4 offers modern features and better compatibility but requires migration effort.

**Implementation**: 
- Test PyMC3 compatibility fixes with different package versions
- Evaluate `bayesian-hierarchical-ff-modern.py` PyMC4 implementation
- Choose primary implementation based on performance, stability, and maintainability
- Maintain fallback option if needed

**Status**: 🔄 **IN PROGRESS** - Both versions available, evaluation needed

### Progress Monitoring Standardization

**Decision**: Use `alive_progress` consistently across all pipeline stages for better user experience.

**Rationale**: Progress monitoring provides clear feedback on long-running operations and helps users understand pipeline status. Consistent implementation across all scripts improves user experience.

**Implementation**: 
- Created `scripts/utils/progress_monitor.py` for centralized progress monitoring
- Integrated progress bars into data collection and validation scripts
- Plan to add progress monitoring to all analysis scripts

**Status**: ✅ **COMPLETED** - Progress monitoring utility created and integrated

### Pipeline Orchestration

**Decision**: Create a master pipeline script that coordinates all stages with proper sequencing and error handling.

**Rationale**: Running individual scripts manually is error-prone and doesn't provide a unified user experience. A master orchestrator ensures proper execution order and handles failures gracefully.

**Implementation**: 
- Created `scripts/run_pipeline.py` with basic orchestration structure
- Plan to add proper stage sequencing, dependency management, and error recovery
- Add configuration options for pipeline customization

**Status**: 🔄 **IN PROGRESS** - Basic structure created, enhancement needed

### Environment Management

**Decision**: Use conda environments with helper scripts for easy execution and environment management.

**Rationale**: The user had conda activation issues that were resolved. Using conda environments provides better dependency isolation and the helper scripts make execution easier.

**Implementation**: 
- Created `scripts/run_with_conda.sh` for easy script execution
- Created `Makefile` with common pipeline operations
- Documented environment setup and management procedures

**Status**: ✅ **COMPLETED** - Environment management working with helper scripts

### Data Pipeline Flow

**Decision**: Establish a clear pipeline flow from data collection through analysis to strategy generation.

**Rationale**: The existing scripts work independently but don't have a clear execution order. A defined pipeline flow ensures data consistency and proper analysis sequencing.

**Implementation**: 
1. Data Collection: `01_collect_data.py` → `02_validate_data.py`
2. Analysis: `montecarlo-historical-ff.py` → `bayesian-hierarchical-ff.py` (or modern version)
3. Strategy Generation: `snake_draft_VOR.py` (can run independently)
4. Pipeline Orchestration: `run_pipeline.py` coordinates all stages

**Status**: 🔄 **IN PROGRESS** - Flow defined, implementation in progress

### Error Handling and Robustness

**Decision**: Implement comprehensive error handling and graceful degradation across all pipeline stages.

**Rationale**: The existing scripts have limited error handling, which can cause pipeline failures. Robust error handling ensures the pipeline continues execution even when individual stages fail.

**Implementation**: 
- Add error handling to all scripts
- Implement graceful degradation for failed stages
- Add comprehensive logging and debugging
- Create recovery mechanisms for common failures

**Status**: 🔄 **IN PROGRESS** - Basic error handling in place, comprehensive implementation needed

### Testing Strategy

**Decision**: Use incremental testing approach with quick validation scripts to ensure rapid iteration.

**Rationale**: The user emphasized "small tests so that these iterations don't take a million years." Incremental testing allows for quick validation of changes without running the full pipeline.

**Implementation**: 
- Created quick testing scripts for data validation, collection, and Bayesian modeling
- Plan to implement comprehensive testing framework
- Add unit tests, integration tests, and end-to-end pipeline tests

**Status**: 🔄 **IN PROGRESS** - Quick testing scripts created, comprehensive testing framework needed

## Current Status Summary

- ✅ **Script Organization**: Complete - All scripts organized into logical directories
- ✅ **Environment Management**: Complete - Conda working with helper scripts
- ✅ **Progress Monitoring**: Complete - Centralized progress monitoring utility
- 🔄 **Script Consolidation**: In Progress - Legacy script analysis and merging
- 🔄 **PyMC Compatibility**: In Progress - Both versions available, evaluation needed
- 🔄 **Pipeline Orchestration**: In Progress - Basic structure created, enhancement needed
- 🔄 **Error Handling**: In Progress - Basic implementation, comprehensive work needed
- 🔄 **Testing Framework**: In Progress - Quick tests created, comprehensive framework needed

## Bayesian Model Enhancement Decisions

### Problem Statement

**Decision**: Focus on substantial improvement of Bayesian model performance over baseline.

**Rationale**: The current model only marginally outperforms a simple 7-game average (3.69 vs 3.71 MAE), despite its sophisticated hierarchical structure. Research indicates that additional data sources and advanced techniques could provide substantial improvements.

**Status**: 🎯 **CURRENT FOCUS** - Active enhancement effort

### Research-Based Data Source Prioritization

**Decision**: Prioritize data sources based on expected impact and implementation difficulty.

**Rationale**: Research indicates varying impact levels for different data sources. Focusing on high-impact, manageable implementations first maximizes ROI.

**Data Source Impact Assessment**:
| Data Source | Expected Impact | Implementation Difficulty | Priority |
|-------------|----------------|-------------------------|----------|
| Weather Data | 5-10% improvement | Medium | High |
| Vegas Odds | 3-8% improvement | Medium | High |
| Snap Counts | 2-6% improvement | Low | High |
| Advanced Stats | 2-4% improvement | Medium | Medium |
| Injury Data | 2-5% improvement | Low | Medium |
| Time Series | 2-5% improvement | High | Medium |

**Status**: 📊 **RESEARCH COMPLETE** - Implementation prioritized

### Model Architecture Decisions

**Decision**: Use enhanced mean calculation with multiple data source effects.

**Rationale**: Bayesian hierarchical structure allows for principled integration of multiple data sources while maintaining uncertainty quantification.

**Prior Specifications**:
- **Intercept**: Normal(0, 2.0) - Overall adjustment
- **Average multiplier**: Normal(1.0, 0.1) - Baseline correction
- **Snap effect**: Normal(0, 1.0) - Playing time impact
- **Injury penalty**: Normal(-2.0, 1.0) - Availability impact
- **Weather effects**: Normal(0, 0.5) - Environmental impact
- **Vegas effects**: Normal(0, 0.5) - Game context impact

**Status**: 🏗️ **DESIGN COMPLETE** - Implementation in progress

### Success Criteria Definition

**Decision**: Set clear, measurable success criteria for model enhancement.

**Rationale**: Clear success metrics ensure focused development and objective evaluation of improvements.

**Success Criteria**:
- **Minimum**: 5% MAE improvement over baseline (MAE ≤ 3.53)
- **Target**: 10% MAE improvement over baseline (MAE ≤ 3.34)
- **Stretch**: 15% MAE improvement over baseline (MAE ≤ 3.15)

**Status**: 🎯 **DEFINED** - Metrics established for evaluation

## Next Priority Actions

1. **Complete legacy script analysis** to understand what needs to be preserved
2. **Begin script consolidation** starting with data collection scripts
3. **Resolve PyMC compatibility** by evaluating both versions
4. **Enhance pipeline orchestration** with proper sequencing and error handling
5. **Implement comprehensive testing** framework for all components
6. **Fix data loading issues** in unified Bayesian model
7. **Implement weather and Vegas odds APIs** for model enhancement


