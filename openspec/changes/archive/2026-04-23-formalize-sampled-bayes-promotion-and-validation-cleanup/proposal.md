## Why

The draft engine now has a coherent production model stack, but two important pieces are still unresolved. First, repeated non-fatal warnings in validation and advanced-stat pipelines create noisy runs and blur the line between genuine model problems and API/reporting drift. Second, the sampled hierarchical estimator exists as an evaluation lane, but there is still no decision-grade comparison process against the empirical-Bayes production baseline, so promotion remains intentionally unresolved rather than evidence-based.

This change is needed now because the remaining issues are no longer implementation sprawl problems; they are model-governance and validation-semantics problems. If they are left ambiguous, the repo will continue to emit warning-heavy runs, report undefined rank metrics as if they were estimated, and carry a sampled-Bayes lane that is technically present but not yet operationally decisive.

## What Changes

- Replace warning-producing `groupby.apply(...)` patterns in `src/ffbayes/analysis/advanced_stats_calculator.py` with explicit aggregation paths that do not rely on deprecated pandas grouping semantics.
- Redefine undefined rank-correlation reporting in `src/ffbayes/analysis/bayesian_vor_comparison.py` and downstream validation artifacts so constant-input slices are reported as not estimable rather than coerced to `0.0`.
- Tighten validation-report semantics so accuracy, calibration, and ranking outputs distinguish:
  - estimated zero association
  - unavailable or undefined association
  - omitted slices due to insufficient variation or support
- Formalize a repeatable sampled-versus-empirical comparison workflow around the existing evaluation lane in `src/ffbayes/analysis/sampled_player_model.py` as a diagnostics and explicit-tooling surface rather than a new supported production CLI path.
- Normalize backend alias drift in `src/ffbayes/data_pipeline/nflverse_backend.py` so current-player context accepts the installed `nflreadpy` player-team column surface (`latest_team` -> canonical `recent_team`) instead of warning and dropping current-team enrichment.
- Remove eager package-level imports from `src/ffbayes/draft_strategy/__init__.py` so pre-draft orchestration does not preload executable strategy modules and trigger `runpy` module-import warnings.
- Require the sampled hierarchical evaluation lane to persist decision-grade comparison artifacts that summarize:
  - searched configurations
  - convergence diagnostics
  - calibration
  - forecast accuracy
  - ranking quality
  - runtime cost
  - rookie versus veteran behavior
  - promotion decision and rationale
- Require a final production-mode interpretation report that runs the supported workflow on the full available data surface, summarizes the observed results, states the statistical and operational interpretation of those results, and makes an explicit decision about how the repo should move forward, including an explicit incomplete-evaluation outcome when the bounded sampled search did not finish.
- Keep empirical Bayes as the production estimator unless the sampled hierarchical estimator materially wins on the declared comparison contract.
- Explicitly support the possibility that sampled Bayes improves rookie or sparse-history handling without winning overall, and require that case to be reported as a distinct non-promotion outcome rather than flattened into pass/fail language.
- Update dashboard-facing evidence semantics and operator docs so undefined validation metrics render as `n/a` or equivalent, not as fabricated numeric neutrality.
- Add regression coverage so warning cleanup and validation-semantic changes cannot silently regress, and so the sampled-Bayes lane cannot be promoted or exposed without passing the formal comparison gate.
- **BREAKING** Stop treating undefined rank-correlation values as numeric `0.0` in supported validation outputs, diagnostics, and dashboard evidence surfaces.

## Capabilities

### New Capabilities
- `sampled-bayes-promotion-governance`: defines the formal comparison workflow, persisted search artifacts, decision criteria, and allowed promotion outcomes for the sampled hierarchical estimator relative to the empirical-Bayes production baseline.
- `validation-metric-semantics`: defines how accuracy, calibration, and rank-based validation metrics must represent undefined or non-estimable slices, including dashboard/evidence rendering expectations and regression-test requirements.

### Modified Capabilities
- `decision-evidence-panel`: validation and evidence requirements change so undefined correlation metrics are rendered explicitly as not estimable, while sampled-versus-empirical comparison outcomes remain out of the live dashboard until formal promotion occurs.
- `documentation-guide-suite`: README and `docs/` requirements change to document the warning cleanup, the formal sampled-comparison workflow, the production-default rule for empirical Bayes, and the meaning of non-estimable validation metrics.

## Impact

- Affected code includes:
  - `src/ffbayes/analysis/advanced_stats_calculator.py`
  - `src/ffbayes/analysis/bayesian_vor_comparison.py`
  - `src/ffbayes/analysis/sampled_player_model.py`
  - `src/ffbayes/data_pipeline/nflverse_backend.py`
  - `src/ffbayes/draft_strategy/__init__.py`
  - dashboard/evidence consumers under `src/ffbayes/draft_strategy/`
  - validation and governance tests under `tests/`
- Affected operator-facing surfaces include validation artifacts, dashboard evidence summaries, staged `site/` outputs when evidence content changes, and documentation that describes model-comparison outcomes.
- Affected review-facing surfaces include one final interpretation report written from full-data production-mode execution and used to decide whether sampled Bayes is promoted, retained only as follow-up input, or not carried forward.
- Sampled-Bayes comparison remains a diagnostics and explicit-tooling workflow, not a new default operator CLI surface, and the sampled estimator must not become a supported production path unless the formal promotion criteria are satisfied.
- This change affects runtime diagnostics and possibly year-scoped comparison outputs under the canonical `seasons/<year>/...` tree, but it should not reintroduce parallel artifact families or a second production dashboard path.
