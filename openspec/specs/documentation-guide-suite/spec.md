## Purpose
Defines the documentation guide contract for operator workflow, technical method explanation, and lay interpretation.

## Requirements

### Requirement: Operator documentation SHALL explain the current dashboard workflow end to end
The guide suite MUST include a dashboard operator guide that explains how to use the canonical runtime dashboard, the repo-local shortcut, and the staged Pages copy, including the steps an operator takes before, during, and after a draft.

The operator guide MUST describe non-estimable validation metrics as unavailable states rather than neutral numeric values, and it MUST explain that sampled-versus-empirical comparison is an evaluation workflow unless a formal promotion outcome exists.

#### Scenario: Operator guide explains surface authority
- **WHEN** the operator guide describes dashboard files
- **THEN** it MUST identify the canonical runtime HTML/payload pair as authoritative and MUST describe repo `dashboard/` and repo `site/` as derived surfaces with different purposes

#### Scenario: Operator guide explains draft-day actions
- **WHEN** the operator guide describes dashboard usage
- **THEN** it MUST explain how to review recommendations, interpret the inspector, update draft state, use the evidence and provenance sections, and handle finalized draft downloads

#### Scenario: Operator guide explains post-draft flow
- **WHEN** the operator guide covers after-draft actions
- **THEN** it MUST document finalized draft ingest, retrospective generation, and the distinction between runtime-local evaluation artifacts and Pages staging

#### Scenario: Operator guide explains shipped war-room visuals when present
- **WHEN** the shipped dashboard includes timing, scarcity, or comparative decision visuals
- **THEN** the operator guide MUST explain what each visual is for, how it reacts to current board state, and how it should be used alongside the existing recommendation lanes, inspector, and trust surfaces

#### Scenario: Operator guide explains unavailable validation metrics
- **WHEN** the operator guide describes evidence or validation tables
- **THEN** it MUST explain that `n/a` or `not estimable` means the metric could not be validly estimated for that slice and MUST NOT describe that state as equivalent to a measured zero relationship

#### Scenario: Operator guide explains sampled evaluation without overstating it
- **WHEN** the operator guide mentions sampled-versus-empirical comparison artifacts
- **THEN** it MUST explain that sampled diagnostics remain outside the live dashboard and operator workflow unless a formal promotion outcome exists, rather than implying that the existence of sampled diagnostics changes the live board by itself

### Requirement: The authoritative technical guide SHALL describe the implemented draft engine truthfully
The guide suite MUST use one authoritative technical or methods guide to describe the implemented draft board pipeline, prediction target, posterior construction, decision-policy layer, baselines, validation scope, interval semantics, and sampled-model governance in terms that match the current code and emitted artifacts.

#### Scenario: Technical guide distinguishes implemented vs conceptual content
- **WHEN** the authoritative technical guide discusses mathematical or modeling details
- **THEN** it MUST clearly distinguish implemented current-board behavior from conceptual intuition, optional analyses, and deprecated or compatibility-only surfaces

#### Scenario: Technical guide states targets and baselines explicitly
- **WHEN** the authoritative technical guide describes model outputs or validation
- **THEN** it MUST identify the forecast target, decision target, replacement baseline definitions, and internal-validation scope in explicit terms

#### Scenario: Technical guide explains interval semantics
- **WHEN** the authoritative technical guide describes uncertainty, floor/ceiling values, confidence intervals, or evidence summaries
- **THEN** it MUST specify what each interval or range represents and MUST NOT conflate predictive intervals, posterior intervals, percentile bands, or confidence intervals of the mean

#### Scenario: Technical guide explains unavailable rank metrics explicitly
- **WHEN** the authoritative technical guide describes validation artifacts that include rank-correlation or ranking-quality metrics
- **THEN** it MUST explain that unavailable values represent non-estimable slices, not numeric zero-association findings

#### Scenario: Technical guide explains sampled-governance outcomes truthfully
- **WHEN** the authoritative technical guide documents the sampled hierarchical estimator
- **THEN** it MUST identify the bounded search workflow, the persisted comparison artifact, the allowed decision outcomes, and the rule that empirical Bayes remains production unless the promotion artifact explicitly says otherwise

### Requirement: Layperson documentation SHALL teach interpretation without overclaiming certainty
The guide suite MUST include a non-technical guide that explains the workflow from data collection through final results at roughly a high-school-math level and teaches users how to interpret draft outputs without implying external validation or causal certainty.

#### Scenario: Lay guide distinguishes descriptive vs predictive claims
- **WHEN** the layperson guide explains outputs
- **THEN** it MUST distinguish descriptive summaries, predictive rankings, and decision-support heuristics and MUST explicitly avoid causal interpretation

#### Scenario: Lay guide explains what key metrics do and do not mean
- **WHEN** the layperson guide introduces board value, VOR proxy, next-pick survival, fragility, upside, decision evidence, or freshness
- **THEN** it MUST explain both how the metric is intended to help and what conclusions the user MUST NOT draw from it in plain language that does not require advanced statistical training

#### Scenario: Lay guide explains terms by components
- **WHEN** the layperson guide introduces a term or derived metric that could be unclear to a non-technical reader
- **THEN** it MUST break the term into its component ideas and explain what is happening rather than only naming the metric

#### Scenario: Lay guide treats unavailable evidence honestly
- **WHEN** the layperson guide describes validation or evidence values that appear as `n/a` or `not estimable`
- **THEN** it MUST explain that those values mean the repo could not estimate that comparison cleanly for the slice rather than implying the slice had neutral performance

#### Scenario: Lay guide does not overclaim sampled diagnostics
- **WHEN** the layperson guide mentions sampled-versus-empirical comparison results
- **THEN** it MUST explain that those diagnostics are evaluation work unless a formal promotion decision has changed the live estimator
