## ADDED Requirements

### Requirement: Repo SHALL publish an authoritative guide suite for the supported pre-draft workflow
The repository MUST publish a durable guide suite under `docs/` that covers the supported `ffbayes pre-draft` workflow for distinct audiences instead of relying only on README snippets or generated dashboard copy.

#### Scenario: Guide suite is indexed by audience and purpose
- **WHEN** a user opens the repo documentation index
- **THEN** the docs MUST link to operator, layperson, statistician, metric-reference, and path/data-lineage guides as distinct documentation surfaces

#### Scenario: Guide suite stays scoped to the supported workflow
- **WHEN** the guide suite describes the primary workflow
- **THEN** it MUST describe the supported `pre_draft` flow and MUST distinguish optional or compatibility-only commands from the default operator path

### Requirement: Operator documentation SHALL explain the current dashboard workflow end to end
The guide suite MUST include a dashboard operator guide that explains how to use the canonical runtime dashboard, the repo-local shortcut, and the staged Pages copy, including the steps an operator takes before, during, and after a draft.

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

### Requirement: The authoritative technical guide SHALL describe the implemented draft engine truthfully
The guide suite MUST use one authoritative technical/methods guide to describe the implemented draft board pipeline, prediction target, posterior construction, decision-policy layer, baselines, validation scope, and interval semantics in terms that match the current code and emitted artifacts.

#### Scenario: Technical guide distinguishes implemented vs conceptual content
- **WHEN** the authoritative technical guide discusses mathematical or modeling details
- **THEN** it MUST clearly distinguish implemented current-board behavior from conceptual intuition, optional analyses, and deprecated or compatibility-only surfaces

#### Scenario: Technical guide states targets and baselines explicitly
- **WHEN** the authoritative technical guide describes model outputs or validation
- **THEN** it MUST identify the forecast target, decision target, replacement baseline definitions, and internal-validation scope in explicit terms

#### Scenario: Technical guide explains interval semantics
- **WHEN** the authoritative technical guide describes uncertainty, floor/ceiling values, confidence intervals, or evidence summaries
- **THEN** it MUST specify what each interval or range represents and MUST NOT conflate predictive intervals, posterior intervals, percentile bands, or confidence intervals of the mean

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

### Requirement: Guide wording SHALL be adapted to the target audience
The guide suite MUST preserve canonical metric and trust-surface terminology while adapting the surrounding explanation to the intended audience instead of copying one glossary voice into every document.

#### Scenario: Audience-specific glossary explanation
- **WHEN** a guide explains a dashboard metric or trust surface
- **THEN** it MUST keep the canonical term but MAY adapt the explanation style, level of mathematical detail, and examples for operators, lay readers, or statisticians

### Requirement: Guide structure SHALL follow shared documentation conventions
The guide suite MUST follow a common documentation convention set so the docs remain predictable across audiences and resistant to drift.

#### Scenario: Each guide declares audience, scope, and trust level up front
- **WHEN** a reader opens a guide in the suite
- **THEN** the first screenful MUST identify who the guide is for, which supported workflow or surface it covers, and what trust boundary or interpretation limit the reader should keep in mind

#### Scenario: Guides reuse stable section patterns
- **WHEN** a guide explains a workflow, metric family, or trust surface
- **THEN** it SHOULD organize the explanation with stable recurring sections such as what it is, when to use it, what to inspect, what not to infer, and related commands or paths

#### Scenario: Commands and paths are paired with purpose and authority
- **WHEN** a guide lists a command, file, or path
- **THEN** it MUST explain what that command or path is for and whether it is authoritative, derived, optional, or publish-only rather than listing it without context

#### Scenario: Guides do not introduce competing primary names
- **WHEN** a guide explains a canonical metric or trust surface
- **THEN** it MUST NOT introduce a conflicting primary label for that concept even if the prose around it is adapted for the audience

### Requirement: Minimal payload snippets MAY be used to clarify workflow and interpretation
The guide suite MAY include small payload snippets when they materially improve understanding of a step, metric, or trust surface, but it MUST avoid turning the docs into raw payload dumps.

#### Scenario: Minimal snippet clarifies a guide step
- **WHEN** a small payload excerpt helps explain what a user should inspect during the workflow
- **THEN** the guide MAY include that minimal excerpt alongside an explanation of what to notice and why it matters

#### Scenario: Snippet use stays concise
- **WHEN** payload snippets are included in the guide suite
- **THEN** they MUST be limited to the fields needed for understanding and MUST NOT replace the narrative explanation

#### Scenario: Lay guide treats evidence as internal and directional
- **WHEN** the layperson guide describes backtest or decision-evidence results
- **THEN** it MUST identify that evidence as internal and directional rather than external proof that the board is universally correct

#### Scenario: Lay guide does not confuse visuals with new model semantics
- **WHEN** the layperson guide mentions shipped war-room visuals
- **THEN** it MUST explain that those visuals help interpret timing, scarcity, or baseline disagreement and MUST NOT imply that the visual layer is a separate model or a stronger validation claim than the underlying evidence supports

### Requirement: Documentation SHALL expose canonical path and data-lineage guidance
The guide suite MUST include a path and lineage reference that maps raw runtime inputs, processed datasets, canonical run artifacts, repo-local shortcuts, staged Pages files, and cloud publish targets.

#### Scenario: Path guide shows runtime and repo distinctions
- **WHEN** a user consults the path guide
- **THEN** it MUST distinguish runtime-root paths, repo-tracked files, staged `site/` outputs, and cloud-mirrored artifacts and MUST identify which ones are authoritative

#### Scenario: Path guide explains lineage from source data to final board
- **WHEN** a user consults the lineage guide
- **THEN** it MUST describe the flow from collected season files through preprocessing, unified dataset construction, VOR snapshot generation, dashboard artifact creation, and optional publish or retrospective steps
