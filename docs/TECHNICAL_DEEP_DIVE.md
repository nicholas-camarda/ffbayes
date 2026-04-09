# Technical Deep Dive

Audience: statisticians, technical reviewers, and contributors who need the implemented board math instead of a conceptual marketing summary.

Scope: the supported `pre-draft` workflow, the implemented player-posterior model, the board-construction layer, the recommendation policy, and the current trust surfaces.

Trust boundary: this document describes the implemented current board. Optional analyses and compatibility surfaces are labeled separately. Internal holdout backtests are directional evidence, not external validation.

## What This Is

This is the single authoritative technical and methods guide for the draft board. It replaces overlapping "deep dive" and "statistician guide" narratives.

The key distinction is:

- implemented current board behavior: what the supported `pre-draft` board actually does
- conceptual intuition: simplified explanations for why the implemented math is structured that way
- optional analyses: commands that exist but are not the default pre-draft operator path
- deprecated or compatibility-only surfaces: outputs kept for compatibility, not as the primary supported workflow

## When To Use It

Use this document when you need to answer questions like:

- what is the actual forecast target?
- how are priors and posterior estimates combined?
- what do `Board value score`, `Simple VOR proxy`, `Expected regret`, and `Fragility score` mean mathematically?
- what does the evidence panel validate, and what does it not validate?

## What To Inspect

Primary implementation sources:

- `src/ffbayes/analysis/bayesian_player_model.py`
- `src/ffbayes/draft_strategy/draft_decision_system.py`
- `src/ffbayes/draft_strategy/draft_decision_strategy.py`
- `src/ffbayes/analysis/draft_retrospective.py`
- `config/pipeline_pre_draft.json`

Primary emitted artifacts:

- `runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html`
- `runs/<year>/pre_draft/artifacts/draft_strategy/draft_decision_backtest_<year_range>.json`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`

## What Not To Infer

- Do not read the current board as a pure Monte Carlo ranking. The supported board is driven by posterior player projections plus a decision policy.
- Do not read the current board as a pure VOR ranking. `Simple VOR proxy` is a baseline comparator, not the full contextual score.
- Do not treat internal holdout backtests as external validity.
- Do not conflate floor and ceiling with frequentist confidence intervals unless the range is explicitly defined that way.

## Under The Hood In One Pass

If you want the shortest mathematically honest summary, the current board does this:

```math
\text{historical player-season table}
\rightarrow \text{prior features for each target player}
\rightarrow \text{recency-weighted empirical-Bayes regression}
\rightarrow \text{posterior mean and posterior uncertainty}
\rightarrow \text{starter and replacement baselines}
\rightarrow \text{board-level contextual score}
\rightarrow \text{draft-slot timing and regret policy}
\rightarrow \text{recommendation lanes, evidence, and dashboard trust surfaces}
```

The important thing to notice is that there are two different mathematical layers:

1. a player-level posterior layer
2. a draft-decision policy layer

That separation is the main reason the board can be confusing if the document only shows a few equations. A player can have a strong posterior projection but still be a weaker "pick now" recommendation if timing and roster context say waiting is fine.

## Implemented Workflow

The supported `pre-draft` runner in `config/pipeline_pre_draft.json` performs:

1. data collection
2. data validation
3. preprocessing
4. traditional VOR draft strategy
5. unified dataset creation
6. hybrid MC analysis artifacts
7. draft decision strategy
8. draft decision backtest

The default operator-facing board comes from step 7, with evidence supplied by step 8.

## Forecast Target And Decision Target

### Forecast Target

The implemented player model in `bayesian_player_model.py` forecasts player-season fantasy points for the target season using lagged historical player-season information and draft-time-safe features.

The main exported player-level outputs are:

- `posterior_mean`
- `posterior_std`
- `posterior_floor`
- `posterior_ceiling`
- `posterior_prob_beats_replacement`
- `uncertainty_score`

### Decision Target

The draft board does not stop at a player projection. It converts player-level posterior outputs into a draft-time decision surface:

- who is strongest on raw board value?
- who is strongest relative to starter and replacement baselines?
- who is likely to survive to the next pick?
- how costly is it to wait?
- how does current roster need change the action?

That second layer is what turns a projection table into a draft board.

## Player Prior Construction

The prior features come from `_player_prior_features(...)` in `bayesian_player_model.py`.

For a player with history, the prior mean is:

```math
\mathrm{prior\_mean}
=
\mathrm{shrinkage}\cdot \mathrm{recent\_mean}
+ (1-\mathrm{shrinkage})\cdot \mathrm{position\_mean}
+ 0.30\cdot \mathrm{player\_trend}
```

Where:

- `recent_mean` is the recent player-season scoring average
- `position_mean` is the historical average for the position
- `player_trend` is the within-player season trend
- `shrinkage = season_count / (season_count + 2.5)`

The prior standard deviation is widened or tightened from player volatility and position-level spread:

```math
\mathrm{prior\_std}
=
\max\left(
\mathrm{player\_weighted\_std}\cdot
\max\left(0.75,\ 1.20 - 0.08\cdot \min(\mathrm{season\_count},4)\right),
\ 0.55\cdot \mathrm{position\_std},
\ 6.0
\right)
```

If a player has no usable history, the prior falls back to position-level values rather than pretending to know a player-specific mean.

### What Data Enters The Prior

The prior is not built from one raw projection column. It is built from a draft-time-safe feature bundle that includes:

- player-season fantasy points from prior seasons
- games played and games missed
- age and years in league
- team-change rate
- role volatility
- recent ADP and ADP rank
- prior VOR-style values and market-proxy values
- site disagreement and related market instability features

In other words, the prior is trying to answer:

```math
\text{Before this season starts,}
\quad
\text{what should I expect from this player,}
\quad
\text{how noisy should that expectation be,}
\quad
\text{and how much should I shrink toward the position average?}
```

### Recency Weighting

Historical seasons are weighted with exponential decay:

```math
w_s = \mathrm{decay}^{(\mathrm{target\_season} - \mathrm{season}_s)}
```

with `decay = 0.72` in `_recency_weights(...)`.

That means:

- last season gets the most weight
- older seasons still matter
- old seasons are down-weighted rather than thrown away entirely

### Replacement Baseline In The Prior

Even before the draft board is built, `_player_prior_features(...)` computes a position-level replacement baseline from the position scoring distribution:

```math
\mathrm{replacement\_baseline}
=
Q_{\mathrm{replacement\_quantile}}(\mathrm{position\_points})
```

with the default `replacement_quantile = 0.2`.

That gives the player model an early notion of "beats replacement" before the board later recomputes league-shape-specific baselines.

## Empirical-Bayes Regression Layer

`fit_bayesian_regression(...)` fits a transparent empirical-Bayes linear model with recency weighting.

Inputs include:

- prior mean and prior standard deviation
- recent and latest points
- weighted player mean and volatility
- trend, injury, age, team-change, and role-volatility features
- ADP, VOR-style, and market-proxy features
- position indicators

Recent seasons are weighted more heavily with:

```math
w = \exp(-0.18\cdot \mathrm{season\_gap})
```

This regression produces:

- `regression_mean`
- `regression_std`

### Regression Model Form

At this stage the model is doing weighted linear regression with a Gaussian prior on coefficients:

```math
y = X\beta + \varepsilon
```

```math
\varepsilon \sim \mathcal{N}(0,\ \sigma^2_{\mathrm{obs}})
\qquad
\beta \sim \mathcal{N}\!\left(0,\ \Lambda_{\mathrm{prior}}^{-1}\right)
```

Where:

- `y` is target-season fantasy points in the training examples
- `X` contains standardized numeric features plus position indicators
- the coefficient prior acts like regularization

The implementation solves this in precision form:

```math
\Sigma
=
\left(
\Lambda_{\mathrm{prior}} + \beta X^\top W X
\right)^{-1}
```

```math
\mu_\beta
=
\beta \Sigma X^\top W y
```

where:

- `W` is the diagonal matrix of recency weights
- `beta = 1 / observation_variance`

This is why the document calls it empirical Bayes rather than a fully hand-tuned subjective prior: the regression structure is learned from the historical examples, then combined with shrinkage and coefficient regularization.

### Why This Regression Exists At All

The prior layer alone would mostly say:

- what this player did recently
- how volatile that player was
- what this position usually looks like

The regression layer adds a second question:

```math
\text{Across many historical examples, how do age, injury history, role volatility,}
\quad
\text{market features, and position-specific patterns shift the target outcome?}
```

That is what allows the board to generalize beyond a naive "last season plus shrinkage" rule.

## Posterior Combination

The final posterior combines the prior distribution with the empirical-Bayes regression estimate in closed form:

```math
\mathrm{posterior\_var}
=
\frac{1}{
\frac{1}{\mathrm{prior\_var}}
+
\frac{1}{\mathrm{regression\_var}}
}
```

```math
\mathrm{posterior\_mean}
=
\mathrm{posterior\_var}
\left(
\frac{\mathrm{prior\_mean}}{\mathrm{prior\_var}}
+
\frac{\mathrm{regression\_mean}}{\mathrm{regression\_var}}
\right)
```

The player table then exports:

- `posterior_std = sqrt(posterior_var)`
- `posterior_floor = posterior_mean - 1.2816 * posterior_std`
- `posterior_ceiling = posterior_mean + 1.2816 * posterior_std`

Under the Normal approximation used here, those floor and ceiling values are approximately 10th and 90th posterior percentile points, not a 95 percent confidence interval of the mean.

`posterior_prob_beats_replacement` is the posterior probability that the player clears the position replacement baseline.

### Intuition For The Posterior Combination

This is a precision-weighted average:

- if the prior is tight and the regression estimate is noisy, the posterior stays closer to the prior
- if the regression estimate is tight and the prior is wide, the posterior moves more toward the regression estimate
- if both are uncertain, the posterior variance stays wider

So the system is not "averaging two scores." It is averaging two uncertain distributions.

### One-Player Walkthrough

For a single player, the chain is:

```math
\text{historical seasons}
\rightarrow
\left(
\mathrm{recent\_mean},
\mathrm{player\_trend},
\mathrm{player\_weighted\_std},
\mathrm{position\_mean},
\mathrm{replacement\_baseline}
\right)
\rightarrow
(\mathrm{prior\_mean},\mathrm{prior\_std})
\rightarrow
(\mathrm{regression\_mean},\mathrm{regression\_var})
\rightarrow
(\mathrm{posterior\_mean},\mathrm{posterior\_std})
\rightarrow
(\mathrm{posterior\_floor},\mathrm{posterior\_ceiling},\Pr[\text{beats replacement}])
```

That player table is the input to the draft board. The board does not go back to raw weekly simulation draws at this point.

## Baselines Used By The Board

The board uses two distinct baselines in `draft_decision_system.py`:

- `starter_baseline`: a position-specific baseline derived from league starter slots
- `replacement_baseline`: a position-specific baseline derived from effective replacement slots

These create:

```math
\mathrm{starter\_delta}
=
\mathrm{proj\_points\_mean} - \mathrm{starter\_baseline}
```

```math
\mathrm{replacement\_delta}
=
\mathrm{proj\_points\_mean} - \mathrm{replacement\_baseline}
```

The dashboard label for `replacement_delta` is `Simple VOR proxy`. That is a baseline comparator, not the full contextual board.

### How The Baselines Are Computed

The board recomputes league-shape-specific baselines using `_position_baseline(...)`:

```math
\mathrm{baseline}(\mathrm{position},\mathrm{slot\_count})
=
\text{projection of the player ranked at slot\_count within that position}
```

So if the league shape changes, the starter and replacement baselines can move even when the player posterior table does not.

This matters because:

- `starter_delta` is about lineup advantage
- `replacement_delta` is about replacement-level advantage

Those are related but not identical targets.

## Risk, Upside, And Market Signals

The board then creates additional signals from posterior and historical features.

### Fragility Score

`Fragility score` combines:

- limited historical depth
- games missed and injury-related penalties
- age
- team changes
- role volatility
- site disagreement or ADP spread
- posterior unreliability through `1 - posterior_prob_beats_replacement`

Higher values mean the player profile is shakier.

### Upside Score

`Upside score` combines:

- ceiling-over-mean gap
- posterior probability of beating replacement
- next-pick survival rank percentile
- raw projection rank percentile

Higher values mean more ceiling and breakout leverage.

### Market Gap

`market_gap = market_rank - model_rank`

Positive values mean the current model likes the player more than the market cost suggests.

## Implemented Board Value Formula

The implemented board foundation is `board_value_score`:

```math
\mathrm{board\_value\_score}
=
0.40\,z(\mathrm{proj\_points\_mean})
+ 0.24\,z(\mathrm{starter\_delta})
+ 0.18\,z(\mathrm{replacement\_delta})
+ 0.10\,z(\Pr[\text{beats replacement}])
+ 0.05\,z(\mathrm{market\_gap})
+ 0.03\,z(\mathrm{starter\_need})
- \left(0.06\cdot \mathrm{risk\_multiplier}\right) z(\mathrm{fragility\_score})
```

Where `risk_multiplier` depends on risk tolerance:

- low: `0.80`
- medium: `1.00`
- high: `1.18`

The exported `draft_score` is currently this `board_value_score`. The dashboard labels it `Board value score`.

### Why Everything Is Z-Scored Here

The board combines terms measured on different scales:

- fantasy points
- probabilities
- rank gaps
- risk scores

Using `z(...)` puts these on a common standardized scale so the weights act on comparable inputs rather than letting one raw unit dominate only because it has a larger numeric range.

### What The Score Is Actually Doing

The formula can be read as:

```math
\text{good projection}
+ \text{starter advantage}
+ \text{replacement advantage}
+ \text{confidence of clearing replacement}
+ \text{some market-disagreement alpha}
+ \text{a little roster-need context}
- \text{fragility penalty}
```

That is why `Board value score` is best interpreted as a cleaned-up base ranking, not yet the final take-now/wait recommendation.

## Recommendation Policy Layer

The recommendation layer is separate from the raw board value ordering.

### Availability To Next Pick

`Availability to next pick` is a next-pick survival estimate built from ADP, ADP dispersion, and uncertainty.

The implementation uses a logistic transform:

```math
z = \frac{\mathrm{ADP} - \mathrm{target\_pick}}{\mathrm{spread}}
```

```math
\mathrm{availability\_to\_next\_pick}
=
\frac{1}{1+\exp(-z)}
```

with:

- `spread = adp_std` when available
- a fallback spread if ADP dispersion is missing
- additional widening from `uncertainty_score`

So the idea is:

- if ADP is much later than your next pick, survival rises
- if ADP is much earlier than your next pick, survival falls
- more uncertainty widens the spread and softens confidence

### Position Run Risk

`Position run risk` is derived from the remaining count at a position versus expected demand.

This is not a market model of every other drafter. It is a local scarcity heuristic tied to the currently available player pool and remaining roster demand.

### Starter Slot Urgency And Specialist Terms

The policy layer also computes:

- `starter_slot_urgency`: how much offensive starter need remains at the position
- `specialist_need_bonus`: a late-round-only bonus for DST and K when those slots still need filling
- `specialist_urgency`: a scarcity-style urgency term for late specialists

Those terms are why the action policy can differ from pure player value even when two players have similar board scores.

### Expected Regret

`Expected regret` is the wait penalty:

```math
\mathrm{expected\_regret}
=
\left(
0.55\cdot \mathrm{lineup\_gain\_percentile}
+ 0.25\cdot \mathrm{starter\_slot\_urgency}
+ 0.20\cdot \mathrm{position\_run\_risk}
\right)
\left(1-\mathrm{availability\_to\_next\_pick}\right)
```

### Action Utilities

The board computes separate utilities for acting now versus waiting:

```math
\mathrm{current\_pick\_utility}
=
\mathrm{draft\_score}
+ \mathrm{specialist\ bonuses}
+ 0.32\cdot \mathrm{starter\_slot\_urgency}
+ 0.22\cdot \mathrm{lineup\_gain\_percentile}
+ 0.08\cdot \Pr[\text{beats replacement}]
+ 0.06\cdot \mathrm{position\_run\_risk}
+ \mathrm{risk\_bias}\cdot \mathrm{upside\_score}
```

```math
\mathrm{wait\_utility}
=
\mathrm{draft\_score}\cdot \mathrm{availability\_to\_next\_pick}
+ 0.06\cdot \mathrm{upside\_score}
- 0.85\cdot \mathrm{expected\_regret}
```

That is why the recommendation lanes are not identical to the raw board ranking.

### Policy Eligibility

The board also applies gating rules before final "pick now" recommendations:

- DST and K are blocked outside the late specialist window
- secondary QB and TE picks can be suppressed when open offensive starter slots remain and the value edge is not large enough

So a player can be strong on raw board value but still be de-prioritized by the action layer.

### One Board-Row Walkthrough

For one available player at your current draft slot, the action chain is:

```math
\text{posterior projection row}
\rightarrow
(\mathrm{starter\_delta},\mathrm{replacement\_delta})
\rightarrow
(\mathrm{fragility\_score},\mathrm{upside\_score})
\rightarrow
\mathrm{board\_value\_score}
\rightarrow
\mathrm{availability\_to\_next\_pick}
\rightarrow
(\mathrm{starter\_slot\_urgency},\mathrm{position\_run\_risk})
\rightarrow
\mathrm{expected\_regret}
\rightarrow
(\mathrm{current\_pick\_utility},\mathrm{wait\_utility})
\rightarrow
\text{recommendation lane}
```

That is the cleanest way to think about "what is happening under the hood" on draft day.

## Decision Evidence And Validation Scope

`Decision evidence` is built from the draft-decision backtest plus freshness provenance.

The evidence surface tells you:

- whether a contextual board beat or lagged a simpler VOR-style baseline in internal holdout seasons
- how large the average lineup delta was
- which seasons were used
- what limitations apply

It does not tell you:

- that the board has been externally validated
- that the contextual score is universally better in all future leagues
- that a current recommendation is guaranteed to succeed

### What Is The Estimand Here

The evidence surface is not estimating a causal treatment effect of drafting one player. It is comparing strategy outputs on internal historical holdout seasons.

So the estimand is closer to:

```math
\text{average lineup-point difference between strategies}
\quad
\text{within the repo's internal holdout setup}
```

That is useful for calibration and trust-building, but it should not be interpreted as a universal external estimate.

## Interval Semantics

Terms used in the current board mean different things:

- `posterior_floor` and `posterior_ceiling`: approximate 10th and 90th posterior percentile points under the Normal approximation
- `uncertainty_score`: a normalized uncertainty-width score, not a probability
- `Availability to next pick`: a survival probability for draft timing, not a predictive interval for fantasy points
- `Expected regret`: a heuristic wait penalty, not a calibrated expected-value estimate in the causal-inference sense
- backtest summaries: internal seasonal comparison summaries, not confidence intervals of future performance

## Worked End-To-End Reading Guide

If you want to understand one dashboard row from top to bottom, read it in this order:

1. `proj_points_mean`
   This is the posterior central estimate for player-season fantasy points.
2. `proj_points_floor` and `proj_points_ceiling`
   These are approximate posterior percentile endpoints, not a 95 percent confidence interval of the mean.
3. `starter_delta`
   This asks whether the player helps your likely starting lineup.
4. `replacement_delta`
   This gives the simpler VOR-style baseline comparison.
5. `fragility_score` and `upside_score`
   These shape the risk/ceiling interpretation.
6. `draft_score` or `Board value score`
   This is the contextual base ranking.
7. `availability_to_next_pick`
   This is the timing survival estimate.
8. `expected_regret`
   This is the cost of waiting.
9. recommendation lane
   This is the action output after the policy layer.

If a reader cannot follow that chain, the document has not done its job.

## War-Room Visual Semantics When Present

If the current dashboard build includes `war_room_visuals`, the visuals are derived from existing recommendation, tier-cliff, and evidence semantics:

- `wait-vs-pick frontier`: timing tradeoff between board value, survival, and wait regret
- `positional cliffs`: where position groups drop sharply
- `contextual vs baseline explainer`: why the contextual board differs from the `Simple VOR proxy`

These visuals do not create a separate model. They are interpretation surfaces built on top of the same board and evidence contract.

## Optional Analyses

These commands exist, but they are optional rather than part of the default `ffbayes pre-draft` operator workflow:

- `ffbayes mc`
- `ffbayes agg`
- `ffbayes compare`
- `ffbayes bayesian-vor`
- `ffbayes publish --year <year>`

They can still be useful, but they should not be described as the current board's primary math unless the output path and producing command are made explicit.

## Commands And Paths

Authoritative runtime artifacts:

- `runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- `runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html`
- `runs/<year>/pre_draft/artifacts/draft_strategy/draft_decision_backtest_<year_range>.json`

Derived surfaces:

- `dashboard/index.html`
- `site/index.html`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`
