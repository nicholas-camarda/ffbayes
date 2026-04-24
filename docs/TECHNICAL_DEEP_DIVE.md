# Technical Deep Dive

Audience: statisticians, technical reviewers, and contributors who need the
implemented math and decision logic.

Scope: the supported `pre-draft` workflow, player posterior model, board-value
formula, recommendation policy, and trust surfaces.

Trust boundary: this describes the current implemented board. Internal holdout
backtests are directional evidence, not external validation. Non-default analyses
are labeled explicitly.

When a rank-based validation slice lacks variation, the artifact records it as
unavailable and the dashboard renders `n/a` or `not estimable`. That is not a
measured zero relationship.

## What This Is

This is the technical source for what the supported draft board actually does.
It separates:

- implemented current board behavior: what the supported `pre-draft` board actually does
- conceptual intuition: simplified explanations for why the implemented math is structured that way
- additional non-default analyses: commands that exist but are not the default pre-draft operator path

If you need plain-English definitions first, start with
[LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md). This document assumes you want the
math and implementation contract.

## When To Use It

Use this document to answer:

- what is the actual forecast target?
- how are priors and posterior estimates combined?
- what do `Board value score`, `Simple VOR proxy`, `Expected regret`, and `Fragility score` mean mathematically?
- what does the evidence panel support, and what does it not prove?

## What To Inspect

Implementation sources:

- `src/ffbayes/analysis/bayesian_player_model.py`
- `src/ffbayes/draft_strategy/draft_decision_system.py`
- `src/ffbayes/draft_strategy/draft_decision_strategy.py`
- `src/ffbayes/analysis/draft_retrospective.py`
- `config/pipeline_pre_draft.json`

Emitted artifacts:

- `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- `seasons/<year>/draft_strategy/draft_board_<year>.html`
- `seasons/<year>/draft_strategy/draft_decision_backtest_<year_range>.json`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`

## Interpretation Boundaries

- The board is driven by posterior player projections plus a decision policy.
- `Simple VOR proxy` is a baseline comparator; the contextual board score also uses starter advantage, replacement advantage, uncertainty, market gap, roster need, timing, and regret.
- Player forecasts use the hierarchical empirical-Bayes estimator.
- Internal holdout backtests are directional evidence, not external validity.
- Floor and ceiling fields are posterior percentile summaries unless another range is explicitly defined.

## Under The Hood In One Pass

Shortest technically accurate summary:

1. historical player-season table
2. prior features for each target player
3. recency-weighted empirical-Bayes regression
4. posterior mean and posterior uncertainty
5. starter and replacement baselines
6. board-level contextual score
7. draft-slot timing and regret policy
8. recommendation lanes, evidence, and trust surfaces

There are two separate mathematical layers:

1. a player-level posterior layer
2. a draft-decision policy layer

A player can have a strong posterior projection but still be a weaker "pick now"
recommendation if timing and roster context say waiting is acceptable.

## Bayesian Terms Used Here

For plain-English definitions of prior, posterior, uncertainty, shrinkage, and
related dashboard language, start with [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md).
This section keeps only the technical meaning needed for the implementation
details below. These are implementation terms in an empirical-Bayes pipeline:
the model does not sample a full joint posterior over all model parameters.

- `prior`: the draft-time player distribution built from safe pre-season
  features before the empirical regression update.
- `likelihood`: the weighted historical regression evidence model that
  connects draft-time features to target-season fantasy points.
- `posterior`: the updated player distribution after combining the prior and the
  empirical regression estimate.
- `posterior_mean`: the season-total central estimate used as an input to board
  construction, not the final pick-now recommendation.
- `posterior_std`, `posterior_floor`, and `posterior_ceiling`: uncertainty and
  percentile summaries from the posterior predictive player distribution.
- `posterior_prob_beats_replacement`: the posterior probability that the player
  clears the replacement baseline used by the player model.

The implementation is empirical Bayes: historical data estimate the regression
and uncertainty behavior, while player-specific prior features provide each
target player a draft-time starting distribution.

## Implemented Workflow

The supported `pre-draft` runner in `config/pipeline_pre_draft.json` performs:

1. data collection
2. data validation
3. preprocessing
4. unified dataset creation
5. optional traditional VOR baseline artifact generation
6. draft decision strategy
7. draft decision backtest

The default operator-facing board comes from step 6. Player-forecast generation
is internal to `draft_decision_strategy`, and evidence is supplied by step 7.

## Forecast Target And Decision Target

### Forecast Target

The implemented player model in `bayesian_player_model.py` forecasts season-total fantasy points for the target season by modeling scoring rate and availability separately, then composing them through posterior predictive simulation.

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

For a player with history, the prior structure is built from separate rate and availability components.

The rate prior mean is a shrinkage blend of recent player rate and position rate, with team-season context allowed to adjust it:

$$
\begin{aligned}
\mu_{\text{rate prior}}
&= \alpha\,r_{\text{recent}} \\
&\quad + (1-\alpha)\,r_{\text{position}}
\end{aligned}
$$

Where:

- `recent_rate` is the recent scoring rate when active
- `position_rate_mean` is the historical rate average for the position
- `shrinkage` is:

$$
\alpha =
\frac{n_{\text{seasons}}}{n_{\text{seasons}}+2.5}
$$

The constant `2.5` is a prior-strength setting. It makes the position-level
reference behave like roughly two and a half stabilizing seasons. With one NFL
season of player history, the player-specific rate receives about:

$$
\frac{1}{1+2.5}\approx 0.29
$$

of the blend. With five seasons, it receives about:

$$
\frac{5}{5+2.5}\approx 0.67
$$

That is deliberate: one season should move the prior, but it should not erase
the position context.

The availability prior mean is built from weighted historical games played, with team-season context allowed to adjust it:

$$
\mu_{\text{games prior}} =
\text{weighted historical games played}
$$

Season-total priors are then composed from those two components. The production season-total posterior is not a direct one-stage mean-only regression.

If a player has no usable history, the prior is driven by position context plus explicit current/prior draft-year rookie inputs such as draft pick, combine-derived signal, and live depth-chart context rather than pretending to know a player-specific NFL history.

### What Data Enters The Prior

The prior is not built from one raw projection column. It is built from a draft-time-safe feature bundle that includes:

- season-total fantasy points from prior seasons
- scoring rate from prior seasons
- games played and games missed
- age and years in league
- team-season context and team-change indicators
- current/prior draft-year rookie draft pick, combine signal, and live depth-chart context when available
- team-change rate
- role volatility
- recent ADP and ADP rank
- prior VOR-style values and market-proxy values
- site disagreement and related market instability features

The rookie draft/combine fields are intentionally not a broad historical
prospect-feature backfill. The installed nflreadpy draft data has usable player
IDs, but the combine feed does not expose stable IDs in the same way, so the
project does not train historical rows by fuzzy backfilling combine data across
old seasons. The supported behavior is narrower: use the current and immediately
prior draft years to support the live board and no-history player priors, with
hard coverage checks so the dashboard cannot publish silently null rookie
context.

The prior is trying to answer three linked questions:

- what should I expect from this player before the season starts?
- how noisy is that expectation?
- how much should the estimate shrink toward the position average?

Together, those answers define the player-specific prior distribution.

### Recency Weighting

Historical seasons are weighted with exponential decay:

$$
w_s =
d^{t_{\text{target}}-s}
$$

where `d = 0.72` in `_recency_weights(...)`.

That means:

- last season gets the most weight
- older seasons still matter
- old seasons are down-weighted rather than thrown away entirely

The value `0.72` means each additional year back keeps 72 percent of the prior
year's weight. For example:

$$
w_{\text{last year}} = 0.72
\qquad
w_{\text{two years back}} = 0.72^2 \approx 0.52
\qquad
w_{\text{three years back}} = 0.72^3 \approx 0.37
$$

So the model favors recent form while still allowing older seasons to stabilize
thin histories.

### Replacement Baseline In The Prior

Even before the draft board is built, `_player_prior_features(...)` computes a position-level replacement baseline from the position scoring distribution:

$$
B_{\text{replacement}} =
Q_{0.20}(Y_{\text{position points}})
$$

The default replacement quantile is:

$$
q_{\text{replacement}}=0.20
$$

That gives the player model an early notion of "beats replacement" before the board later recomputes league-shape-specific baselines.

The 20th percentile is intentionally a low position-specific reference point,
not a median starter threshold. At this stage the model needs a broad
"replacement-level" anchor for posterior probability calculations; the later
draft-board layer computes league-shape-specific starter and replacement
baselines again.

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

$$
w =
\exp(-0.18\cdot \Delta_{\text{season}})
$$

The `0.18` decay coefficient is the regression-layer recency penalty. It implies
roughly:

$$
\exp(-0.18)\approx 0.84
\qquad
\exp(-0.36)\approx 0.70
\qquad
\exp(-0.54)\approx 0.58
$$

for examples one, two, and three seasons behind the most recent training season.
This is milder than dropping old examples outright. The regression still learns
from older data, but the newest player-season relationships contribute more.

This regression produces:

- `regression_mean`
- `regression_std`

### Regression Model Form

At this stage the model is doing weighted linear regression with a Gaussian prior on coefficients:

$$
y = X\beta + \varepsilon
$$

$$
\varepsilon \sim \mathcal{N}(0,\ \sigma^2_{\text{obs}})
\qquad
\beta \sim \mathcal{N}\!\left(0,\ \Lambda_{\text{prior}}^{-1}\right)
$$

Where:

- `y` is target-season fantasy points in the training examples
- `X` contains standardized numeric features plus position indicators
- the coefficient prior acts like regularization

The implementation solves this in precision form:

$$
\Sigma =
\left(\Lambda_{\text{prior}} + \tau X^\top W X\right)^{-1}
$$

$$
\mu_\beta =
\tau \Sigma X^\top W y
$$

where:

- $W$ is the diagonal matrix of recency weights
- $\tau = 1 / \sigma^2_{\text{obs}}$
- $\Lambda_{\text{prior}}$ is the coefficient-prior precision matrix

This is why the document calls it empirical Bayes rather than a fully hand-tuned subjective prior: the regression structure is learned from the historical examples, then combined with shrinkage and coefficient regularization.

### Why This Regression Exists At All

The prior layer alone would mostly say:

- what this player did recently
- how volatile that player was
- what this position usually looks like

The regression layer adds a second question:

Given historical player-season training examples, how do age, experience,
injury and availability history, role volatility, market and ADP features, and
position-specific patterns shift the target-season outcome?

That is what allows the board to generalize beyond a naive "last season plus shrinkage" rule.

## Posterior Combination

The final posterior combines the prior distribution with the empirical-Bayes regression estimate in closed form:

$$
\sigma^2_{\text{post}} =
\left(
\frac{1}{\sigma^2_{\text{prior}}} + \frac{1}{\sigma^2_{\text{reg}}}
\right)^{-1}
$$

$$
\mu_{\text{post}} =
\sigma^2_{\text{post}}\left(
\frac{\mu_{\text{prior}}}{\sigma^2_{\text{prior}}} + \frac{\mu_{\text{reg}}}{\sigma^2_{\text{reg}}}
\right)
$$

The player table then exports:

- `posterior_rate_mean` and `posterior_rate_std`
- `posterior_games_mean` and `posterior_games_std`
- `posterior_mean`
- `posterior_std`
- `posterior_floor`
- `posterior_ceiling`

The season-total posterior is composed by simulation:

$$
r^{(m)} \sim \mathcal{N}(\mu_{\text{rate}},\sigma^2_{\text{rate}})
\qquad
g^{(m)} \sim \mathcal{N}(\mu_{\text{games}},\sigma^2_{\text{games}})
$$

$$
y^{(m)} = \max(r^{(m)},0)\cdot \operatorname{clip}(g^{(m)},0,G_{\text{season}})
$$

with `512` draws per player. The exported summaries are:

$$
\mu_{\text{post}} =
\frac{1}{512}\sum_{m=1}^{512} y^{(m)}
$$

$$
\begin{aligned}
Q_{\text{floor}}
&= Q_{0.10}\!\left(y^{(1)},\ldots,y^{(512)}\right) \\
Q_{\text{ceiling}}
&= Q_{0.90}\!\left(y^{(1)},\ldots,y^{(512)}\right)
\end{aligned}
$$

The 10th and 90th percentiles are chosen as a readable downside/upside band.
They are not a 95 percent confidence interval of the mean. The draw count `512`
is an implementation balance: enough draws to produce stable dashboard
summaries, small enough to keep repeated backtests and dashboard generation
fast.

`posterior_prob_beats_replacement` is the posterior probability that the player clears the position replacement baseline:

$$
\Pr(Y > R) =
\Phi\left(
\frac{\mu_{\text{post}}-R}{\max(\sigma_{\text{post}},1)}
\right)
$$

where $R$ is the replacement baseline and $\Phi$ is the standard Normal CDF. The
$\max(\sigma_{\text{post}},1)$ denominator prevents unstable probabilities when
a simulated posterior spread is extremely small.

### Intuition For The Posterior Combination

This is a precision-weighted average:

- if the prior is tight and the regression estimate is noisy, the posterior stays closer to the prior
- if the regression estimate is tight and the prior is wide, the posterior moves more toward the regression estimate
- if both are uncertain, the posterior variance stays wider

So the system is not "averaging two scores." It is averaging two uncertain distributions.

### One-Player Walkthrough

For a single player, the chain is:

1. historical seasons
2. recent mean, trend, volatility, position mean, and replacement baseline
3. prior mean and prior standard deviation
4. regression mean and regression variance
5. posterior mean and posterior standard deviation
6. posterior floor, posterior ceiling, and beat-replacement probability

That player table is the input to the draft board. The board does not go back to raw weekly simulation draws at this point.

## Baselines Used By The Board

The board uses two distinct baselines in `draft_decision_system.py`:

- `starter_baseline`: a position-specific baseline derived from league starter slots
- `replacement_baseline`: a position-specific baseline derived from effective replacement slots

These create:

$$
D_{\text{starter}} =
\mu_{\text{proj}} - B_{\text{starter}}
$$

$$
D_{\text{replacement}} =
\mu_{\text{proj}} - B_{\text{replacement}}
$$

The dashboard label for `replacement_delta` is `Simple VOR proxy`. That is a baseline comparator, not the full contextual board.

### How The Baselines Are Computed

The board recomputes league-shape-specific baselines using `_position_baseline(...)`:

$$
B(p,k) =
\text{projection of the player ranked } k \text{ within position } p
$$

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

$$
\begin{aligned}
B_{\text{value}}
&= 0.40\,z(\mu_{\text{proj}}) \\
&\quad + 0.24\,z(D_{\text{starter}}) \\
&\quad + 0.18\,z(D_{\text{replacement}}) \\
&\quad + 0.10\,z(\Pr[\text{beats replacement}]) \\
&\quad + 0.05\,z(M_{\text{gap}}) \\
&\quad + 0.03\,z(N_{\text{starter}}) \\
&\quad - \left(0.06\cdot R_{\text{risk}}\right) z(F_{\text{fragility}})
\end{aligned}
$$

Where `risk_multiplier` depends on risk tolerance:

- low: `0.80`
- medium: `1.00`
- high: `1.18`

The exported `draft_score` is currently this `board_value_score`. The dashboard labels it `Board value score`.

The board-value weights are policy weights, not regression coefficients learned
directly from the backtest. They encode the intended ordering of concerns:

$$
0.40 > 0.24 > 0.18 > 0.10 > 0.05 > 0.03
$$

The largest term keeps the board anchored to the central posterior projection.
The next two terms give starter and replacement advantage real influence, so the
board is not just a raw points sort. Smaller terms let replacement probability,
market gap, and current starter need break ties or nudge close calls without
dominating the core projection. The fragility penalty is deliberately modest and
risk-adjusted:

$$
P_{\text{fragility}} =
0.06\cdot R_{\text{risk}}\cdot z(F_{\text{fragility}})
$$

so changing risk tolerance changes how much shakiness hurts the score, while
preserving the same base board-value structure.

### Why Everything Is Z-Scored Here

The board combines terms measured on different scales:

- fantasy points
- probabilities
- rank gaps
- risk scores

Using `z(...)` puts these on a common standardized scale so the weights act on comparable inputs rather than letting one raw unit dominate only because it has a larger numeric range.

### What The Score Is Actually Doing

The formula can be read as:

- good projection
- plus starter advantage
- plus replacement advantage
- plus confidence of clearing replacement
- plus some market-disagreement alpha
- plus a little roster-need context
- minus fragility penalty

That is why `Board value score` is best interpreted as a cleaned-up base ranking, not yet the final take-now/wait recommendation.

## Recommendation Policy Layer

The recommendation layer is separate from the raw board value ordering.

### Availability To Next Pick

`Availability to next pick` is a next-pick survival estimate built from ADP, ADP dispersion, and uncertainty.

The implementation uses a logistic transform:

$$
z = \frac{\mathrm{ADP} - P_{\text{target}}}{S_{\text{spread}}}
$$

$$
A_{\text{next pick}} =
\frac{1}{1+\exp(-z)}
$$

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

$$
\begin{aligned}
R_{\text{wait}}
&= \left(
0.55\cdot G_{\text{lineup}}
{}+ 0.25\cdot U_{\text{starter slot}}
{}+ 0.20\cdot R_{\text{position run}}
\right) \\
&\quad \cdot \left(1-A_{\text{next pick}}\right)
\end{aligned}
$$

### Action Utilities

The board computes separate utilities for acting now versus waiting:

$$
\begin{aligned}
U_{\text{pick now}}
&= S_{\text{draft}}
{}+ B_{\text{specialist}} \\
&\quad + 0.32\cdot U_{\text{starter slot}}
{}+ 0.22\cdot G_{\text{lineup}} \\
&\quad + 0.08\cdot \Pr[\text{beats replacement}]
{}+ 0.06\cdot R_{\text{position run}} \\
&\quad + B_{\text{risk}}\cdot S_{\text{upside}}
\end{aligned}
$$

$$
\begin{aligned}
U_{\text{wait}}
&= S_{\text{draft}}\cdot A_{\text{next pick}} \\
&\quad + 0.06\cdot S_{\text{upside}}
{}- 0.85\cdot R_{\text{wait}}
\end{aligned}
$$

That is why the recommendation lanes are not identical to the raw board ranking.

### Policy Eligibility

The board also applies gating rules before final "pick now" recommendations:

- DST and K are blocked outside the late specialist window
- secondary QB and TE picks can be suppressed when open offensive starter slots remain and the value edge is not large enough

So a player can be strong on raw board value but still be de-prioritized by the action layer.

### One Board-Row Walkthrough

For one available player at your current draft slot, the action chain is:

1. posterior projection row
2. starter delta and replacement delta
3. fragility score and upside score
4. board value score
5. availability to next pick
6. starter-slot urgency and position-run risk
7. expected regret
8. current-pick utility and wait utility
9. recommendation lane

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

the average lineup-point difference between the contextual board strategy and a
VOR-style baseline strategy on internal historical holdout seasons.

That is useful for internal strategy support and trust triage, but it should not be interpreted as a universal external estimate.

## Interval Semantics

Terms used in the current board mean different things:

- `posterior_floor` and `posterior_ceiling`: approximate 10th and 90th percentile points from simulated season-total posterior draws
- `uncertainty_score`: a normalized uncertainty-width score, not a probability
- `Availability to next pick`: a survival probability for draft timing, not a predictive interval for fantasy points
- `Expected regret`: a heuristic wait penalty, not a calibrated expected-value estimate in the causal-inference sense
- backtest summaries: internal seasonal comparison summaries, not confidence intervals of future performance

## Worked End-To-End Reading Guide

If you want to understand one dashboard row from top to bottom, read it in this order:

1. `proj_points_mean`
   This is the posterior central estimate for season-total fantasy points.
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

## Inspector Projection Breakdown

The player inspector exposes the rate-and-availability decomposition in a dedicated `Projection breakdown` section rather than in the main board columns.

The visible inspector fields are:

- `Season total mean`
- `Rate when active`
- `Expected games`
- `Availability rate`
- `Current team`
- `Team change`

When available, the same section also shows current/prior draft-year rookie context such as draft pick, combine-derived signal, and depth-chart rank. That inspector section is explanatory. The canonical board ordering still comes from the season-total decision contract.

## War-Room Visual Semantics When Present

If the current dashboard build includes `war_room_visuals`, the visuals are derived from existing recommendation, tier-cliff, and evidence semantics:

- `wait-vs-pick frontier`: timing tradeoff between board value, survival, and wait regret
- `positional cliffs`: where position groups drop sharply
- `contextual vs baseline explainer`: why the contextual board differs from the `Simple VOR proxy`

These visuals do not create a separate model. They are interpretation surfaces built on top of the same board and evidence contract.

## Additional Commands

These commands exist outside the default `ffbayes pre-draft` operator workflow:

- `ffbayes mc`
- `ffbayes bayesian-vor`
- `ffbayes publish --year <year>`

They should not be described as the current board's primary math unless the output path and producing command are made explicit.

## Commands And Paths

Authoritative runtime artifacts:

- `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- `seasons/<year>/draft_strategy/draft_board_<year>.html`
- `seasons/<year>/draft_strategy/draft_decision_backtest_<year_range>.json`

Derived surfaces:

- `dashboard/index.html`
- `site/index.html`
- `site/dashboard_payload.json`
- `site/publish_provenance.json`
