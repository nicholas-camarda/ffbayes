# Layperson Guide

Audience: draft users who want the board explained in plain language, without
reading model code.

Scope: the supported `pre-draft` workflow, from building the board to reading it
on draft day.

Trust boundary: the board is decision support. It predicts and ranks from
historical evidence; it does not prove what will happen.

## What This Is

This guide answers three questions:

1. What is the board trying to do?
2. What do the main dashboard terms mean?
3. What should you avoid over-interpreting?

## When To Use It

Use this guide when you want the plain-English version of:

- where the board comes from
- what the main numbers mean
- what the dashboard is trying to help you do
- what conclusions you should avoid

## What To Inspect

Use these companion docs only when needed:

- [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md)
- [METRIC_REFERENCE.md](METRIC_REFERENCE.md)
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)

Main dashboard sections:

- recommendation lanes
- player inspector
- `Decision evidence`
- `Freshness and provenance`

## Interpretation Boundaries

- Do not treat the board as a promise about the future.
- Do not treat `Decision evidence` as proof that the model is correct in every league.
- Do not treat a high rank as a causal statement like "drafting this player causes you to win."

## The Big Picture

The board does four jobs:

1. gathers and cleans recent football and fantasy data
2. estimates each player's likely season output and uncertainty
3. compares each player to starter and replacement-level alternatives
4. turns those values into draft-day advice about picking now versus waiting

The result is not just "who projects best." It is "who is worth taking now,
given value, timing, and roster context."

## Step By Step

### 1. Data Collection

The project collects recent season data and builds a cleaned dataset.

Plain-language version:

- get recent fantasy results
- align names, positions, and related features
- build one analysis-ready table that the board can use

### 2. Player Projection

For each player, the model asks:

- what has this player done before?
- what do players like this usually do?
- how uncertain is this player compared with more stable profiles?

That produces a central estimate plus a range.

### Bayesian Words In Plain English

The board uses Bayesian updating language because it starts with a player
expectation, then updates that expectation with evidence from historical data.

- `Prior`: the starting expectation before the final model update. In this repo,
  it comes from draft-time-safe information such as recent player history,
  position context, availability, age, role, market signals, and rookie context
  when available.
- `Evidence`: the historical player-season patterns that can move the starting
  expectation up, down, or leave it mostly alone.
- `Posterior`: the updated forecast after the starting expectation and the
  evidence are combined.
- `Uncertainty`: how wide the plausible range is around the forecast. Wider
  uncertainty does not automatically mean the player is bad; it means the board
  has less narrow evidence about the outcome.
- `Shrinkage`: pulling thin or extreme player histories back toward broader
  position and player patterns instead of overreacting to a small sample.
- `Floor` and `ceiling`: lower and upper model outcome summaries. They are not
  guarantees and they are not exact worst-case or best-case outcomes.

The short version: the board is not just averaging last year's points. It is
combining a starting expectation, historical evidence, and uncertainty into a
forecast the draft policy can use.

### 3. Baseline Comparison

A raw point estimate is not enough for drafting. The board also asks:

- how much better is this player than a likely starter at the same position?
- how much better is this player than a replacement-level option at the same position?

Those comparisons matter because a 10-point edge at a thin position means something different from a 10-point edge at a deep position.

### 4. Draft Timing

The board also asks:

- if I do not take this player now, how likely is it that the player survives to my next pick?
- if the player does not survive, how much value might I lose by waiting?

That is why the dashboard has both value numbers and timing numbers.

### 5. Decision Evidence

The project also checks how the contextual board compared with a simpler VOR-style baseline on holdout seasons from the past.

That evidence is helpful, but it is still internal evidence. It tells you whether the system has some support inside its own historical setup. It does not prove future correctness in every league.

## Key Terms By Components

### Board Value Score

What it is:

- projected points
- plus edge over likely starters
- plus edge over replacement-level players
- plus a small market-disagreement signal
- minus a penalty for shakier profiles

What it helps with:

- ranking players on overall draft value before the "pick now or wait" decision layer

What it does not mean:

- it is not a guarantee the player will finish at that rank
- it is not a causal statement

### Simple VOR Proxy

What it is:

- projected points
- minus replacement-level points at that position

What it helps with:

- giving you a simpler baseline value-over-replacement view

What it does not mean:

- it is not the same as the full contextual board

### Availability To Next Pick

Break the term into parts:

- `availability`: how likely the player is still there later
- `next pick`: your next time on the clock

What it is:

- an estimated survival chance based on draft-market timing information

What it helps with:

- deciding whether waiting is realistic

What it does not mean:

- it is not certainty
- it is not a season-performance probability

### Expected Regret

Break the term into parts:

- `expected`: a model-based estimate
- `regret`: the cost of passing now and losing the player later

What it is:

- a wait penalty that gets larger when a player is valuable and unlikely to survive

What it helps with:

- deciding whether passing on the player is too risky

What it does not mean:

- it is not an emotional claim
- it is not a guaranteed point loss

### Fragility Score

Break the term into parts:

- injury or missed-time risk
- uncertainty from thin or noisy history
- role volatility
- disagreement or instability in supporting signals

What it is:

- a shakiness score

What it helps with:

- identifying players whose profile is less stable

What it does not mean:

- it is not a medical diagnosis
- it does not mean the player will fail

### Upside Score

Break the term into parts:

- ceiling above the average expectation
- chances of beating replacement
- timing leverage and raw projection strength

What it is:

- a ceiling and breakout score

What it helps with:

- identifying players who could return more than a conservative expectation

What it does not mean:

- it is not a promise of a breakout

### Decision Evidence

Break the term into parts:

- `decision`: this is about draft choices
- `evidence`: support from internal historical testing

What it is:

- a summary of how the contextual board compared with a simpler baseline on holdout seasons

What it helps with:

- deciding how much internal support the board currently has

What it does not mean:

- it is not outside validation
- it is not proof that the board will beat every alternative in your league

### Freshness And Provenance

Break the term into parts:

- `freshness`: how current the inputs are
- `provenance`: where the dashboard came from

What it is:

- metadata about whether the run used current expected inputs and how a staged surface was built

What it helps with:

- making sure you are not relying on stale or degraded outputs

What it does not mean:

- it does not tell you the picks are correct by itself

## How To Interpret The Dashboard

### Recommendation Lanes

Interpret them as:

- `pick now`: strong current action candidates
- `fallback`: reasonable alternatives if the top option is gone
- `can wait`: players who may survive to the next pick without too much cost

Do not interpret them as:

- guaranteed best action
- guaranteed survival

### Player Inspector

Use it to understand why a player is high or low.

Look for:

- board value
- simple VOR baseline
- timing
- upside
- fragility
- explanation flags

If you want the forecast details, open `Projection breakdown`. That is where the dashboard separates:

- total season expectation
- scoring rate when active
- expected games
- availability rate

Example interpretations:

- a team-changing veteran RB can still have a strong season-total forecast, but the `Team change` flag and the breakdown make it easier to see why uncertainty widened
- a rookie WR can show a thinner history and wider uncertainty because draft-year
  context and live depth-chart rank are doing more of the work than established
  NFL history

### Decision Evidence

Use it to ask:

- does this board have internal support?
- is the evidence degraded or unavailable?
- how big was the historical difference versus the baseline?

Do not use it to say:

- "this model is proven in all situations"

## What To Check On Draft Day

You do not need to read the dashboard payload or command output to use the
board. On draft day, check the visible dashboard surfaces:

- the board tells you whether evidence is available
- the board tells you whether the run was fresh or degraded
- if a validation cell says `n/a` or `not estimable`, the dashboard could not judge that slice cleanly
- `n/a` does not mean the relationship was exactly zero

Use [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) when you want to see concrete
artifact shapes. Use [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md)
when you need the actual commands.

## If Your Dashboard Includes War-Room Visuals

Treat them as quick interpretation tools:

- `Wait vs Pick Frontier`: value now versus waiting risk
- `Positional Cliffs`: where a position group may drop off
- `Contextual vs baseline`: why the board differs from a simpler VOR-style view

They are still part of the same board, not a separate model.

## Commands And Paths

Purpose: know where to go next without needing internal code details.

- workflow commands: [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md)
- artifact examples: [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)
- authoritative paths: [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md)

The repo and live dashboard use the same Bayesian-style player forecast described
above: a starting expectation updated with historical evidence and uncertainty.
