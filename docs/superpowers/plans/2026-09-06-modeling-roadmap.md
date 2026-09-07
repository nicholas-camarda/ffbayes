# FFBayes modeling assessment and implementation roadmap

Assessed 2026-09-06 against `source` HEAD `5c5b57bfe4dfeb689484746a23a5a3420c502c79` and the existing working tree. This is a proposal, not an implemented model or evidence of improved win probability. Existing dashboard/documentation edits were preserved.

## Assessment

The active dashboard does not perform Bayesian inference. It scores ESPN projections, calculates league-wide replacement values, adds positional scarcity, and combines model rank (60%) with ADP rank (40%). There is no fitted likelihood, posterior update, or posterior predictive player distribution in this path. A normal curve around ADP is a probability heuristic; it does not make the player model Bayesian.

The software has useful foundations: explicit scoring configuration, player identity checks, source validation, reproducible payloads, and draft-state management. Those engineering properties make it a credible platform for improvement. They do not establish an advantage over autodraft.

### Direct evidence from the current implementation

| Finding | Evidence | Decision consequence |
|---|---|---|
| Fixed ranking formula | `source/src/ffbayes/draft_2026/engine.py:284–295` | Weights are not learned or validated by the active engine. Rank blending also discards the size of point differences. |
| Owned players do not affect remaining values | Replacement and ranks are computed before `your_ids`/`taken_ids` are applied in `engine.py:239–345` | Filling QB does not lower the marginal value of another QB. Selected players are removed from availability, but remaining ranks and scarcity stay fixed. |
| Timing is unconditional | `engine.py:204–213` uses `sigma=max(6,0.18*ADP)` | It does not condition on the fact that a player has survived to the current pick, observed picks, or opponents' roster needs. |
| Probability-sounding evidence is heuristic | `source/src/ffbayes/draft_2026/pipeline.py:116–159` | `expected_regret` is a difference in projected player points; `position_run_risk` is scarcity divided by 25, clipped to [0,1]. Neither is the estimated quantity its name suggests. |
| General league configuration is incomplete | `league.py` and `calculate_replacement_levels` | One shared FLEX eligibility set cannot separately represent ordinary FLEX and superflex. Zero-demand positions are rejected by replacement calculation, affecting formats with no kicker or defense. |

A runtime probe using the checked-in engine fixture compared an empty roster with three owned QBs at the same pick. Every decision score, rank, and scarcity value was unchanged. The targeted engine/source/state/pipeline suite passed: **48 tests**. This verifies the inspected mechanics, not predictive quality. No historical model was refitted.

### Historical artifacts: clues, not current-model validation

Under `runtime/seasons/2026/`:

- `diagnostics/validation/sampled_bayes_promotion_report_2018-2025.json` records 72 sampled configurations, all rejected for `sampled_component_convergence_failed`. Its decision was to retain an empirical-Bayes estimator. This describes the historical experiment, not today's dashboard.
- `model_evaluation/bayesian_vs_vor_backtest_2018_to_2025.json` reports Bayesian MAE 36.69 versus market-proxy MAE 30.25, and rank correlation 0.671 versus 0.691. These stored results do not support a forecasting advantage over that proxy. Mixed player populations and the original evaluation implementation need auditing before stronger interpretation.
- `draft_strategy/draft_decision_backtest_2023-2025.json` reports mean lineup points 3165.99 for `draft_score` versus 3084.87 for `market`, an approximately 2.63% difference. Season differences are -196.52, +16.44, and +423.44. This is three seasons at one draft slot, not a championship-rate estimate.
- That same draft report includes a 2025 `consensus` roster of eight QBs and six defenses. The `market` roster lacks TE and K, and `draft_score` lacks DST. These comparisons are insufficient evidence against a competent roster-constrained autodraft policy.

The research-partner helper outputs are saved in `runtime/reviews/2026-09-06-model-backend/`. They are heuristic inventories; the substantive assessment above comes from reading code, inspecting artifacts, and the runtime probe. Its proposed runtime path is incorrect for this workspace; the root manifest and actual directories establish `source/` and `runtime/` as canonical. No topology change is required.

## General architecture

Build one reusable forecast backend. League and platform settings should be inputs, not separate model implementations or bespoke tuning for each user.

```text
Versioned observations and timestamped projection snapshots
        -> Shared weekly NFL stat distributions
        -> Configurable scoring and lineup rules
        -> Roster utility and opponent/market models
        -> Draft, lineup, and waiver decisions
        -> Independent historical and prospective evaluation
```

Forecast carries, targets, receptions, yards, touchdowns, and playing availability. Derive fantasy points from the same stat draws for standard, half-PPR, PPR, and other supported scoring. Apply nonlinear bonuses within each simulated week, rather than thresholding the mean projection. Preserve statistical constraints and shared team/game effects.

Distinguish player-performance uncertainty from draft-selection uncertainty. A player can have a stable performance forecast and an uncertain draft price, or the reverse. ADP primarily belongs in the acquisition model; test any incremental forecasting use separately.

Define a rules schema with separate eligible positions for each slot group, roster limits, bench/IR, scoring, schedules, playoff rules, and transaction rules. Use provider adapters for IDs, stat codes, ranking feeds, and draft events. Start the decision implementation with snake drafts across formats; auction, dynasty, best ball, and IDP need distinct policies and evaluation before being declared supported.

## Ordered implementation steps

### 1. Establish an honest baseline and data contract

Proposed additions: `src/ffbayes/data/snapshots.py`, `evaluation/rolling.py`, `evaluation/baselines.py`, and reproducible evaluation fixtures.

- Freeze the current engine as a versioned baseline.
- Inventory historical preseason projections and ADP. Record provider publication/availability time, fetch time, season/week, stable IDs, schema, content hash, and code version. A recent download of an old season does not establish that the data was available before its draft.
- Build player-week outcomes, including zero-point/inactive weeks. Define the eligible player population from information available at prediction time, including rookies and players who subsequently disappear.
- Establish baselines: current FFBayes, raw provider projection, roster-constrained ADP, and a simple roster-aware VOR policy. All must receive the same information and satisfy the same rules.
- Separate an actual platform-autodraft emulator from generic ADP heuristics. Validate the emulator against observed platform draft behavior before naming it as that platform's baseline.
- Use rolling season/week cutoffs. Fit preprocessing, priors, projection blends, and hyperparameters using training periods only; tune in earlier validation periods; reserve later periods for final evaluation. Years already used in historical searches are development evidence, not untouched holdouts.

**Acceptance:** temporal assertions reject future features; all agents draft legal, complete rosters; every reported score is reproducible from a manifest. If historical as-of inputs are unavailable, label replay results exploratory and begin prospective snapshot collection through a separately configured workflow.

### 2. Generalize rules and make decisions roster-aware

Proposed additions: `rules.py`, `decision/lineup.py`, `decision/roster_value.py`. Integrate through the existing engine and payload boundary.

- Replace the single FLEX definition with explicit slot eligibility groups. Permit absent K/DST and support separate FLEX/superflex without special-case ranking formulas.
- Enforce draft completion feasibility after every candidate pick, including position limits and remaining selections.
- Rank candidates by expected incremental usable lineup points, including bench insurance and plausible future replacements. Begin with point forecasts and simple continuation policies; posterior draws can replace point forecasts later through the same interface.
- Keep league-wide VOR as an explanatory baseline. Compute dynamic acquisition opportunity from remaining players and team needs; do not simply recalculate full-league demand on the shrinking pool.
- Rename heuristic evidence fields until their probabilistic definitions are implemented.

**Acceptance:** controlled fixtures show the value of an extra QB falls when the starting slot is already strong, rises in two-QB/superflex formats, and can remain positive as injury insurance. No duplicate slot assignment, impossible roster completion, or early specialist accumulation caused by raw point rankings. Paired draft evaluations compare this policy with Step 1 baselines.

### 3. Add a small, genuine Bayesian forecast challenger

Proposed additions: `forecast/base.py`, `forecast/hierarchical.py`, `forecast/predictive.py`, `evaluation/calibration.py`.

- First establish a simple probabilistic baseline using historical projection errors, grouped by position and projected usage. Then fit a Bayesian challenger with partial pooling and learned uncertainty.
- Use external projections as informative predictors/anchors and historical performance as evidence. Avoid treating projections and the historical inputs used to generate them as independent observations.
- Model availability, opportunity conditional on playing, and efficiency separately. Start with pooled availability and usage models; introduce injury-state duration or role transitions only when reliable inputs and validation justify them.
- For example, use a Bernoulli availability component, count distributions for touches/targets, bounded catch/share components, and position-specific efficiency models. Constrain team opportunity totals and include shared team/game variation as the simulator develops.
- Start with one provider and a small identifiable model. Add source blending only when timestamped histories exist, learning source bias and correlated errors rather than assuming independent experts.
- Emit posterior predictive stat draws with model ID and as-of timestamp. Distinguish uncertainty about latent ability from variability in future games. Cache draws for fast dashboard decisions; fit outside the interactive request path.

**Acceptance:** prior predictive checks produce plausible football outcomes; posterior predictive checks reproduce relevant zero rates/tails; sampling diagnostics and simulation-based calibration support computational correctness. Held-out CRPS/log score, interval coverage, and point error are reported by position, usage tier, and experience. Narrower intervals alone do not count as improvement. Retain the simpler baseline if the challenger fails.

Posterior prediction integrates future-observation variation over parameter uncertainty; merely sampling around fixed point projections is not equivalent. See [Stan posterior prediction](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html), [predictive checks](https://mc-stan.org/docs/2_38/stan-users-guide/posterior-predictive-checks.html), and [simulation-based calibration](https://mc-stan.org/docs/2_37/stan-users-guide/simulation-based-calibration.html).

### 4. Learn draft availability and evaluate waiting

Proposed additions: `market/selection.py`, `decision/draft_rollout.py`; extend draft state with selecting-team identity and uncertainty for missing events.

- As an interim correction, condition survival probabilities on a player still being available at the current pick, with explicit pick-index conventions and numerically stable tail calculations. This repairs conditioning but does not validate the normal assumption.
- Fit a selection model from complete draft logs using platform rank/ADP, round, scoring, roster needs, and observed picks. ADP averages alone cannot identify the distribution of draft positions.
- Simulate opponents selecting without replacement, updating rosters after each pick. Compare taking A now with taking B now and pursuing A or a fallback later, using matched random scenarios.
- Define regret as a difference in expected roster utility under explicit actions. Do not calculate it from two players' raw point projections.

**Acceptance:** next-pick availability is calibrated on held-out draft logs; all simulated drafts obey constraints; two-turn fixtures verify opportunity costs. Improvement must survive several opponent policies, not just the same model used to plan picks. Without adequate logs, label availability as a scenario estimate rather than a calibrated probability.

### 5. Evaluate seasons and championship utility

Proposed additions: `simulation/season.py`, `evaluation/tournaments.py`.

- Evaluate real weekly outcomes with a lineup-selection policy using only information available before lineup lock. Hindsight-best lineups are an upper bound, not a primary result.
- Include byes, legal substitutions, injury absences, team/game correlations, fantasy schedules, tiebreaks, playoffs, and realistic replacement/waiver access.
- Initially hold in-season management equal across draft policies to isolate draft effects. Later compare full systems with transaction budgets and identical information access.
- Evaluate all draft slots across 8/10/12/14-team formats, standard/half/PPR, one-QB/two-QB/superflex, shallow/deep benches, and no-K/no-DST cases. Use a representative crossed matrix rather than claim every possible combination was tested.
- Report usable points, weekly wins, playoff rate, championship rate, lower-tail results, and differences from each competent baseline. Break results down by format and season.

**Acceptance:** a predeclared primary decision metric improves on untouched or prospective data, with uncertainty reported across seasons/leagues. Repeated seeds measure simulation noise, not new independent football seasons. Report modeled championship probabilities as conditional on simulator assumptions; simulation alone cannot establish a real-world edge.

### 6. Reuse the backend for weekly decisions

Proposed additions: `forecast/update.py`, `decision/waivers.py`, `decision/start_sit.py`.

- Update usage and role estimates as new weekly evidence arrives, with shrinkage to avoid overreacting to one game.
- Use the same roster utility function for start/sit and add/drop choices. Add FAAB or waiver-priority opportunity costs when the applicable rules are supplied.
- Log recommendations before outcomes and assess changes against a frozen simple weekly baseline.

**Acceptance:** forward week-by-week evaluation uses release timestamps, realistic free-agent availability, and transaction constraints. Add/drop recommendations account for the dropped player's value. Improvements must persist under equal management effort assumptions.

Data constraint: nflverse documents regular player-stat and snap-count updates, but its injury source ended after 2024; FTN participation data is released after the postseason, not live during the season. Do not silently substitute either into a live/as-of feature set. Verify a separate timely source or explicitly omit unsupported features. [nflverse availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

## First implementation batch and promotion policy

Start with Steps 1 and 2: baseline contracts, legal roster simulation, generic slot rules, and marginal roster value. These address demonstrated defects and create a fair test bed for the Bayesian challenger. A larger historical MCMC search is not the next milestone.

Then implement Step 3 behind an explicit model version. Steps 3 and 4 can develop independently once data contracts exist; integrate both through Step 5 before claiming a league-winning advantage. Step 6 reuses the same forecasts and roster utility.

Predeclare meaningful improvement thresholds before the final evaluation, based on baseline variance and the practical size of an advantage. Use paired comparisons and ablations: better forecasts with fixed policy, better policy with fixed forecasts, then the combination. Require acceptable format-specific behavior, reproducibility, and interactive latency (proposed target: under two seconds at the 95th percentile with cached draws). Keep the current baseline available for rollback and comparison.

The intended outcome is a general, testable decision system. Being Bayesian is a modeling choice; earning confidence requires calibrated predictions and better decisions against credible alternatives.
