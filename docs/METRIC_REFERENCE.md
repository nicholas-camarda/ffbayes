# Metric Reference

Audience: operators, lay readers, and contributors who need short dashboard term
definitions.

Scope: canonical dashboard labels, payload keys, and what each term can and
cannot support.

Trust boundary: labels and payload keys should not drift. Explanations can vary
by audience; names should not.

## What This Is

This is the short glossary. Read the table left to right:

1. dashboard label
2. payload key
3. what it means
4. what to inspect
5. what not to infer

## When To Use It

Use this when:

- a dashboard label is unclear
- you want the short version before reading the technical guide
- you want to check the exact payload key

## What To Inspect

Primary guide-facing payload fields:

- `metric_glossary`
- `model_overview`
- `decision_evidence`
- `analysis_provenance`
- `publish_provenance`

## Interpretation Boundaries

- Do not rename primary metrics in other docs.
- Do not treat a glossary summary as a full validation statement.
- Do not assume similar-sounding terms mean the same thing.
- In this reference, `posterior` means the updated player forecast after the
  starting expectation and historical evidence are combined.

## Metric Table

| Canonical term | Payload key | What it is | What to inspect | What not to infer |
| --- | --- | --- | --- | --- |
| `Board value score` | `draft_score`, `board_value_score` | the board's base contextual player value before action rules | projection, starter edge, replacement edge, fragility, market gap | not a guarantee, not the pick-now instruction by itself |
| `Simple VOR proxy` | `replacement_delta` | projected edge over replacement at the position | baseline comparison against the contextual board | not the same thing as the full board value score |
| `Starter delta` | `starter_delta` | projected edge over a typical starter at the position | whether a player meaningfully helps your likely starting lineup | not a whole-roster evaluation |
| `Posterior beat-replacement probability` | `posterior_prob_beats_replacement` | chance the updated forecast clears replacement-level value | replacement probability beside projection uncertainty | not certainty and not the same as draft survival |
| `Availability to next pick` | `availability_to_next_pick` | estimated chance the player survives until your next turn | timing safety | not certainty, not a performance interval |
| `Expected regret` | `expected_regret` | estimated cost of waiting instead of taking now | whether passing is likely to be expensive | not a causal effect estimate |
| `Fragility score` | `fragility_score` | shakiness from uncertainty, injury, role volatility, and thin history | stability risk | not a medical diagnosis |
| `Upside score` | `upside_score` | ceiling and breakout leverage | swing-for-the-fences appeal | not a promise of a breakout |
| `Wait signal` | `wait_signal` | plain-language summary of whether waiting is acceptable | take-now versus wait framing | not certainty about survival |
| `Market gap` | `market_gap` | difference between model rank and market rank | where the board disagrees with cost | not proof the market is wrong |
| `Decision evidence` | `decision_evidence` | internal holdout comparison between contextual and baseline strategies | status, winner, season count, limitations | not external validation |
| `Freshness and provenance` | `analysis_provenance` | run freshness plus source and staging metadata | status, warnings, override usage, source files | not proof the picks are correct |
| `n/a` / `Not estimable` | validation table cells | validation metric state showing the slice could not support that estimate cleanly | whether the slice was constant or too thin | not the same as a measured zero relationship |
| `Season total mean` | `posterior_mean`, `proj_points_mean` | central season-total forecast shown in the inspector | updated forecast mean for total points | not a pick-now instruction by itself |
| `Rate when active` | `posterior_rate_mean` | expected scoring pace when the player is active | scoring-rate component of the player forecast | not full-season value without availability |
| `Expected games` | `posterior_games_mean` | expected availability over the fantasy season | games-played component of the player forecast | not a medical forecast |
| `Posterior floor` | `posterior_floor`, `proj_points_floor` | lower season-total forecast summary from posterior draws | downside range beside the central projection | not an exact worst-case outcome |
| `Posterior ceiling` | `posterior_ceiling`, `proj_points_ceiling` | upper season-total forecast summary from posterior draws | upside range beside the central projection | not an exact best-case outcome |
| `Posterior standard deviation` | `posterior_std` | spread of the season-total posterior forecast | whether the player forecast is narrow or wide | not a performance score by itself |
| `Uncertainty score` | `uncertainty_score` | normalized uncertainty-width signal used by the board | whether uncertainty is affecting interpretation or timing | not a probability |
| `Availability rate` | `availability_rate_projection` | expected-games fraction of the modeled season length | whether availability is dragging down total value | not draft-survival probability |

## Trust Surfaces

The dashboard also exposes short trust-surface summaries through
`metric_glossary`, `model_overview`, `decision_evidence`,
`analysis_provenance`, and `publish_provenance`.

If `Decision evidence` includes `n/a` or `Not estimable`, read that as "the
metric could not be estimated cleanly for this slice," not as "the relationship
was exactly zero." For concrete artifact shapes, use
[OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md).

## Commands And Paths

Purpose: find the authoritative source for metric names and trust messaging
without turning this file into an artifact guide.

- authoritative runtime payload: `seasons/<year>/draft_strategy/dashboard_payload_<year>.json`
- staged Pages payload: `site/dashboard_payload.json`
- staged Pages provenance: `site/publish_provenance.json`
- path authority details: [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md)
