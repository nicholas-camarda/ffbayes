# Metric Reference

Audience: operators, lay readers, and contributors who need a concise definition of dashboard terms without reading the full technical guide.

Scope: the current canonical metric names, trust surfaces, and related dashboard language used by the supported `pre-draft` workflow.

Trust boundary: canonical names should stay stable across docs and the dashboard payload. Explanations can vary by audience, but the primary metric names should not be silently renamed.

## What This Is

This document defines the dashboard's canonical terms and explains what each one helps with and what it does not justify.

## When To Use It

Use this reference when:

- a dashboard label is unclear
- you want the short version before reading the technical guide
- you want to check whether a term is part of the current board or only an optional analysis

## What To Inspect

Primary guide-facing payload fields:

- `metric_glossary`
- `model_overview`
- `decision_evidence`
- `analysis_provenance`
- `publish_provenance`

## What Not To Infer

- Do not rename primary metrics in other docs.
- Do not treat a glossary summary as a full validation statement.
- Do not assume similar-sounding terms mean the same thing.

## Metric Table

| Canonical term | What it is | What to inspect | What not to infer |
| --- | --- | --- | --- |
| `Board value score` | the board's base contextual player value before action rules | projection, starter edge, replacement edge, fragility, market gap | not a guarantee, not the pick-now instruction by itself |
| `Simple VOR proxy` | projected edge over replacement at the position | baseline comparison against the contextual board | not the same thing as the full board value score |
| `Starter delta` | projected edge over a typical starter at the position | whether a player meaningfully helps your likely starting lineup | not a whole-roster evaluation |
| `Availability to next pick` | estimated chance the player survives until your next turn | timing safety | not certainty, not a performance interval |
| `Expected regret` | estimated cost of waiting instead of taking now | whether passing is likely to be expensive | not a causal effect estimate |
| `Fragility score` | shakiness from uncertainty, injury, role volatility, and thin history | stability risk | not a medical diagnosis |
| `Upside score` | ceiling and breakout leverage | swing-for-the-fences appeal | not a promise of a breakout |
| `Wait signal` | plain-language summary of whether waiting is acceptable | take-now versus wait framing | not certainty about survival |
| `Market gap` | difference between model rank and market rank | where the board disagrees with cost | not proof the market is wrong |
| `Decision evidence` | internal holdout comparison between contextual and baseline strategies | status, winner, season count, limitations | not external validation |
| `Freshness and provenance` | run freshness plus source and staging metadata | status, warnings, override usage, source files | not proof the picks are correct |

## Trust Surfaces

### Metric Glossary

Purpose: short dashboard-owned summaries of the canonical metrics.

### Model Overview

Purpose: short dashboard-owned explanation of how the current board is structured.

### Decision Evidence

Purpose: summarize internal holdout support and interpretation limits.

### Analysis Provenance

Purpose: expose freshness state and source-input provenance for the runtime board.

### Publish Provenance

Purpose: expose when `site/` was staged and from which dashboard artifacts.

## Minimal Payload Example

```json
{
  "metric_glossary": {
    "draft_score": {
      "label": "Board value score"
    },
    "replacement_delta": {
      "label": "Simple VOR proxy"
    }
  },
  "model_overview": {
    "headline": "The draft board uses posterior player projections plus a starter-first decision policy."
  }
}
```

What to notice:

- the canonical labels live in the payload
- docs may explain them differently for different audiences, but should keep the labels

## War-Room Visual Terms When Present

If the current dashboard build includes `war_room_visuals`, the key related terms are:

- `Wait vs Pick Frontier`: timing tradeoff surface
- `Positional Cliffs`: position drop-off surface
- `Contextual vs baseline`: disagreement explainer between the contextual board and `Simple VOR proxy`

These are interpretation surfaces, not separate model outputs.

## Commands And Paths

Purpose: find the authoritative source for metric names and trust messaging.

- authoritative runtime payload: `runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json`
- staged Pages payload: `site/dashboard_payload.json`
- staged Pages provenance: `site/publish_provenance.json`
