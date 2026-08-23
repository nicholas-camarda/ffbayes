# 2026 draft engine: technical overview

This document describes the implemented current-season board. The source of
truth is the code under `src/ffbayes/draft_2026/`; this page explains its data
flow and equations so a reviewer can follow a row from inputs to recommendation.

## 1. Input and validation sequence

`ffbayes dashboard --year 2026` performs these steps:

1. Fetch the public ESPN fantasy feed for the requested season.
2. Fetch the current nflverse roster release and reconcile stable identities,
   positions, active status, and team status.
3. Parse projected statistics and ADP. Require minimum current-player,
   projection, and market coverage by position.
4. Reject retired, historical-only, inactive, duplicate, season-mismatched, or
   missing-ADP rows. Required transport or semantic failures produce a blocked
   service rather than an older or neutralized board.
5. Load one or more explicit `LeagueProfile` objects. The checked-in
   `config/leagues/example_2026.json` is only a portable template; ignored
   `*.local.json` files carry a user's actual settings.
6. Apply each profile to the same validated input snapshot and emit a payload
   with source, profile, state, board, and code digests.

The source adapters and coverage thresholds are implemented in
`src/ffbayes/draft_2026/sources.py`; orchestration and provenance are in
`pipeline.py`.

## 2. League profile contract

Profiles explicitly define season, team count, snake format, optional runtime
draft slot, scoring weights and position overrides, weekly bonuses, starter
slots, FLEX eligibility, bench/IR slots, waiver fields, and settings metadata.
The profile is validated before it can reach the engine. A draft slot may be
`null` until draft day; a current pick without a valid slot cannot produce
snake-pick timing.

## 3. Scoring

For player *i*:

```text
P_i = Σ_s projection_{i,s} × effective_weight_s + weekly_bonus_points_i
```

`effective_weight_s` is the profile's position-specific override when one is
configured, otherwise the global scoring weight. Weekly bonuses are evaluated
from weekly projection stats. The scoring stage rejects missing, negative, or
non-finite output.

## 4. Demand and replacement

Base position demand is starter slots multiplied by team count. FLEX demand is
allocated one slot at a time to the eligible position with the highest next
projected player after base demand. Bench demand is allocated one slot at a time
among QB/RB/WR/TE by the lowest current ADP. This produces a final demand count
for every valued position (QB, RB, WR, TE, DST, K).

For position *p*, replacement is the projected score at the final demand index:

```text
R_p = sorted_projected_points_p[final_demand_p - 1]
```

The engine requires at least one additional player beyond that index and at
least two distinct non-zero projected values. This makes replacement-level
calculations fail closed when the projection pool is too shallow or has no
signal.

## 5. Value and scarcity

```text
VOR_i = P_i - R_position(i)
```

Scarcity is the local projected-point drop around both starter demand and final
replacement demand. The internal value signal is standardized VOR plus
`0.25 × standardized_scarcity`. Model rank is combined with ADP rank as:

```text
board_priority = 0.60 × model_rank + 0.40 × market_rank
```

Rows are sorted by ascending `board_priority`, with VOR and ADP as tie-breakers.
This is why changing team count, FLEX slots, scoring, or roster depth can
change replacement levels, scarcity, valuation, and the resulting order.

## 6. Draft timing

For a snake draft, the service computes the user's next pick after the current
overall pick. ADP availability uses a normal approximation:

```text
sigma = max(6, 0.18 × ADP)
availability = 1 - Phi((next_pick - ADP) / sigma)
```

The recommendation is `draft_now` below availability `0.35` and `can_wait`
otherwise. If the user has not entered a slot, availability is null and the
recommendation is `slot_required`. There is no 50% missing-data fallback.

## 7. Runtime state and output

State is keyed by profile ID and stable ESPN IDs. Taken and yours cannot overlap;
unknown IDs are rejected. Successful board responses contain the decision table,
replacement demand/levels, coverage report, runtime state, and provenance
digests. Snapshot export is atomic and occurs only after
`validate_output_provenance` passes.

## 8. What this path does not claim

The 2026 dashboard is a league-aware projection and draft-timing tool. It does
not establish causal effects, guarantee outcomes, or replace a platform login.
The current path does not use a private-league API, historical staged HTML, or a
Bayesian posterior as a hidden substitute for current projections and ADP.
