# 2026 metric reference

The dashboard values come from `src/ffbayes/draft_2026/engine.py`. A profile
changes the scoring weights, roster demand, FLEX allocation, bench demand, and
snake-pick timing; those settings are part of the calculation, not just labels.

## Scoring

For player *i*, projected fantasy points are:

```text
P_i = Σ_s projection_{i,s} × scoring_weight_s + configured bonus points
```

Position-specific scoring overrides replace the global weight for that stat.
Configured weekly bonuses are evaluated from weekly projection stats. Negative,
non-finite, or missing scored values fail validation.

## Roster demand and replacement

Base demand for position *p* is:

```text
base_p = starter_slots_p × team_count
```

FLEX demand is allocated greedily across the profile's eligible positions by
the next projected player after base demand. Bench demand is allocated across
QB/RB/WR/TE in current ADP order. The final demand is base starters plus FLEX
allocation plus bench allocation.

The replacement level for position *p* is the projected score at the final
demand index:

```text
R_p = sorted_projected_points_p[final_demand_p - 1]
```

The engine requires at least one player beyond that index and usable projection
signal. Shallow projections or empty positions fail closed.

## Value and scarcity

Value over replacement is:

```text
VOR_i = P_i - R_position(i)
```

Scarcity is the local projected-point drop around both starter demand and final
replacement demand. It is scaled within the current board; it is not a league-
independent constant.

The internal value signal combines standardized VOR with a quarter-weighted
standardized scarcity term. The board then blends model rank (60%) and ADP rank
(40%) into `board_priority`; lower priority is better. This market blend keeps
the board responsive to both league-adjusted value and current draft cost.

## ADP timing

ADP is the current ESPN market reference. Missing or stale ADP is an input
error, not a neutral default.

For an explicit snake-draft next pick *q*, availability uses a normal
approximation centered at ADP with:

```text
σ = max(6, 0.18 × ADP)
availability = 1 - Φ((q - ADP) / σ)
```

The action is `draft_now` when availability is below `0.35`, otherwise
`can_wait`. Without a runtime draft slot, availability is null and the action
is `slot_required`; the system never substitutes a mechanical 50% value.

## Draft state

`available`, `taken`, and `mine` are keyed by stable ESPN IDs. A player cannot
be both taken and yours, and unknown IDs are rejected. Queue state is retained
per profile by the dashboard service.

## Interpretation

These are descriptive decision-support metrics. They are not causal effects,
calibrated win probabilities, or guarantees about future performance. A
blocked or unavailable metric means it was not estimable from valid inputs; it
does not mean a measured zero relationship.
