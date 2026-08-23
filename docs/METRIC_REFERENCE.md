# 2026 Dashboard Metric Reference

Audience: anyone reading a current board row.

Scope: the values returned by the Python 2026 engine.

Trust boundary: these are model outputs from validated inputs, not guarantees
about future performance or draft outcomes.

## What This Is

The dashboard exposes the following fields for each player.

| Label | Meaning |
| --- | --- |
| Projected points | Configured league scoring applied to the current-season projection stats. |
| Replacement level | The projected score at the position-specific replacement demand implied by team count, starters, FLEX allocation, and ADP-allocated bench demand. |
| VOR | Projected points minus that player's position replacement level. |
| Scarcity | Local projected-point drop around starter and replacement demand for the position. |
| ADP | Current ESPN market average draft position. Missing ADP fails closed. |
| Availability to next pick | The model probability that the player remains available at the explicit next snake pick, using ADP dispersion. It is null until a draft slot is entered. |
| `draft_now` | The next-pick availability is below the configured action threshold. |
| `can_wait` | The next-pick availability is at or above the configured action threshold. |
| `fallback` | A lower-priority actionable option returned by a future UI lane; it is never a missing-data default. |
| Draft slot required | Slot-neutral valuation is available, but timing cannot be estimated without a runtime slot. |
| Taken / Mine | Explicit draft-state status keyed by stable ESPN ID. |

## Interpretation Boundaries

VOR, scarcity, and availability are descriptive and decision-support metrics.
They do not establish causal effects, calibration, or guaranteed draft value.
ADP is a market signal rather than an outcome label. A blocked or unavailable
metric means it was not estimable from valid inputs; it does not mean a measured
zero relationship. The phrase `does not mean a measured zero relationship`
is the intended interpretation of unavailable metrics.

## Related dashboard language

The current page also exposes the concepts **Board value score**, **Simple VOR proxy**,
**Expected regret**, **Fragility score**, **Upside score**, **Decision evidence**,
**Freshness and provenance**, and **Projection breakdown** only when
the corresponding payload fields exist. The current 2026 service does not invent
those fields or neutral defaults when they are absent.
