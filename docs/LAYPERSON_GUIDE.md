# Plain-language guide

FFBayes is a draft-day assistant. It combines current public projections, your
league's scoring and roster rules, and market ADP to answer two practical
questions: who is valuable for this league, and who is likely to still be there
at your next pick?

## Use it

```bash
ffbayes dashboard --year 2026
```

Select your local league profile, enter your draft slot and current overall
pick, then update Taken/Mine/Queue as the draft moves. If the slot is unknown,
the board still shows league-adjusted value but waits to make a next-pick
recommendation.

## What the numbers mean

- **Projected points**: the player's expected stat line converted with your
  scoring settings.
- **Replacement level**: the score of the last realistic player needed at that
  position after starters, FLEX slots, and bench demand are counted.
- **VOR**: projected points above that replacement score.
- **Scarcity**: how quickly projected value drops around the roster demand.
- **ADP**: where the current market is drafting the player.
- **Availability**: the modeled chance the player survives to your next pick.
- **Recommendation**: `draft_now`, `can_wait`, `slot_required`, `taken`, or
  `mine`.

## Keep the right expectations

The board is a structured way to make a pick under uncertainty, not a promise
about the season. A high rank is not a causal claim that drafting a player
guarantees a win. A blocked metric means the required input was not reliable
enough to calculate it; it is not a hidden zero.

For equations and validation details, see the [metric reference](METRIC_REFERENCE.md)
and [technical deep dive](TECHNICAL_DEEP_DIVE.md).
