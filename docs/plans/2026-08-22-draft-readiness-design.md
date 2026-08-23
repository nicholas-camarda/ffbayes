# 2026 draft-readiness design

## Objective

Generate a separate, fail-closed draft board for each configured 2026 league. A board is usable only when its current player universe, projections, market data, league settings, valuation inputs, and output provenance all pass semantic validation.

## Source contract

- ESPN's 2026 fantasy player feed supplies the current fantasy catalog, raw season projections, ESPN ownership/ADP, and market ranks. The ingestion records the request URL, response timestamp, season, row counts, coverage, and SHA-256 digest.
- The current NFL roster feed supplies an independent eligibility cross-check for offensive players. A player must be current in both sources; a historical record alone can never create a board row. Team defenses are represented by current ESPN D/ST entities.
- Every network source is fetched into a run-scoped directory. Reuse is allowed only when a manifest proves the same source, season, league profile, digest, and an acceptable age. No glob-based "latest" lookup is part of the canonical board path.
- A source failure, truncated response, schema change, stale season, or inadequate coverage stops the affected board. There is no fabricated, neutral, or historical substitute.

## League contract

Each league uses a required named profile containing league ID, team count, draft format and slot, scoring rules, starter slots, bench, IR, flex eligibility, and relevant acquisition constraints. Profiles are model inputs, not labels: they determine projected points, replacement demand, scarcity, snake-pick timing, and recommendations.

The two requested profiles have distinct FLEX and bench structures. Unknown league settings remain explicit unresolved local fields and block certification; defaults do not fill them.

## Valuation and recommendation contract

- League-specific projected points are computed from raw ESPN projected statistics and the configured scoring rules.
- Replacement levels are calculated from the validated current projection pool using league-size and roster/FLEX demand. Every required position must have adequate depth beyond its replacement index.
- VOR is projected points above the position replacement baseline. Scarcity is derived from the slope and remaining supply near league demand, not raw quarterback point totals.
- Market data is independent of model value. Missing or non-informative ADP cannot be replaced by model rank. Availability is estimated from empirical ESPN ADP dispersion and is unavailable when a player lacks usable market data.
- The recommendation policy balances VOR, scarcity, roster need, market opportunity cost, and the user's actual next snake pick. One-QB demand and specialist timing are enforced through league configuration.

## Fail-closed invariants

Production validation rejects:

- players absent from the current eligible universe;
- inadequate positional, projection, or ADP coverage;
- missing or saturated market values in the decision-relevant range;
- missing required league fields or invalid roster/scoring definitions;
- invalid or non-finite replacement levels;
- source seasons or timestamps inconsistent with the requested board;
- quality reports whose score contradicts blockers or severe missingness;
- dashboard payloads whose inputs, code revision, league profile, or digests differ from the validated run manifest.

Tests mutate each boundary to prove rejection. Sensitivity tests prove that FLEX count, league size, scoring, and draft slot alter model outputs where expected.

## Outputs

Each league gets its own run directory containing raw source snapshots, manifests, validation report, board CSV/JSON, dashboard payload/HTML, and a certification report. No site publishing, merge, push, or deployment is part of this work.
