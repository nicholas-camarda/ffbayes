# Dashboard operator guide

This is the user-facing guide for a draft-day board. The repository is
league-agnostic: the checked-in example profile is a template, and each user
keeps their own league rules in an ignored local profile.

## One command

```bash
ffbayes dashboard --year 2026
```

The command fetches the public 2026 inputs, validates the player universe,
projections, ADP, replacement depth, and league profiles, then opens a local
dashboard. It does not require a fantasy-platform account, password, or
private-league connection.

## First-time setup

1. Copy `config/leagues/example_2026.json` to
   `config/leagues/my-league.local.json`.
2. Edit the copy to match the league: team count, scoring, roster slots,
   FLEX eligibility, bench/IR slots, and snake-draft settings.
3. Leave `draft_slot` as `null` if the draft order is not known yet.
4. Run the command above. Local `*.local.json` profiles are discovered
   automatically. Use `--profile path/to/profile.json` when an explicit profile
   list is preferred.

The `.local.json` suffix is ignored by Git, so private league names and rules
are not part of the public repository.

## Draft-day workflow

1. Select the profile and confirm its read-only roster/scoring summary.
2. Enter your draft slot (1 through the configured team count) and current
   overall pick. The board cannot estimate snake timing until the slot is set.
3. After each pick, update the current overall pick and mark players as
   **Taken** or **Mine**. Add candidates to **Queue** when useful.
4. Use the recommendation, availability, VOR, scarcity, and ADP columns as
   decision aids. Export a snapshot when you want a durable record of the
   validated state.

Each profile has independent draft state. Changing the selected league does not
carry over slot, pick, taken IDs, roster IDs, or queue IDs.

## What the board checks before it opens

- Current-season, active, fantasy-relevant player IDs and positions.
- Sufficient projection depth for every configured position and replacement
  calculation.
- Fresh, complete ADP for the market universe; missing ADP is an error, not a
  50% availability guess.
- Explicit league settings and valid snake-draft bounds.
- Output/provenance digests that connect the board to the current inputs,
  profile, runtime state, and code revision.

If a required source or check fails, the page shows a blocked state and does
not present a board from an older run.

## Run artifacts

Run-scoped material is written under
`runtime/runs/dashboard_2026/<timestamp>/`. Exported snapshots are under that
run's `snapshots/` directory and contain the profile, runtime state, board,
coverage report, source manifests, timestamps, and SHA-256 digests.

## Related documentation

- [Metric reference](METRIC_REFERENCE.md) — definitions and equations.
- [Data lineage and paths](DATA_LINEAGE_AND_PATHS.md) — sources and provenance.
- [Output examples](OUTPUT_EXAMPLES.md) — payload fields and snapshot shape.
- [Technical deep dive](TECHNICAL_DEEP_DIVE.md) — implemented calculation flow.
