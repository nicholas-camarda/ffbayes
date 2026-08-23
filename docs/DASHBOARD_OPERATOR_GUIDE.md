# Dashboard Operator Guide

Audience: the person running the 2026 fantasy-football draft.

Scope: the one local command, the two league profiles, draft-day controls,
blocked states, and snapshots.

Trust boundary: the loopback service owns one fresh, validated input snapshot;
the browser displays the Python response and never supplies valuation defaults.

## What This Is

The supported operator workflow is:

```bash
ffbayes dashboard --year 2026
```

It fetches only public ESPN and nflverse data, validates the current-player
universe, projections, ADP, replacement depth, league rules, and provenance,
then opens a local dashboard. No account or private-league connection is used.

## When To Use It

Use this guide when you need to run the dashboard before or during the draft.
Draft slot is entered at runtime because it is not known in advance. The same
command is the only supported way to create the current 2026 board.

## What To Inspect

Select the appropriate independent profile:

| Profile | League identity | Roster |
| --- | --- | --- |
| `league-1-2026` | Bill's Underbit… — Med School Friends | 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 D/ST, 1 K, 6 bench, 1 IR |
| `league-2-2026` | Nicholas's Nif… — Camarda-Klein Family | 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 D/ST, 1 K, 7 bench, 1 IR |

Stable scoring, team count, snake format, FLEX eligibility, and waiver fields
are displayed read-only. Enter draft slot (1 through team count) and current
overall pick explicitly. Until a slot is entered the board may show valuation,
but availability and timing actions are `Draft slot required`.

## Interpretation Boundaries

- `draft_now`, `fallback`, and `can_wait` are decision aids, not guarantees.
- VOR and scarcity are descriptive model outputs, not causal claims.
- ADP is a market reference; it is not substituted for missing projections.
- A blocked state means the board is not certified and must not be used.

## Commands And Paths

The normal command surface has one entrypoint. The service prints a loopback URL
and writes run-scoped material under `runtime/runs/dashboard_2026/<timestamp>/`.
Snapshots are written under that run's `snapshots/` directory. No `site/`,
GitHub Pages, cloud mirror, or older staged HTML is consulted by this workflow.

Lower-level collection, model, publishing, and retrospective modules remain
developer/maintenance code for tests and research. They are intentionally not
documented as alternate ways to produce the draft board.

## Before The Draft

1. Run `ffbayes dashboard --year 2026`.
2. Confirm the status says the source snapshot and coverage validation passed.
3. Select the league and verify its read-only roster shape.
4. Enter your draft slot and current overall pick.
5. Check the freshness/provenance panel and export a snapshot if desired.

## During The Draft

- Update current overall pick after every selection.
- Mark another team's selection as **Taken**; mark your selection as **Mine**.
- Add candidates to **Queue**. The complete state is sent to the service for
  each recalculation.
- Switching leagues restores the other profile's independent slot, pick, and
  player state.

Rows show projected points, replacement level, VOR, scarcity, ADP,
availability at the next snake pick, and the Python-generated action. Stable
`espn_id` values identify all state actions.

## Blocked States

The dashboard remains open with an exact message when ESPN or nflverse is
unavailable, truncated, stale, wrong-season, or semantically inadequate; when
league configuration is invalid; when a slot/current pick is impossible; or
when a provenance digest is inconsistent. It never falls back to an older
runtime output or a neutral/default market value.

## Snapshot Contents

A snapshot is a local, read-only JSON artifact containing effective profile
settings, runtime draft state, board rows, source manifests and SHA-256 values,
profile/runtime/state/board digests, code revision, and generation time. The
service writes it atomically only after the board passes provenance validation.

## Historical and Developer Material

The repository contains historical analysis and frontend research notes. Those
documents are not current operator instructions. The public GitHub `master`
branch is external state and is not changed by an unpushed worktree.
