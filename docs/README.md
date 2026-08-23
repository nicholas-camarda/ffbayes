# FFBayes Documentation

The current 2026 operator workflow is one command:

```bash
ffbayes dashboard --year 2026
```

Start with [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md). It
describes both named leagues, runtime draft-slot entry, draft-state controls,
blocked states, and local snapshots.

## Guide map

- [DASHBOARD_OPERATOR_GUIDE.md](DASHBOARD_OPERATOR_GUIDE.md): current draft-day use.
- [DATA_LINEAGE_AND_PATHS.md](DATA_LINEAGE_AND_PATHS.md): fresh inputs, run roots, and provenance.
- [METRIC_REFERENCE.md](METRIC_REFERENCE.md): board labels and interpretation boundaries.
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md): current payload and snapshot shape.
- [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md): model and research notes; the current operator path is the dashboard command above.
- [DASHBOARD_FRONTEND_ARCHITECTURE.md](DASHBOARD_FRONTEND_ARCHITECTURE.md): current service/page boundary.
- [DASHBOARD_FRONTEND_CUTOVER.md](DASHBOARD_FRONTEND_CUTOVER.md): historical frontend notes, explicitly non-operator.
- [LAYPERSON_GUIDE.md](LAYPERSON_GUIDE.md): plain-language interpretation.

## Trust Model

The local service is authoritative for a run. Required public sources use fresh
cache-off fetches; stale outputs and staged pages are not inputs. A board is
usable only after current-player, projection, ADP, replacement, league, and
provenance checks pass. No account, session, or private-league source is used.

The public GitHub branch is external state. Local documentation in an unpushed
worktree is not a claim that GitHub has already been updated.

## Current path

Run the command, select a league, enter the draft slot/current pick, operate the
state controls, and export a snapshot. The profile JSON files contain stable
league rules; runtime slot/state never mutates them.
