# First modeling implementation batch

Completed the first practical slice of the 2026-09-06 roadmap in the canonical source working tree. Luna xhigh authored the initial implementation; its run stopped on a workspace-credit error, and the parent agent completed review fixes, integration, documentation, and validation. No commit or deployment was performed.

Implemented:

- Exact configurable lineup assignments and whole-league completion checks, including overlapping FLEX/superflex and optional K/DST.
- Roster-aware marginal replacement-adjusted utility after the first owned selection. The existing VOR/ADP rank breaks ties; the initial empty-roster board retains its static ranking. The 5% bench term is explicitly heuristic.
- Separate draftability and ownership; no recommendation when the roster is full. Picked and queued rows stay visible after ranking changes and reloads.
- Independent static ranking baseline, provider-point and ADP policies, a roster utility ablation, and the integrated live policy, all using the same legal-completion wrapper against fixed opponents.
- Reproducible JSON evaluation CLI with explicit synthetic mode, declared as-of timestamp validation, scoring consistency checks, source/input/code digests, and complete pick histories. Outputs score projected utility only.
- Legacy profile serialization and checkpoint digest compatibility; an old-schema checkpoint with the previously captured public-profile digest restores picks, queue, slot, and clock.

Verification:

- Full suite: 109 tests passed.
- After the final UI evidence addition: all 39 affected pipeline/service/browser tests passed again.
- Hostile browser rendering passed; lint and mypy passed; git diff --check passed.
- Static baseline matched the original HEAD ranking on the conventional fixture.
- Exact solver tested against exhaustive overlapping-slot assignments.
- A 290-player board with 14 owned players rebuilt in approximately 0.5 seconds after optimization (one local timing, not a latency guarantee).
- Documented synthetic CLI executed successfully; report: runtime/evaluation/synthetic.result.json.

Boundaries:

This is still a heuristic decision backend. It does not fit a Bayesian forecast, learn opponent behavior, assemble historical player-week outcomes, perform rolling model fitting, or demonstrate an autodraft/championship advantage. Per-position roster caps, auction/dynasty policies, and real weekly outcome evaluation remain future work. Valid historical preseason projection/ADP snapshots have not been established by the raw-file inventory. The next implementation phase should acquire valid as-of inputs and build calibrated probabilistic forecast challengers on this evaluation foundation.

Usage and input contracts: source/docs/EVALUATION_BASELINES.md.
Metric definitions and configuration: source/docs/METRIC_REFERENCE.md.
