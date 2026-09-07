# Fair baseline evaluation

The first modeling batch includes a small, deterministic draft harness at
`ffbayes.evaluation.baselines`. It compares the frozen static VOR/ADP v1
FFBayes rank (`current_ffbayes`), a raw provider projection rank
(`provider_projection`), ADP (`adp`), the integrated live FFBayes board
(`ffbayes_live`), and a pure roster utility ablation (`roster_aware_vor`).

Every policy receives the same player snapshot, league profile, snake order,
seed, and legal-pick wrapper. For a focal team (configured slot by default, otherwise slot 1), the candidate
policy drafts against the same fixed opponent policy (ADP by default), and
the output reports focal and opponent projected starter points. This is not realized performance or weekly lineup evaluation. A global reservation
check prevents one team from consuming the last player needed to complete
another team's roster. Final rosters are validated against every configured
slot group, including overlapping FLEX and SUPERFLEX groups.

The input file is JSON with this shape:

```json
{
  "cutoff": "2026-08-22T12:00:00Z",
  "profile": { "...": "complete league profile" },
  "source_manifests": [
    {
      "source": "provider snapshot",
      "season": 2026,
      "as_of": "2026-08-22T11:55:00Z",
      "published_at": "2026-08-22T11:55:00Z",
      "fetched_at": "2026-08-22T11:56:00Z",
      "sha256": "64-character lowercase SHA-256 of the original source snapshot",
      "schema_version": "provider-v1",
      "row_count": 400
    }
  ],
  "players": [
    {
      "espn_id": 101,
      "projection_season": 2026,
      "position": "WR",
      "projected_points": 220.0,
      "adp": 12.0,
      "projection_as_of": "2026-08-22T11:55:00Z",
      "adp_as_of": "2026-08-22T11:55:00Z",
      "projection_stats": { "...": "raw stats for current_ffbayes" }
    }
  ]
}
```

Run a reproducible focal-team comparison with:

```bash
PYTHONPATH=src python -m ffbayes.evaluation.baselines \
  --input ../runtime/evaluation/example.json \
  --output ../runtime/evaluation/example.result.json \
  --seed 7 --focal-team 3 --opponent-policy adp
```

The harness rejects duplicate, fractional, boolean, or missing player IDs,
invalid projections or ADP, future-dated feature rows, mismatched source
seasons, and source manifests without explicit timestamps and hashes. When raw
stats are supplied, their profile-scored points must match `projected_points`.
As-of mode requires a cutoff, source manifests, and per-row `projection_as_of`
and `adp_as_of`. This validates declared timestamps; it does not independently
prove a provider's historical publication time or authenticate a source hash.

For an explicitly synthetic mechanics check, run the included example from the
source directory:

```bash
PYTHONPATH=src python -m ffbayes.evaluation.baselines \
  --input examples/evaluation_synthetic.json \
  --output ../runtime/evaluation/synthetic.result.json --synthetic
```

Synthetic reports carry `synthetic_unverified` and cannot claim as-of provenance.
All reports contain policy settings, focal team, code and input digests, and full
pick histories. Team indices are zero-based. Ties use stable player IDs;
`--seed` is reserved metadata, and repeated seeds do not create independent
trials in this deterministic harness. Do not interpret projected utility gains
as a demonstrated advantage over platform autodraft or as championship odds.

`current_ffbayes` is a compatibility name for the independent static VOR/ADP
rank in `evaluation/static_v1.py`. Its formula matches conventional legacy
profiles and extends replacement demand to the new slot configuration; it is
not an exact replay of the old engine for previously unsupported formats.
All strategies receive the same league-completion constraint wrapper, which
is a benchmark safeguard and not a claim to emulate a platform's autodraft.

The current raw historical files do not establish valid preseason projection
and market snapshots. This batch supplies temporal contracts and a reproducible
policy harness; player-week outcome assembly, rolling model fitting, and real
outcome evaluation still need those data and are not implemented here.
