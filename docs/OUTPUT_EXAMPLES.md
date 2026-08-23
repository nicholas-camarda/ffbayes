# 2026 Dashboard Output Shape

Audience: reviewers checking a local snapshot.

Scope: the current payload returned by `/api/board` and written by snapshot export.

Trust boundary: the example is illustrative; a real payload must pass
`validate_output_provenance` with current source manifests.

## What This Is

The supported command is `ffbayes dashboard --year 2026`. It does not create a
public Pages artifact. A snapshot is written only after a valid board response.

## Example payload

```json
{
  "schema_version": "draft_2026_v1",
  "season": 2026,
  "generated_at": "2026-08-23T15:00:00+00:00",
  "league_profile": {
    "profile_id": "league-1-2026",
    "league_name": "Bill's Underbit…",
    "team_count": 10,
    "draft_format": "snake",
    "draft_slot": 4,
    "scoring_label": "Half PPR",
    "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "DST": 1, "K": 1},
    "bench_slots": 6,
    "ir_slots": 1
  },
  "current_pick": 4,
  "next_pick": 17,
  "runtime_state": {
    "draft_slot": 4,
    "current_pick": 4,
    "taken_ids": [12345],
    "your_ids": [23456],
    "queue_ids": [34567]
  },
  "decision_table": [
    {
      "board_rank": 1,
      "espn_id": 34567,
      "name": "Example Player",
      "position": "WR",
      "projected_points": 245.1,
      "replacement_level": 132.4,
      "vor": 112.7,
      "scarcity": 8.2,
      "adp": 9.4,
      "availability_next_pick": 0.31,
      "recommendation": "draft_now",
      "roster_status": "available",
      "is_available": true
    }
  ],
  "provenance": {
    "code_revision": "<git revision>",
    "profile_sha256": "<digest>",
    "state_sha256": "<digest>",
    "board_sha256": "<digest>",
    "source_manifests": ["<ESPN manifest>", "<nflverse manifest>"]
  }
}
```

## What To Inspect

Confirm the season, profile ID, current pick, stable `espn_id`, coverage report,
source timestamps, and all provenance digests. A slot-neutral payload has null
availability and `slot_required` recommendations; it must not contain a guessed
50% availability.

## Interpretation Boundaries

The payload is a validated decision aid, not a guarantee. Historical or staged
HTML files are not current inputs, and missing/unavailable metrics are not
silently converted to zero.
