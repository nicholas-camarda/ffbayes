# 2026 dashboard output shape

The local dashboard returns this payload from `POST /api/board` and writes the
same validated shape when a snapshot is exported. Values below are illustrative
only.

## Example payload

```json
{
  "schema_version": "draft_2026_v1",
  "season": 2026,
  "generated_at": "2026-08-23T15:00:00+00:00",
  "league_profile": {
    "profile_id": "example-2026",
    "league_name": "Example League",
    "team_count": 12,
    "draft_format": "snake",
    "draft_slot": 4,
    "scoring_label": "Half PPR",
    "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1},
    "bench_slots": 6,
    "ir_slots": 1,
    "flex_eligible": ["RB", "WR", "TE"]
  },
  "current_pick": 4,
  "next_pick": 21,
  "runtime_state": {
    "draft_slot": 4,
    "current_pick": 4,
    "taken_ids": [12345],
    "your_ids": [23456],
    "queue_ids": [34567]
  },
  "replacement": {
    "levels": {"QB": 165.2, "RB": 128.4, "WR": 135.7, "TE": 94.1, "DST": 70.0, "K": 82.0},
    "demand": {"QB": 18, "RB": 48, "WR": 53, "TE": 30, "DST": 12, "K": 12}
  },
  "coverage_report": {"status": "passed"},
  "decision_table": [
    {
      "board_rank": 1,
      "espn_id": 34567,
      "name": "Example Player",
      "position": "WR",
      "projected_points": 245.1,
      "replacement_level": 135.7,
      "vor": 109.4,
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

## What to inspect

Confirm the season, profile settings, current and next pick, stable `espn_id`
values, coverage status, source timestamps, and provenance digests. If no draft
slot has been entered, availability is `null` and recommendations are
`slot_required`; the payload must not contain a guessed neutral probability.

The payload is a validated decision aid generated from the current input
snapshot. Missing metrics are rejected by the service and are not converted to
zeros.
