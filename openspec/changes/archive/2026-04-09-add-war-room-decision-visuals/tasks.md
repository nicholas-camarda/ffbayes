## 1. Define the normalized visualization contract

- [x] 1.1 Add a normalized `war_room_visuals` payload section for timing, scarcity, and comparative explanation semantics.
- [x] 1.2 Map current recommendation, tier-cliff, and disagreement data into model-agnostic fields plus user-facing labels for the current baseline.
- [x] 1.3 Define unavailable or degraded states for each visualization section so the dashboard can explain missing inputs instead of silently omitting visuals.
- [x] 1.4 Keep the visualization contract additive and backward-compatible so existing dashboard lifecycle commands and staged surfaces do not break when the new sections are absent or extended later.

## 2. Implement the interactive war-room visuals

- [x] 2.1 Add a compact `wait-vs-pick` frontier near the recommendation lanes and sync it to current local board state.
- [x] 2.2 Add a positional cliff map that highlights sharp drop-offs by position, defaults to recommendation-relevant positions, and syncs to board filtering and player selection.
- [x] 2.3 Add a comparative contextual-versus-baseline explainer through progressive disclosure in the selected-player inspector instead of as a standalone analytics page.
- [x] 2.4 Ensure the new visuals update correctly for `taken`, `mine`, queue, current pick, next pick, scoring preset, and selected-player changes.

## 3. Harden the dashboard contract and documentation

- [x] 3.1 Add pytest coverage for the normalized visualization payload, its degraded/unavailable states, and backward-compatible payload evolution.
- [x] 3.2 Extend dashboard smoke or integration coverage for rendering and state updates of the new war-room visuals.
- [x] 3.3 Update README and related docs so the new visuals are described as decision aids inside the local war room dashboard.
- [x] 3.4 Verify `refresh-dashboard` and `publish-pages` preserve the enhanced dashboard without introducing new artifact paths or lifecycle drift.
