## Why

The live draft dashboard already has strong underlying decision data, but the current UI still relies heavily on tables, pills, and prose to communicate timing risk, positional scarcity, and contextual-versus-baseline disagreement. That makes the board harder to trust and slower to use during a live draft than it needs to be, especially now that the visualization surface has been cleaned up and the dashboard is the clear product surface.

## What Changes

- Add a normalized war-room visualization contract to the dashboard payload so decision visuals depend on stable semantic fields rather than the current draft-score formula internals.
- Add a live `wait-vs-pick` timing frontier to show which candidates are worth taking now versus safely waiting on.
- Add a positional cliff map to show where each position group drops off sharply and which candidates sit just before those cliffs.
- Add an inspector-first progressive-disclosure comparative explainer for contextual-versus-baseline disagreements instead of creating a separate analytics surface.
- Keep the new visuals interactive and tied to live board state so they react to `taken`, `mine`, queue, current pick, next pick, and selected-player changes.
- Keep the visualization payload additive and backward-compatible so `refresh-dashboard`, `publish-pages`, staged `site/`, and future model upgrades can evolve without breaking the supported dashboard surface.
- Preserve the existing runtime artifact names and `site/` Pages staging contract; the change extends the dashboard payload and UI rather than introducing a separate visualization artifact family.
- Update docs and dashboard tests so the new visuals are described as draft-decision aids, not decorative analytics.

## Capabilities

### New Capabilities
- `war-room-decision-visualizations`: define the normalized payload contract and interactive dashboard behavior for timing, scarcity, and comparative decision visuals inside the live draft war room.

### Modified Capabilities
- `visualization-surface-governance`: extend visualization governance so new dashboard visuals must be decision-first, progressively disclosed, and resilient to future model changes.

## Impact

- Affects dashboard payload generation and UI rendering primarily in `src/ffbayes/draft_strategy/draft_decision_system.py`.
- Affects staged and local dashboard behavior through the existing `dashboard_payload.json`, runtime dashboard HTML, repo-local `dashboard/`, and staged `site/` surfaces, without changing the current `pre_draft` artifact names.
- Affects dashboard smoke and pytest coverage for payload semantics, interactive rendering, and `publish-pages` / `refresh-dashboard` compatibility.
- Affects README and docs text describing how to use the war room dashboard and what the new visuals mean.
