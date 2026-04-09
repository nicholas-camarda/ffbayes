## Context

The live draft dashboard is already the canonical visualization product for this repo, and it already emits enough structure to support better draft-time visuals: `recommendation_summary` contains timing and regret data, `tier_cliffs` contains scarcity breakpoints, and `decision_evidence` contains contextual-versus-baseline disagreement rows. The problem is not lack of data. The problem is that the current war room still makes users translate those concepts manually from cards, tables, and text while the draft room is moving.

At the same time, the dashboard should not become a bloated analytics playground. The useful visual additions are the ones that answer immediate draft-time questions and remain valid even if the underlying model becomes more sophisticated than today’s `draft_score` plus simple VOR proxy. This design therefore treats the visual layer as a decision-support interface built on stable semantic contracts rather than as a direct rendering of current formula internals.

## Goals / Non-Goals

**Goals:**
- Add interactive war-room visuals that directly improve draft-time decisions.
- Start with timing risk, positional scarcity, and contextual-versus-baseline comparison because those are already latent in the payload and most useful in the war room.
- Keep the visuals integrated into the existing dashboard flow through progressive disclosure rather than creating a separate plot gallery or analytics tab.
- Introduce a normalized visualization payload contract so future model upgrades can feed the same visual components without breaking them.
- Preserve existing `pre_draft`, `draft_strategy`, `refresh-dashboard`, and `publish-pages` artifact names and staging behavior.

**Non-Goals:**
- Do not add standalone static PNG dashboards or revive removed legacy visualization modules.
- Do not redesign the Pages loading model or create a separate visualization artifact family outside the current dashboard payload and HTML.
- Do not hardcode the visual layer to the current `draft_score` weighting formula or assume the simple baseline will always be VOR.
- Do not try to visualize every available metric in v1; the emphasis is utility over completeness.

## Decisions

1. **Add a normalized `war_room_visuals` payload layer.**
   The dashboard should not bind new visuals directly to raw implementation fields such as `replacement_delta`, `expected_regret`, or current component weights. Instead, payload generation should produce a normalized visualization section with semantic concepts such as contextual score, baseline score, timing survival, wait regret, and positional cliff strength, plus user-facing labels for the current baseline.

   Alternatives considered:
   - Read existing raw payload fields directly from the frontend. Rejected because it couples the UI to today’s model internals and makes future model upgrades brittle.
   - Recompute everything client-side from the raw decision table. Rejected because the staged payload does not currently carry all live timing fields board-wide and because semantic drift between Python and JS would be harder to control.

2. **Ship the visuals as one primary pair plus one secondary explainer.**
   The primary additions should be the `wait-vs-pick` timing ladder and positional tier-break strips because they answer immediate draft-time questions without forcing the operator to decode overlapping marks or annotation-heavy chips. The contextual-versus-baseline comparison should appear as a secondary explainer tied to the selected player / evidence flow, not as a third large always-open chart.

   Alternatives considered:
   - Ship all three as equally prominent panels. Rejected because that risks crowding the war room and weakening the recommendation lanes.
   - Delay the comparative explainer entirely. Rejected because disagreement interpretation is one of the dashboard’s strongest trust-building opportunities.

3. **Use progressive disclosure inside the existing war-room layout.**
   The timing ladder belongs near the `pick now` surface, the positional cliff view belongs directly above the player board as a collapsed-by-default panel with a one-line summary, and the comparative explainer should live first inside the selected-player inspector rather than as a persistent standalone panel. Queue and roster state should stay in the right column near the selected-player context rather than being buried below reference material. This keeps the visuals attached to the decisions they support while limiting bloat in the default dashboard state.

   Alternatives considered:
   - Add a dedicated “visualizations” tab. Rejected because it separates the plots from the live drafting action and invites low-value add-ons.
   - Hide everything behind the evidence panel. Rejected because timing and scarcity are first-order draft decisions, not just explanatory afterthoughts.

4. **Make the visuals state-aware and synchronized with local draft actions.**
   The frontier, cliff map, and comparative explainer should update when the local board state changes through `taken`, `mine`, queue, current pick, next pick, scoring preset, or selected player changes. They should behave like living war-room controls, not static reports.

   Alternatives considered:
   - Render visuals from the initial exported payload only. Rejected because the dashboard’s local utility depends on reacting to draft-room state changes.

5. **Preserve the current artifact lifecycle.**
   The new visuals should flow through the existing dashboard payload, runtime HTML, repo-local shortcut, and `site/` Pages staging path. `refresh-dashboard` should continue to regenerate the HTML from payload alone, and `publish-pages` should stage the enhanced dashboard without special-case handling.

   Alternatives considered:
   - Emit separate visualization JSON or image artifacts. Rejected because it would complicate lifecycle synchronization and revive the artifact-sprawl problem that the visualization cleanup just removed.

6. **Render positional scarcity as tier-break strips, defaulted to recommendation-relevant positions.**
   The positional scarcity view should present each relevant position as a compact ordered strip with one clearly emphasized strongest break by default, rather than as a chip cloud with inline text annotations. The view should live in a collapsed-by-default details panel immediately above the player board, with a concise summary visible when collapsed. It should still default to the positions most relevant to the active recommendation lanes or selected-player context, with an explicit way to inspect all positions. That keeps scarcity visible without turning the war room into a dense league-wide board atlas by default.

   Alternatives considered:
   - Show all positions by default. Rejected because it adds too much visual noise in the main draft flow and weakens the connection between the chart and the current decision.
   - Annotated chip clusters. Rejected because they bury the actual cliff signal inside text fragments and make the break harder to read at draft speed.

7. **Evolve the visualization contract additively.**
   The normalized `war_room_visuals` payload should be treated as a semantic compatibility layer. New model outputs may extend it, rename user-facing baseline labels, or improve the underlying calculations, but they should not require the visual components to bind directly to volatile formula fields or to drop existing stable keys without an explicit migration.

   Alternatives considered:
   - Let the frontend read whatever raw model fields happen to exist. Rejected because it makes both model iteration and dashboard evolution fragile.
   - Freeze the current model-specific fields as the long-term UI contract. Rejected because it would make future model upgrades artificially expensive.

## Risks / Trade-offs

- [War-room clutter] -> New visuals could crowd the current dashboard. Mitigation: progressive disclosure, compact default sizing, and a rule that each visual must answer one clear draft-time question.
- [Semantic contract drift] -> The normalized visualization layer could fall out of sync with the raw recommendation logic. Mitigation: derive the visualization contract in one place during payload generation and add tests that compare semantic outputs against known recommendation inputs.
- [Future model mismatch] -> A later model may not use VOR as the right baseline language. Mitigation: keep the contract generic around `contextual` versus `baseline` labels and values, with the current simple VOR proxy treated as the present baseline rather than a permanent assumption.
- [State recomputation complexity] -> Live updates may require recomputing timing or selection-derived visuals in-browser. Mitigation: keep the Python-emitted payload authoritative for semantic structure and limit client recomputation to local state filtering and selection updates.
- [Pages density] -> The staged `site/` version could become too heavy or visually noisy for read-only viewers. Mitigation: prioritize compact summaries in default view and preserve collapsible details for heavier explanations.

## Migration Plan

1. Extend payload generation with a normalized `war_room_visuals` section while preserving existing fields during the transition.
2. Add the new visual components to the dashboard HTML/JS and wire them into existing local state changes.
3. Keep current tables and inspector content temporarily where needed, then trim redundant text once the visuals are proven.
4. Verify `refresh-dashboard`, repo-local shortcut staging, and `publish-pages` still regenerate and stage the enhanced dashboard without special handling.
5. Preserve backward compatibility by evolving the visualization payload additively and verifying the staged/local dashboard surfaces still render when optional new sections are absent.
