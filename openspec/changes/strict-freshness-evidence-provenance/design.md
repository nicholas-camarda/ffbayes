## Context

FFBayes already centralizes its runtime paths, pipeline entrypoints, and published dashboard staging, but trust-related behavior is still split across configuration, runtime JSON, and UI rendering. The current pre-draft pipeline allows stale seasons in multiple steps, the backtest is generated as a historical artifact but reduced to a compact dashboard summary, and `publish-pages` stages `site/` output without a first-class provenance manifest.

This change crosses the pre-draft pipeline, runtime artifacts, Pages staging, and dashboard rendering. It therefore needs explicit decisions about where freshness is enforced, how evidence is normalized into the dashboard payload, and how provenance moves from runtime outputs into the repo-tracked `site/` copy.

Stakeholders are the local draft operator who needs reliable recommendations during a live draft, and the repo/Page viewer who needs to understand how current and trustworthy the staged dashboard is.

## Goals / Non-Goals

**Goals:**
- Make stale-season tolerance explicit and visible instead of silent.
- Reuse the existing `ffbayes` CLI, runtime output layout, and `site/` staging flow.
- Promote the draft backtest from a side artifact into a decision-evidence panel that explains evidence, limitations, and decision relevance.
- Attach publish-time provenance to staged Pages outputs and payloads without changing the canonical `dashboard_payload.json` / `index.html` filenames.
- Preserve the supported `pre_draft` phase and current artifact roots under `runs/<year>/pre_draft/...`.

**Non-Goals:**
- Building live draft ingestion or provider sync.
- Changing the underlying modeling approach or replacing the current backtest methodology.
- Renaming current artifact families such as `draft_strategy`, `dashboard_payload.json`, or `site/index.html`.
- Introducing a new storage root or bypassing `path_constants.py`.

## Decisions

### 1. Freshness becomes fail-closed by default, with explicit override and surfaced state

The pipeline and backtest loaders will treat missing latest expected seasons as blocking by default. Any continued execution on degraded data MUST require an explicit override path that is surfaced in runtime artifacts, CLI output, and Pages provenance.

Why this over the status quo:
- It matches the agreed product direction that silent staleness is dangerous.
- It preserves emergency escape hatches for intentional degraded runs without letting them masquerade as fresh outputs.

Alternatives considered:
- Keep silent stale allowance and improve docs only. Rejected because it leaves behavior unchanged at the exact failure boundary that matters.
- Remove all override capability. Rejected because local analysis and offseason workflows may still need intentional degraded runs.

### 2. Decision evidence is computed once in Python and rendered as structured payload sections

The source of truth for decision evidence will remain Python-side in the draft strategy/backtest flow. The dashboard will consume normalized payload sections rather than deriving evidence in browser code from raw backtest tables.

The payload should include:
- evidence status and evaluation scope
- freshness state used for the backtest
- strategy comparison summary
- season-level rows
- stated limitations and interpretation guidance
- failure/degraded reasons when evidence is unavailable or partial

Why this over more client-side logic:
- Keeps business rules near the backtest outputs and existing artifact generation.
- Makes workbook/export/Pages surfaces share the same evidence contract.

Alternatives considered:
- Let `site/index.html` derive the panel from raw JSON tables. Rejected because it duplicates interpretation logic in JS and weakens testability.

### 3. Publish provenance is staged alongside `site/` artifacts and embedded into the dashboard payload

`publish-pages` will stage both the dashboard files and a provenance artifact derived from runtime metadata at publish time. The dashboard payload should also expose the key provenance fields needed by the UI so the site can render them without additional fetch complexity.

The provenance contract should include:
- generation timestamp
- publish timestamp
- source artifact paths or names
- analysis window and freshness status
- whether execution used an explicit stale override
- version or schema marker for staged provenance

Why this over a repo-only note:
- The published site is the public artifact, so provenance must travel with it.
- Embedding a compact form into `dashboard_payload.json` avoids a brittle split between UI-visible and filesystem-only metadata.

Alternatives considered:
- Store provenance only in runtime logs. Rejected because Pages viewers would remain blind to artifact lineage.

### 4. Migration favors compatibility of filenames and paths, not compatibility of silent behavior

Current filenames, runtime directories, and Pages targets remain stable. The behavioral migration is instead in freshness semantics: commands that used to proceed quietly may now fail or emit degraded-state artifacts unless an explicit override is set.

This preserves path compatibility while making trust-related behavior stricter and more legible.

## Risks / Trade-offs

- [Stricter freshness blocking may surprise existing workflows] → Add clear CLI error text, explicit override wording, README updates, and provenance flags that show when a degraded run was intentional.
- [Evidence panel may overstate confidence if backtest limitations are not prominent] → Include mandatory limitations text and evaluation-scope labels in the payload and rendered UI.
- [Multiple artifact surfaces can drift] → Generate evidence and provenance from Python-side source data, then reuse those sections in workbook, runtime JSON, and Pages staging.
- [Pages staging may fail when runtime provenance inputs are missing] → Keep failure modes explicit; stage a degraded provenance record only when the source artifact exists and clearly mark missing fields.

## Migration Plan

1. Define the stricter freshness contract in helpers and pipeline/backtest callers.
2. Update runtime artifact generation so freshness state and override status are serialized into draft outputs.
3. Expand draft dashboard payload generation with structured decision-evidence and provenance sections.
4. Update `publish-pages` to stage provenance metadata with `site/` assets.
5. Refresh dashboard rendering, tests, and docs to reflect new trust semantics.

Rollback strategy:
- Revert to the prior freshness gating and staging behavior while preserving path layout and artifact names.
- Because filenames remain stable, rollback risk is primarily behavioral rather than structural.

## Open Questions

- Whether the stale override should remain `FFBAYES_ALLOW_STALE_SEASON` with stricter semantics or move to a clearer opt-in name while still honoring the old variable temporarily.
- Whether publish provenance needs its own `site/` JSON file in addition to payload embedding, or whether embedded payload metadata plus runtime manifests is sufficient.
- How much season-level evidence detail should be shown directly in the main panel versus tucked behind expandable sections in the dashboard UI.
