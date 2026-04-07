## 1. Freshness Enforcement

- [x] 1.1 Update freshness helpers and affected callers so missing latest expected seasons fail closed by default for pre-draft and backtest flows.
- [x] 1.2 Define the explicit degraded-data override behavior and serialize freshness status, missing years, and override usage into runtime metadata artifacts.
- [x] 1.3 Update pipeline and backtest command messaging so blocked versus degraded freshness states are clearly reported to operators.

## 2. Decision Evidence Payload

- [x] 2.1 Refactor draft backtest and draft strategy artifact generation to produce a normalized decision-evidence structure with evaluation scope, season rows, limitations, and degraded/unavailable states.
- [x] 2.2 Extend dashboard payload generation and workbook-facing summaries to consume the normalized decision-evidence structure without changing canonical artifact names.
- [x] 2.3 Update dashboard rendering so the evidence panel presents comparative results together with limitations and degraded-state messaging.

## 3. Publish-Time Provenance

- [x] 3.1 Define publish-time provenance fields for staged dashboard artifacts, including generation time, publish time, source lineage, freshness state, and override usage.
- [x] 3.2 Update `ffbayes publish-pages` staging so `site/` receives the required provenance metadata while preserving `site/index.html` and `site/dashboard_payload.json`.
- [x] 3.3 Render staged provenance in the dashboard UI so Pages viewers can inspect artifact lineage without relying on runtime-only files.

## 4. Verification And Documentation

- [x] 4.1 Add or update pytest coverage for freshness enforcement, degraded override handling, decision-evidence payload structure, and publish-pages provenance staging.
- [x] 4.2 Extend dashboard smoke coverage to assert evidence-panel and provenance visibility for local and staged Pages contexts where applicable.
- [x] 4.3 Update `README.md` and related command/output documentation for strict freshness semantics, evidence interpretation, and publish provenance behavior.
