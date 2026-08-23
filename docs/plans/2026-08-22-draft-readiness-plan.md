# 2026 draft-readiness implementation plan

1. Add failing source-contract tests for current eligibility, projection depth, ADP coverage, season/freshness, and provenance digests.
2. Implement ESPN 2026 player/projection/market ingestion plus the current-roster cross-check and run-scoped manifests.
3. Add failing league-profile tests for completeness and sensitivity to FLEX count, league size, scoring, and draft slot.
4. Implement explicit league profiles and league-specific scoring/replacement calculations.
5. Add failing recommendation tests for one-QB timing, no missing-ADP fallback, market opportunity cost, and bad-universe rejection.
6. Replace the canonical historical-target and FantasyPros paths with the validated current-season inputs.
7. Repair data-quality scoring and make all critical validation results propagate to the pipeline exit status.
8. Add output-provenance tests tying dashboard payloads to source manifests, profile digest, code revision, and generation time.
9. Run clean isolated pipelines for both leagues, inspect top 25/50/100 and positional boards, and compare major outliers with current external references.
10. Run the complete Python, type, lint, frontend, smoke, integration, ingestion, and pipeline validation suite from the worktree.

