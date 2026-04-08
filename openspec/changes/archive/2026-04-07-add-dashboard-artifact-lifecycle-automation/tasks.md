## 1. Dashboard Refresh Path

- [x] 1.1 Harden `src/ffbayes/refresh_dashboard.py` so it is the cheap HTML regeneration path from an existing payload, with explicit failure on missing or mismatched inputs.
- [x] 1.2 Update the unified CLI in `src/ffbayes/cli.py` and any module-level entrypoints so operators can discover and run the refresh flow without rerunning `draft-strategy`.
- [x] 1.3 Add a non-mutating `refresh-dashboard --check --json` mode that emits a machine-readable stale/fresh result contract for local use and CI.
- [x] 1.4 Make the refresh command output clearly identify the source payload, regenerated HTML, whether Pages staging was requested, and which paths were checked or found stale.

## 2. Stale Artifact Detection And Staging

- [x] 2.1 Extend `src/ffbayes/publish_pages.py` and related dashboard helpers so staging can detect and surface stale or out-of-sync `site/` artifacts instead of silently copying them.
- [x] 2.2 Keep the canonical filenames and directories stable while making source-of-truth boundaries explicit for runtime artifacts, repo-local shortcuts, and staged Pages files.
- [x] 2.3 Add or update tests for stale-detection behavior, Pages staging, and the refresh-to-stage flow.

## 3. CI And Workflow Automation

- [x] 3.1 Add a dedicated GitHub Actions validation workflow that checks `site/` against regeneration from the tracked payload/template and fails on drift before merge.
- [x] 3.2 Keep `.github/workflows/pages.yml` focused on deploying `site/` when the bundle is in sync and does not rename or relocate the published targets.
- [x] 3.3 Update smoke or workflow coverage so the new validation path is exercised alongside existing dashboard and publish tests.

## 4. Documentation And Operator Guidance

- [x] 4.1 Update `README.md` to explain when to use `draft-strategy`, `refresh-dashboard`, `refresh-dashboard --check`, and `publish-pages`, and which artifacts each command owns.
- [x] 4.2 Document the stale-artifact detection behavior, the explicit source-of-truth boundaries, the machine-readable check output, and the expected failure modes when payloads or staged files are missing.
