## Why

The repo has already shown that dashboard artifacts can drift: runtime HTML, staged Pages files, and source templates can fall out of sync unless someone remembers the exact regeneration sequence. That creates a reliability tax on every dashboard change and makes publication correctness depend on manual discipline.

## What Changes

- Add a lightweight dashboard regeneration path that rebuilds the HTML from the current runtime payload without rerunning the full draft strategy.
- Extend that refresh path with a non-mutating machine-readable check mode so local operators and CI can detect stale artifacts from the same command family.
- Add stronger lifecycle checks so stale runtime or Pages artifacts are easier to detect before they are published.
- Tighten the publish-and-stage workflow so `site/`, runtime shortcuts, and canonical artifacts stay aligned with the current dashboard source.
- Add a dedicated validation workflow for dashboard sync checks while keeping `pages.yml` focused on deployment of already-validated `site/`.
- Preserve the existing artifact names and phase structure while making regeneration and restaging more explicit and less error-prone.

## Capabilities

### New Capabilities
- `dashboard-artifact-lifecycle-automation`: define cheap regeneration, restaging, and drift-detection behavior for dashboard artifacts across runtime and `site/`.

### Modified Capabilities
- None.

## Impact

- Affects dashboard generation and staging surfaces, especially `src/ffbayes/publish_pages.py`, `src/ffbayes/refresh_dashboard.py`, the unified CLI, and the runtime dashboard export path in `src/ffbayes/draft_strategy/draft_decision_system.py`.
- Touches repo-tracked `site/` files and runtime dashboard shortcuts under `~/ProjectsRuntime/ffbayes/dashboard/`.
- Adds workflow responsibilities under `.github/workflows/`, separating dashboard-sync validation from Pages deployment.
- Lowers the operational cost of keeping staged Pages output synchronized with the current runtime payload, but it also raises the importance of clear source-of-truth boundaries.
- No phase migration is intended; this remains within the supported `pre_draft` workflow.
