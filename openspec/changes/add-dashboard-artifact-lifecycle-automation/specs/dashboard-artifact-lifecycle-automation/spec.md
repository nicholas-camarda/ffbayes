## ADDED Requirements

### Requirement: Dashboard HTML SHALL be refreshable from an existing payload
The system MUST provide a cheap regeneration path that rebuilds dashboard HTML from an existing dashboard payload without rerunning the full draft-strategy model pipeline.

#### Scenario: Refresh regenerates from payload only
- **WHEN** `ffbayes refresh-dashboard` is run with a valid runtime dashboard payload
- **THEN** the command MUST regenerate the dashboard HTML from that payload without requiring `ffbayes draft-strategy` to rerun

#### Scenario: Missing payload blocks refresh
- **WHEN** the requested dashboard payload does not exist
- **THEN** the command MUST fail with an explicit missing-payload error instead of fabricating a new dashboard artifact

### Requirement: Dashboard refresh SHALL expose a machine-readable check mode
The system MUST expose a non-mutating refresh check mode that reports whether dashboard artifacts are fresh or stale using a structured result contract suitable for CI and local operators.

#### Scenario: Check mode reports synchronized artifacts
- **WHEN** `ffbayes refresh-dashboard --check --json` is run and the compared artifacts match regeneration from the current payload and template
- **THEN** the command MUST emit a machine-readable result indicating a synchronized or fresh state without mutating dashboard files

#### Scenario: Check mode reports stale artifacts
- **WHEN** `ffbayes refresh-dashboard --check --json` is run and one or more compared artifacts differ from regeneration output
- **THEN** the command MUST emit a machine-readable result identifying the stale state and the drifted paths

### Requirement: Stale dashboard artifacts SHALL be detectable by regeneration comparison
The system MUST be able to detect when a staged or runtime dashboard artifact is out of sync with the HTML that would be regenerated from the current payload and template.

#### Scenario: Staged HTML diverges from regenerated HTML
- **WHEN** the committed `site/index.html` or runtime dashboard HTML differs from the HTML regenerated from the current payload
- **THEN** the system MUST surface the artifact as stale or out of sync

#### Scenario: Regenerated HTML matches the staged copy
- **WHEN** the committed staged dashboard matches the HTML regenerated from the current payload
- **THEN** the system MUST treat the artifact as synchronized

### Requirement: Dashboard lifecycle automation SHALL preserve source-of-truth boundaries
The system MUST keep `draft-strategy` as the canonical payload generator, `refresh-dashboard` as the cheap HTML repair path, and `publish-pages` as the Pages staging path while preserving the existing dashboard filenames and directories.

#### Scenario: Pages staging keeps canonical targets stable
- **WHEN** dashboard artifacts are staged for GitHub Pages
- **THEN** the system MUST continue publishing through `site/index.html` and `site/dashboard_payload.json`

#### Scenario: Lifecycle commands expose explicit path ownership
- **WHEN** an operator uses `draft-strategy`, `refresh-dashboard`, or `publish-pages`
- **THEN** the command output or workflow validation MUST make it clear which artifact family is authoritative and which files are derived copies

### Requirement: Dashboard sync validation SHALL run separately from Pages deployment
The system MUST validate dashboard sync in a dedicated workflow or validation path that runs before deployment, while leaving the Pages workflow focused on publishing the already-validated `site/` bundle.

#### Scenario: Validation fails before deploy on dashboard drift
- **WHEN** dashboard templates, payloads, or staged `site/` artifacts drift out of sync in a pull request or validation-triggering push
- **THEN** the dedicated validation workflow MUST fail before the Pages deployment workflow is used to publish the bundle

#### Scenario: Pages workflow remains deploy-focused
- **WHEN** `.github/workflows/pages.yml` runs
- **THEN** it MUST continue deploying the `site/` bundle rather than acting as the first or only place dashboard sync drift is detected
