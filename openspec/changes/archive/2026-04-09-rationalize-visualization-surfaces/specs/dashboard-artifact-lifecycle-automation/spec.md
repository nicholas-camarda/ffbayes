## MODIFIED Requirements

### Requirement: Dashboard lifecycle automation SHALL preserve source-of-truth boundaries
The system MUST keep `draft-strategy` as the canonical runtime dashboard generator, `refresh-dashboard` as the cheap HTML and shortcut repair path, and `publish-pages` as the Pages staging path while preserving the existing dashboard filenames and directories. Repo-local `dashboard/`, runtime-root `dashboard/`, and staged `site/` copies MUST remain derived surfaces of the canonical runtime artifact pair and MUST NOT silently diverge in payload content or trust state.

#### Scenario: Pages staging keeps canonical targets stable
- **WHEN** dashboard artifacts are staged for GitHub Pages
- **THEN** the system MUST continue publishing through `site/index.html` and `site/dashboard_payload.json`

#### Scenario: Pages staging preserves current inline bootstrap
- **WHEN** `publish-pages` stages `site/index.html` for the current visualization lifecycle
- **THEN** the staged HTML MUST continue bootstrapping from the inline payload contract expected by the dashboard until a later change explicitly replaces that load path

#### Scenario: Lifecycle commands expose explicit path ownership
- **WHEN** an operator uses `draft-strategy`, `refresh-dashboard`, or `publish-pages`
- **THEN** the command output or workflow validation MUST make it clear which artifact family is authoritative and which files are derived copies

#### Scenario: Repo-local shortcut mirrors canonical runtime artifacts
- **WHEN** the system stages the repo-root `dashboard/index.html` and paired payload after dashboard generation or refresh
- **THEN** those files MUST be derived from the canonical runtime dashboard HTML and payload for the same year rather than from sample content or a mismatched artifact pair

#### Scenario: Repo-local shortcut surface stays narrow
- **WHEN** the system stages repo-root `dashboard/`
- **THEN** it MUST stage only the convenience pair and direct shortcut companions rather than using that directory as a general visualization artifact sink

#### Scenario: Drifted derived copies are detectable
- **WHEN** repo-local `dashboard/`, runtime-root `dashboard/`, or staged `site/` artifacts differ from regeneration from their authoritative payload and template
- **THEN** the system MUST surface those paths as stale or out of sync instead of implying they are synchronized

## ADDED Requirements

### Requirement: Dashboard shortcut validation SHALL cover local convenience surfaces
The system MUST provide automated validation for repo-local convenience dashboard artifacts in addition to staged Pages validation.

#### Scenario: Repo-local shortcut is stale
- **WHEN** the repo-root `dashboard/index.html` or paired payload diverges from the canonical runtime dashboard artifacts
- **THEN** automated validation MUST be able to detect the mismatch without mutating the files

#### Scenario: Repo-local shortcut is synchronized
- **WHEN** the repo-root `dashboard/` HTML and payload match the canonical runtime dashboard artifact pair
- **THEN** automated validation MUST report the shortcut as synchronized
