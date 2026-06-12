# Dashboard Frontend Cutover

Audience: operators and contributors validating or rolling back the dashboard
renderer cutover.

Scope: cutover status, usability review, pre-draft-day verification, and
rollback commands.

Trust boundary: the frontend renderer is now default. Legacy Python HTML is
rollback-only via `FFBAYES_DASHBOARD_RENDERER=legacy`.

## What This Is

Cutover checklist and usability notes for the switch from the inline Python HTML
dashboard to the React+Vite frontend.

## When To Use It

Use this guide when:

- verifying the dashboard before draft day
- rolling back to the legacy renderer
- planning deletion of legacy `export_dashboard_html()` code
- onboarding contributors to the post-cutover layout

## What To Inspect

- Staged `site/index.html` contains `id="ffbayes-dashboard-payload"`
- Smoke suite: `node tests/dashboard_smoke.mjs`
- Rollback import: `ffbayes.dashboard.legacy_renderer`

## Interpretation Boundaries

- Legacy code is intentionally retained until one stable draft day on the React UI.
- Deleting legacy HTML is a follow-up PR, not part of this cutover.

## Commands And Paths

See the operator checklist and rollback sections below.

## Cutover status

| Item | Status |
|------|--------|
| Default renderer | `frontend` (`FFBAYES_DASHBOARD_RENDERER` unset) |
| Rollback | `FFBAYES_DASHBOARD_RENDERER=legacy` |
| Template artifact | `src/ffbayes/dashboard/assets/dashboard_template.html` (committed) |
| Smoke suite | `node tests/dashboard_smoke.mjs` (targets frontend DOM contract) |
| Legacy code | `ffbayes.dashboard.legacy_renderer` re-exports `export_dashboard_html()` in `draft_decision_system.py` (rollback only) |
| Legacy deletion | **Not yet** — remove after one stable draft day on the React dashboard |

## Usability review (2026-06-12)

### Strengths

- **Draft-day layout**: Three-column war room (recommendations / board / settings+queue) matches legacy mental model; center column is wider on large screens for the player board.
- **Dark theme**: Ported legacy gradient and CSS variables; readable in low-light draft environments.
- **Responsive**: Tablet collapses to two columns with full-width board; mobile stacks single-column with touch-friendly controls and hides low-priority board columns (VOR, survival).
- **Payload embedding**: JSON in `<script type="application/json" id="ffbayes-dashboard-payload">` loads correctly over `file://` and HTTP (fixes prior “stuck on Loading…” when bundle ran before payload).
- **Offline**: Single-file HTML + `localStorage` draft state (`ffbayes-dashboard-state-v2`) — same key as legacy v2 state.
- **Staged vs file**: Finalize bundle UI appears only on `file://` (legacy parity); GitHub Pages stays read-only.
- **Degradation**: Missing optional payload sections hide panels instead of crashing (SectionGate).

### Known gaps / acceptable tradeoffs

- **Reset button**: Legacy `#reset-button` is optional in smoke tests; frontend may not expose an identical control — use browser devtools or clear `ffbayes-dashboard-state-v2` if needed.
- **Board horizontal scroll on mobile**: Expected for wide tables; primary actions remain reachable.
- **Settings in right column on desktop**: Same as legacy; not moved to a top drawer.

### Pre-cutover verification (run before draft day)

```bash
# Frontend toolchain
cd dashboard_frontend && npm ci && npm test && npm run typecheck && npm run build:template

# Python suite
conda run -n ffbayes pytest -q

# Regenerate runtime dashboard (default = frontend)
ffbayes stage-dashboard --year 2026

# Smoke against staged or runtime site dir
FFBAYES_SMOKE_SITE_DIR=site node tests/dashboard_smoke.mjs
```

Open the generated `index.html` locally (`file://`) and on GitHub Pages preview if staged.

## Operator checklist

### Normal use (after cutover)

1. Run pipeline / `ffbayes stage-dashboard --year YYYY` as usual — no env var required.
2. Open `site/index.html` or published Pages URL.
3. Confirm pick number, recommendations, and board populate.
4. On draft day, use `file://` copy for finalize bundle if needed.

### Rollback (one command)

```bash
FFBAYES_DASHBOARD_RENDERER=legacy ffbayes stage-dashboard --year 2026
```

Re-stage `site/` if publishing to GitHub Pages.

### Rebuild template after UI changes

```bash
cd dashboard_frontend && npm ci && npm run build:template
```

Commit the updated `dashboard_template.html` with the frontend source change.

### CI follow-up (recommended)

- Add a job step: `cd dashboard_frontend && npm run build:template && git diff --exit-code src/ffbayes/dashboard/assets/dashboard_template.html`
- Prevents template drift from committed React source.

## Architecture reminder

- **Python** builds and validates `dashboard_payload.json`; **browser** only renders.
- **No Node at pipeline runtime** — only when developers change `dashboard_frontend/`.
- **Publishing**: `publish_pages.py` reinjects payload into staged `site/index.html`.
