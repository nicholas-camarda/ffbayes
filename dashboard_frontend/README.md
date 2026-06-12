# FFBayes Dashboard Frontend

React + Vite source for the draft war room dashboard. Python remains the
payload source of truth; this package builds a single self-contained HTML
template that the pipeline injects at artifact-write time.

## Operator note

Normal draft-day commands (`ffbayes draft-strategy`, `ffbayes stage-dashboard`)
do **not** require Node. The committed template lives at
`src/ffbayes/dashboard/assets/dashboard_template.html`.

## Developer workflow

```bash
cd dashboard_frontend
npm ci
npm run dev          # local UI against public/dashboard_payload.json (gitignored)
npm test             # Vitest unit tests
npm run typecheck
npm run build:template   # writes packaged template into src/ffbayes/dashboard/
```

Commit both frontend source changes and the regenerated
`dashboard_template.html` when UI behavior changes.

## Docs

- Architecture: `docs/DASHBOARD_FRONTEND_ARCHITECTURE.md`
- Cutover and rollback: `docs/DASHBOARD_FRONTEND_CUTOVER.md`
- Operator guide: `docs/DASHBOARD_OPERATOR_GUIDE.md`
