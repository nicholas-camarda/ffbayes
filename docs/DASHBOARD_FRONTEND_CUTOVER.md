# Frontend maintenance

The Python loopback page is the current operator surface:

```bash
ffbayes dashboard --year 2026
```

It is self-contained and does not require Node at draft time. The optional
React package under `dashboard_frontend/` is used to develop and test frontend
components and to regenerate the packaged HTML template.

## Development checks

```bash
cd dashboard_frontend
npm ci
npm test
npm run typecheck
npm run build:template
```

When a frontend change affects the packaged template, review the generated
template and commit it with the source change. Keep valuation and provenance
logic in the Python service; the browser is a renderer and state-entry surface.

The browser smoke test uses a fixture-backed local service and exercises profile
selection, runtime slot validation, draft-state actions, and snapshot export.
