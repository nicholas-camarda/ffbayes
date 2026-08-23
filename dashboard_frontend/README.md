# FFBayes dashboard frontend

This React/Vite package is an optional frontend development surface. The
draft-day operator path is the Python service:

```bash
ffbayes dashboard --year 2026
```

The Python service owns the current payload and all valuation/provenance logic.
Node is not required to run the dashboard.

## Developer workflow

```bash
cd dashboard_frontend
npm ci
npm test
npm run typecheck
npm run build
npm run build:template
```

`build:template` copies the single-file build to
`src/ffbayes/dashboard/assets/dashboard_template.html`, where the Python
package can use it. Review the generated template with frontend source changes.

## Documentation

- [Dashboard architecture](../docs/DASHBOARD_FRONTEND_ARCHITECTURE.md)
- [Frontend maintenance](../docs/DASHBOARD_FRONTEND_CUTOVER.md)
- [Operator guide](../docs/DASHBOARD_OPERATOR_GUIDE.md)
