import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const testsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(testsDir, '..');
const canonicalSmoke = resolve(repoRoot, 'tests/test_draft_2026_dashboard_browser.mjs');

const result = spawnSync(process.execPath, [canonicalSmoke], {
  cwd: repoRoot,
  env: process.env,
  stdio: 'inherit',
});

if (result.error) {
  console.error(result.error);
  process.exit(1);
}

process.exit(result.status ?? 1);
