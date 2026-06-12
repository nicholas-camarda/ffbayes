// Copies the single-file Vite build into the Python package as the dashboard template.
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, '../dist/index.html');
const target = resolve(here, '../../src/ffbayes/dashboard/assets/dashboard_template.html');

const html = readFileSync(dist, 'utf-8');
for (const marker of ['window.FFBAYES_DASHBOARD = __PAYLOAD_JSON__', '/*FFBAYES_PAYLOAD_END*/']) {
  if (!html.includes(marker)) {
    console.error(`Built template is missing required marker: ${marker}`);
    process.exit(1);
  }
}
mkdirSync(dirname(target), { recursive: true });
copyFileSync(dist, target);
console.log(`Template staged at ${target} (${html.length} bytes)`);
