import process from 'node:process';
import readline from 'node:readline';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const python = process.env.PYTHON || 'python';
const serverProcess = spawn(python, ['tests/serve_draft_2026_fixture.py'], {
  cwd: process.cwd(),
  env: { ...process.env, PYTHONPATH: 'src:tests' },
  stdio: ['ignore', 'pipe', 'inherit'],
});
const lines = readline.createInterface({ input: serverProcess.stdout });
const port = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('Timed out waiting for fixture service')), 10000);
  lines.on('line', (line) => {
    if (line.startsWith('PORT=')) {
      clearTimeout(timeout);
      resolve(Number(line.slice('PORT='.length)));
    }
  });
  serverProcess.once('error', reject);
});

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleErrors = [];
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
  await page.locator('#ready').waitFor({ state: 'visible' });
  for (const panel of ['#recommendation-panel', '#timing-frontier', '#positional-cliffs', '#comparative-explainer', '#roster-panel', '#queue-panel', '#freshness-panel', '#provenance-details']) {
    await page.locator(panel).waitFor({ state: 'visible' });
  }
  if (await page.locator('#provenance-details').getAttribute('open') !== null) throw new Error('Technical provenance should be collapsed by default');
  if (await page.locator('#provenance').isVisible()) throw new Error('Raw provenance JSON should not be prominent by default');
  if ((await page.locator('#freshness').textContent()).includes('{')) throw new Error('Data health should be human-readable, not raw JSON');
  if (await page.locator('#frontier .frontier-row').count() < 1) throw new Error('Timing frontier visual rows are missing');
  if (await page.locator('#frontier .metric-track').count() < 3) throw new Error('Timing frontier metric bars are missing');
  if (await page.locator('#cliffs .cliff-card').count() < 1) throw new Error('Positional cliff visual cards are missing');
  if (await page.locator('#comparative .comparative-row').count() < 1) throw new Error('Market/model comparison visual rows are missing');
  if (await page.locator('#comparative .rank-marker').count() < 2) throw new Error('Market/model rank markers are missing');
  const comparativeBox = await page.locator('#comparative-explainer').boundingBox();
  const healthBox = await page.locator('#freshness-panel').boundingBox();
  if (!comparativeBox || !healthBox || healthBox.y <= comparativeBox.y) throw new Error('Data health should follow market/model comparison');

  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 1'));
  await page.locator('#draft-slot').fill('2');
  await page.locator('#recalculate').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 1') && document.querySelector('#status')?.textContent?.includes('next pick 2'));

  const firstId = await page.locator('#board tr').first().getAttribute('data-player-id');
  const recommendationBefore = await page.locator('#recommendation').textContent();
  const frontierBefore = await page.locator('#frontier').textContent();
  const cliffsBefore = await page.locator('#cliffs').textContent();
  await page.locator(`#board tr[data-player-id="${firstId}"] button[data-action="taken"]`).click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 2') && document.querySelector('#status')?.textContent?.includes('next pick 19'));
  await page.waitForFunction((id) => document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('taken'), firstId);
  const recommendationAfter = await page.locator('#recommendation').textContent();
  if (recommendationBefore === recommendationAfter) throw new Error('Recommendation panel did not recompute after a confirmed pick');
  if (frontierBefore === await page.locator('#frontier').textContent()) throw new Error('Timing frontier did not recompute');
  if (cliffsBefore === await page.locator('#cliffs').textContent()) throw new Error('Positional cliffs did not recompute');

  const secondId = await page.locator('#board tr').nth(1).getAttribute('data-player-id');
  await page.locator(`#board tr[data-player-id="${secondId}"] button[data-action="mine"]`).click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 3'));
  await page.waitForFunction((id) => document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('mine'), secondId);

  const thirdId = await page.locator('#board tr').nth(2).getAttribute('data-player-id');
  await page.locator(`#board tr[data-player-id="${thirdId}"] button[data-action="queue"]`).click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 3'));
  await page.waitForFunction((id) => document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('queued'), thirdId);

  await page.locator('#undo').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 2'));
  await page.waitForFunction((id) => !document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('mine'), secondId);

  await page.locator('#league').selectOption('family');
  await page.waitForTimeout(100);
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 1'));
  if (await page.locator('#draft-slot').inputValue() !== '') throw new Error('League state leaked draft slot across profiles');
  await page.waitForFunction((id) => !document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('taken'), firstId);

  consoleErrors.length = 0;
  await page.locator('#draft-slot').fill('0');
  await page.locator('#recalculate').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('draft_slot'));

  consoleErrors.length = 0;
  await page.locator('#draft-slot').fill('9');
  await page.locator('#current-pick').fill('9');
  await page.locator('#recalculate').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 9'));
  const correctedId = await page.locator('#board tr').first().getAttribute('data-player-id');
  await page.locator(`#board tr[data-player-id="${correctedId}"] button[data-action="taken"]`).click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Current pick 10'));
  await page.locator('#snapshot').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Snapshot written'));

  if (consoleErrors.length) throw new Error(`Browser console errors: ${consoleErrors.join('; ')}`);
  console.log('2026 interactive dashboard browser smoke passed');
} finally {
  await browser.close();
  serverProcess.kill('SIGTERM');
}
