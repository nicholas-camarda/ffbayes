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
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
  await page.locator('#ready').waitFor({ state: 'visible' });

  await page.locator('#draft-slot').fill('2');
  await page.locator('#current-pick').fill('2');
  await page.locator('#recalculate').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('next pick 19'));

  const firstId = await page.locator('#board tr').first().getAttribute('data-player-id');
  await page.locator(`#board tr[data-player-id="${firstId}"] button[data-action="taken"]`).click();
  await page.waitForFunction((id) => document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('taken'), firstId);

  await page.locator('#league').selectOption('family');
  await page.waitForTimeout(100);
  await page.waitForFunction((id) => !document.querySelector(`#board tr[data-player-id="${id}"]`)?.textContent?.includes('taken'), firstId);

  await page.locator('#draft-slot').fill('0');
  await page.locator('#recalculate').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('draft_slot'));

  await page.locator('#draft-slot').fill('9');
  await page.locator('#current-pick').fill('9');
  await page.locator('#recalculate').click();
  await page.locator('#snapshot').click();
  await page.waitForFunction(() => document.querySelector('#status')?.textContent?.includes('Snapshot written'));

  console.log('2026 interactive dashboard browser smoke passed');
} finally {
  await browser.close();
  serverProcess.kill('SIGTERM');
}
