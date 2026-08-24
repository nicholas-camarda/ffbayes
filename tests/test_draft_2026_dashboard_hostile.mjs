import process from 'node:process';
import readline from 'node:readline';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const python = process.env.PYTHON || 'python';
const serverProcess = spawn(python, ['tests/serve_draft_2026_hostile_fixture.py'], {
  cwd: process.cwd(),
  env: { ...process.env, PYTHONPATH: 'src:tests' },
  stdio: ['ignore', 'pipe', 'inherit'],
});
const lines = readline.createInterface({ input: serverProcess.stdout });
const port = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('Timed out waiting for hostile fixture service')), 10000);
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
  await page.locator('#board tr').first().waitFor({ state: 'visible' });

  const result = await page.evaluate(() => ({
    rowText: [...document.querySelectorAll('#board tr')].find((row) => row.textContent.includes('hostile-player'))?.textContent || '',
    imageCount: [...document.querySelectorAll('#board tr')].find((row) => row.textContent.includes('hostile-player'))?.querySelectorAll('img').length || 0,
    injectedButtonCount: [...document.querySelectorAll('#board tr')].find((row) => row.textContent.includes('hostile-player'))?.querySelectorAll('button[data-action="evil"]').length || 0,
    hostileExecuted: window.__hostileExecuted === true,
  }));
  if (!result.rowText.includes('<img src=x onerror="window.__hostileExecuted=true"> hostile-player')) {
    throw new Error('Hostile player name was not rendered as literal text');
  }
  if (!result.rowText.includes('<button data-action="evil" onclick="window.__hostileExecuted=true">pwned</button>')) {
    throw new Error('Hostile recommendation was not rendered as literal text');
  }
  if (result.imageCount !== 0 || result.injectedButtonCount !== 0 || result.hostileExecuted) {
    throw new Error(`Hostile input created DOM or executed code: ${JSON.stringify(result)}`);
  }
  console.log('2026 hostile dashboard rendering test passed');
} finally {
  await browser.close();
  serverProcess.kill('SIGTERM');
}
