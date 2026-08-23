import process from 'node:process';
import { chromium } from 'playwright';

let html = '';
for await (const chunk of process.stdin) {
  html += chunk;
}
if (!html.trim()) {
  throw new Error('Expected rendered draft dashboard HTML on stdin');
}

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.setContent(html, { waitUntil: 'load' });

  const result = await page.evaluate(() => {
    const payloadElement = document.querySelector('#draft-2026-payload');
    const rows = [...document.querySelectorAll('#draft-board tbody tr')];
    if (!payloadElement) {
      throw new Error('Validated payload element is missing');
    }
    const payload = JSON.parse(payloadElement.textContent || '');
    return {
      schema: payload.schema_version,
      season: payload.season,
      profileSeason: payload.league_profile?.season,
      coverage: payload.coverage_report?.status,
      sourceSeasons: (payload.provenance?.source_manifests || []).map(
        (manifest) => manifest.season,
      ),
      payloadRows: payload.decision_table?.length || 0,
      renderedRows: rows.length,
      tableVisible: Boolean(document.querySelector('#draft-board')?.offsetParent),
    };
  });

  if (result.schema !== 'draft_2026_v1') throw new Error(`Unexpected schema: ${result.schema}`);
  if (result.season !== 2026 || result.profileSeason !== 2026) throw new Error('Season mismatch');
  if (result.coverage !== 'passed') throw new Error('Coverage is not passed');
  if (!result.sourceSeasons.length || result.sourceSeasons.some((season) => season !== 2026)) {
    throw new Error('Source provenance season mismatch');
  }
  if (!result.payloadRows || result.renderedRows !== Math.min(100, result.payloadRows)) {
    throw new Error('Rendered table does not match the validated payload');
  }
  if (!result.tableVisible) throw new Error('Draft board is not visible');
  if (pageErrors.length) throw new Error(`Browser errors: ${pageErrors.join('; ')}`);

  console.log(`2026 dashboard smoke passed: ${result.renderedRows} rendered rows`);
} finally {
  await browser.close();
}
