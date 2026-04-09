import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');
const repoRoot = resolve(__dirname, '..');
const siteDir = resolve(repoRoot, 'site');
const siteIndexPath = resolve(siteDir, 'index.html');

async function startStaticServer(rootDir) {
  const resolvedRoot = resolve(rootDir);
  const server = createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    const requestPath = url.pathname === '/' ? '/index.html' : url.pathname;
    const filePath = resolve(resolvedRoot, `.${requestPath}`);
    if (!filePath.startsWith(resolvedRoot)) {
      res.statusCode = 403;
      res.end('forbidden');
      return;
    }
    try {
      const body = await readFile(filePath);
      if (filePath.endsWith('.html')) {
        res.setHeader('content-type', 'text/html; charset=utf-8');
      } else if (filePath.endsWith('.json')) {
        res.setHeader('content-type', 'application/json; charset=utf-8');
      }
      res.statusCode = 200;
      res.end(body);
    } catch (_) {
      res.statusCode = 404;
      res.end('not found');
    }
  });
  await new Promise((resolveListen) => server.listen(0, '127.0.0.1', resolveListen));
  const address = server.address();
  return {
    server,
    url: `http://${address.address}:${address.port}/index.html`,
  };
}

async function runSmoke() {
  const localUrl = pathToFileURL(siteIndexPath).href;
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    acceptDownloads: true,
  });
  const page = await context.newPage();
  const downloadDir = await mkdtemp(join(tmpdir(), 'ffbayes-finalize-'));
  const downloads = [];
  page.on('download', (download) => downloads.push(download));

  const selectors = {
    leagueSize: '#league-size',
    draftPosition: '#draft-position',
    currentPickNumber: '#current-pick-number',
    benchSlots: '#bench-slots',
    riskTolerance: '#risk-tolerance',
    undoButton: '#undo-button',
    redoButton: '#redo-button',
    finalizeButton: '#finalize-button',
    finalizeNote: '#finalize-note',
    statusPills: '#status-pills .pill',
    primaryName: '#primary-card .hero-name',
    timingFrontier: '#timing-frontier',
    frontierPoints: '#timing-frontier [data-frontier-player]',
    positionalCliffs: '#positional-cliffs',
    cliffPlayers: '#positional-cliffs [data-cliff-player]',
    inspector: '#player-inspector',
    rosterNeeds: '#roster-need-grid .metric',
    rosterChips: '#my-roster .pill',
  };

  const setNumber = async (selector, value) => {
    await page.locator(selector).fill(String(value));
    await page.locator(selector).dispatchEvent('change');
    await page.waitForTimeout(75);
  };

  const setSelect = async (selector, value) => {
    await page.locator(selector).selectOption(String(value));
    await page.waitForTimeout(75);
  };

  const clickAction = async (action, playerName) => {
    await page.evaluate(({ actionName, targetName }) => {
      const buttons = Array.from(document.querySelectorAll(`button[data-action="${actionName}"]`));
      const button = buttons.find((node) => node.getAttribute('data-player') === targetName);
      if (!button) {
        throw new Error(`Could not find ${actionName} button for ${targetName}`);
      }
      button.click();
    }, { actionName: action, targetName: playerName });
    await page.waitForTimeout(125);
  };

  const clickBoardRow = async (playerName) => {
    await page.evaluate((targetName) => {
      const row = Array.from(document.querySelectorAll('#board-table tr[data-player-row]'))
        .find((node) => node.getAttribute('data-player-row') === targetName);
      if (!row) {
        throw new Error(`Could not find board row for ${targetName}`);
      }
      row.click();
    }, playerName);
    await page.waitForTimeout(125);
  };

  const clickFrontierPoint = async (playerName) => {
    await page.evaluate((targetName) => {
      const button = Array.from(document.querySelectorAll('#timing-frontier [data-frontier-player]'))
        .find((node) => node.getAttribute('data-frontier-player') === targetName);
      if (!button) {
        throw new Error(`Could not find timing frontier point for ${targetName}`);
      }
      button.click();
    }, playerName);
    await page.waitForTimeout(125);
  };

  const clickCliffPlayer = async (playerName) => {
    await page.evaluate((targetName) => {
      const button = Array.from(document.querySelectorAll('#positional-cliffs [data-cliff-player]'))
        .find((node) => node.getAttribute('data-cliff-player') === targetName);
      if (!button) {
        throw new Error(`Could not find cliff player chip for ${targetName}`);
      }
      button.click();
    }, playerName);
    await page.waitForTimeout(125);
  };

  const readPillTexts = async () => page.locator(selectors.statusPills).allTextContents();
  const readText = async (selector) => page.locator(selector).first().textContent();
  const getInputValue = async (selector) => page.locator(selector).inputValue();
  const getSelectValue = async (selector) => page.locator(selector).inputValue();
  const isDisabled = async (selector) => page.locator(selector).isDisabled();
  const readStoredState = async () => page.evaluate(() => JSON.parse(window.localStorage.getItem('ffbayes-dashboard-state-v2') || 'null'));
  const rowStatus = async (playerName) => page.evaluate((targetName) => {
    const row = Array.from(document.querySelectorAll('#board-table tr[data-player-row]'))
      .find((node) => node.getAttribute('data-player-row') === targetName);
    return row?.querySelector('td:nth-child(2) .status-badge')?.textContent?.trim() || null;
  }, playerName);
  const readRosterNeedMap = async () => {
    const entries = await page.locator(selectors.rosterNeeds).evaluateAll((nodes) => nodes.map((node) => ({
      label: node.querySelector('.label')?.textContent?.trim(),
      value: node.querySelector('.value')?.textContent?.trim(),
    })));
    return Object.fromEntries(entries.map((item) => [item.label.replace(' need', ''), Number(item.value || 0)]));
  };

  const findFirstAvailablePlayerByPosition = async (position) => page.evaluate((targetPosition) => {
    const rows = Array.from(document.querySelectorAll('#board-table tr[data-player-row]'));
    const row = rows.find((node) => {
      const status = node.querySelector('td:nth-child(2) .status-badge')?.textContent?.trim();
      const positionText = node.querySelector('td:first-child .tiny')?.textContent?.trim() || '';
      const rowPosition = positionText.split('•')[0]?.trim() || '';
      return status === 'available' && rowPosition === targetPosition;
    });
    return row ? row.getAttribute('data-player-row') : null;
  }, position);

  const waitForPillText = async (expectedText) => {
    await page.waitForFunction(
      (text) => Array.from(document.querySelectorAll('#status-pills .pill'))
        .some((node) => node.textContent.trim() === text),
      expectedText,
    );
  };

  const waitForInspectorText = async (expectedText) => {
    await page.waitForFunction(
      (text) => (document.querySelector('#player-inspector')?.textContent || '').includes(text),
      expectedText,
    );
  };

  const waitForDownloadCount = async (expectedCount) => {
    const started = Date.now();
    while (downloads.length < expectedCount) {
      if (Date.now() - started > 6000) {
        throw new Error(`Timed out waiting for ${expectedCount} downloads, got ${downloads.length}`);
      }
      await page.waitForTimeout(50);
    }
  };

  const saveDownloads = async (downloadSlice) => Promise.all(downloadSlice.map(async (download) => {
    const filename = download.suggestedFilename();
    const outputPath = join(downloadDir, filename);
    await download.saveAs(outputPath);
    return { filename, outputPath };
  }));

  const stubFinalizeConfirm = async (returnValue) => {
    await page.evaluate((nextValue) => {
      window.__ffbayesFinalizeConfirmCalls = [];
      window.confirm = (message) => {
        window.__ffbayesFinalizeConfirmCalls.push(message);
        return nextValue;
      };
    }, returnValue);
  };

  const readFinalizeConfirmCalls = async () => page.evaluate(() => window.__ffbayesFinalizeConfirmCalls || []);

  let staticServer;
  try {
    await page.goto(localUrl, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => window.localStorage.removeItem('ffbayes-dashboard-state-v2'));
    await page.reload({ waitUntil: 'domcontentloaded' });

    if ((await readText('h1')) !== 'FFBayes Draft War Room') {
      throw new Error('Dashboard did not load the expected title');
    }
    const bodyText = (await page.textContent('body')) || '';
    if (!bodyText.includes('Decision evidence') || !bodyText.includes('Freshness and provenance')) {
      throw new Error('Dashboard did not render the decision evidence and provenance sections');
    }
    if (!bodyText.includes('Wait vs Pick Frontier') || !bodyText.includes('Positional Cliffs')) {
      throw new Error('Dashboard did not render the new war-room visual sections');
    }
    if (await page.locator('#reset-button').count()) {
      throw new Error('Reset button should not be present');
    }
    if (!(await page.locator(selectors.finalizeButton).isVisible())) {
      throw new Error('Finalize button should be visible on local file dashboards');
    }
    if (!((await readText(selectors.finalizeNote)) || '').includes('locked HTML snapshot')) {
      throw new Error('Local finalize note did not explain the export bundle');
    }

    await setNumber(selectors.leagueSize, 3);
    await setNumber(selectors.draftPosition, 2);
    await setNumber(selectors.currentPickNumber, 1);
    await setNumber(selectors.benchSlots, 0);
    await setSelect(selectors.riskTolerance, 'high');
    await page.evaluate(() => {
      const state = JSON.parse(window.localStorage.getItem('ffbayes-dashboard-state-v2') || 'null');
      if (!state) {
        return;
      }
      state.rosterSpots = { ...(state.rosterSpots || {}), DST: 0, K: 0 };
      window.localStorage.setItem('ffbayes-dashboard-state-v2', JSON.stringify(state));
    });
    await page.reload({ waitUntil: 'domcontentloaded' });

    if (!(await isDisabled(selectors.undoButton))) {
      throw new Error('Undo should start disabled');
    }
    if (!(await isDisabled(selectors.redoButton))) {
      throw new Error('Redo should start disabled');
    }

    const configuredPills = await readPillTexts();
    if (!configuredPills.includes('League: 3-team')) {
      throw new Error(`Expected league size pill to update, got ${configuredPills.join(' | ')}`);
    }
    if (!configuredPills.includes('Draft slot: 2')) {
      throw new Error(`Expected draft slot pill to update, got ${configuredPills.join(' | ')}`);
    }

    const primaryBeforeTaken = await readText(selectors.primaryName);
    if (!primaryBeforeTaken) {
      throw new Error('Primary recommendation was empty before draft actions');
    }
    if ((await page.locator(selectors.frontierPoints).count()) < 1) {
      throw new Error('Timing frontier did not render any candidate points');
    }
    if ((await page.locator(selectors.cliffPlayers).count()) < 1) {
      throw new Error('Positional cliff map did not render any player chips');
    }
    const frontierLegendText = (await page.locator(selectors.timingFrontier).textContent()) || '';
    if (frontierLegendText.includes('watch')) {
      throw new Error('Timing frontier fell back to a generic watch lane instead of explicit lane labels');
    }

    const frontierCandidate = await page.locator(selectors.frontierPoints).nth(0).getAttribute('data-frontier-player');
    if (!frontierCandidate) {
      throw new Error('Could not read the first timing frontier candidate');
    }
    await clickFrontierPoint(frontierCandidate);
    await waitForInspectorText(frontierCandidate);

    const cliffCandidate = await page.locator(selectors.cliffPlayers).nth(0).getAttribute('data-cliff-player');
    if (!cliffCandidate) {
      throw new Error('Could not read the first positional cliff candidate');
    }
    await clickCliffPlayer(cliffCandidate);
    await waitForInspectorText(cliffCandidate);

    const frontierTextBeforeTaken = (await page.locator(selectors.timingFrontier).textContent()) || '';

    await clickAction('taken', primaryBeforeTaken);
    await waitForPillText('Current pick: 2');
    const primaryAfterTaken = await readText(selectors.primaryName);
    if (primaryAfterTaken === primaryBeforeTaken) {
      throw new Error('Taken did not advance the draft to a new recommendation');
    }
    const frontierTextAfterTaken = (await page.locator(selectors.timingFrontier).textContent()) || '';
    if (frontierTextAfterTaken === frontierTextBeforeTaken) {
      throw new Error('Timing frontier did not respond to a draft-state change');
    }
    if (await isDisabled(selectors.undoButton)) {
      throw new Error('Undo should be enabled after a draft action');
    }
    if (!(await isDisabled(selectors.redoButton))) {
      throw new Error('Redo should remain disabled until an undo happens');
    }

    await page.locator(selectors.undoButton).click();
    await waitForPillText('Current pick: 1');
    const pillsAfterUndoTaken = await readPillTexts();
    if (!pillsAfterUndoTaken.includes('Taken: 0')) {
      throw new Error(`Undo should clear the taken count, got ${pillsAfterUndoTaken.join(' | ')}`);
    }
    if (!(await readText(selectors.primaryName)) || (await readText(selectors.primaryName)) !== primaryBeforeTaken) {
      throw new Error('Undo should restore the original primary recommendation');
    }
    if (await isDisabled(selectors.redoButton)) {
      throw new Error('Redo should be enabled after undo');
    }

    await page.reload({ waitUntil: 'domcontentloaded' });
    if (await isDisabled(selectors.redoButton)) {
      throw new Error('Redo should persist across reloads');
    }

    await page.locator(selectors.redoButton).click();
    await waitForPillText('Current pick: 2');
    const pillsAfterRedoTaken = await readPillTexts();
    if (!pillsAfterRedoTaken.includes('Taken: 1')) {
      throw new Error(`Redo should restore the taken action, got ${pillsAfterRedoTaken.join(' | ')}`);
    }

    await page.locator(selectors.undoButton).click();
    await waitForPillText('Current pick: 1');

    await clickAction('mine', primaryBeforeTaken);
    await waitForPillText('Yours: 1');
    const rosterAfterMine = await page.locator(selectors.rosterChips).allTextContents();
    if (!rosterAfterMine.some((text) => text.includes(primaryBeforeTaken))) {
      throw new Error('Mine did not add the selected player to the roster');
    }
    if ((await readStoredState())?.pickLog?.length !== 1) {
      throw new Error('Mine should create one pick receipt in local state');
    }

    await page.locator(selectors.undoButton).click();
    await waitForPillText('Yours: 0');
    if (await isDisabled(selectors.redoButton)) {
      throw new Error('Redo should be enabled after undoing Mine');
    }
    if ((await readStoredState())?.pickLog?.length !== 0) {
      throw new Error('Undo should restore the pick log');
    }

    await page.locator(selectors.redoButton).click();
    await waitForPillText('Yours: 1');
    if ((await readStoredState())?.pickLog?.length !== 1) {
      throw new Error('Redo should restore the pick receipt');
    }

    const stickyCandidate = await findFirstAvailablePlayerByPosition('WR');
    if (!stickyCandidate) {
      throw new Error('Could not find a WR candidate for inspector testing');
    }

    await clickBoardRow(stickyCandidate);
    await waitForInspectorText(stickyCandidate);
    const inspectorText = (await page.locator(selectors.inspector).textContent()) || '';
    if (!inspectorText.includes('Contextual vs baseline')) {
      throw new Error('Inspector did not render the contextual-versus-baseline explainer');
    }

    const blockerPlayer = await findFirstAvailablePlayerByPosition('RB');
    if (!blockerPlayer) {
      throw new Error('Could not find an RB candidate to trigger a state change');
    }

    await clickAction('taken', blockerPlayer);
    await waitForInspectorText(stickyCandidate);

    await clickAction('taken', stickyCandidate);
    await page.waitForFunction(
      (playerName) => !(document.querySelector('#player-inspector')?.textContent || '').includes(playerName),
      stickyCandidate,
    );
    const primaryAfterInvalidation = await readText(selectors.primaryName);
    if ((await page.locator(selectors.inspector).textContent() || '').includes(stickyCandidate)) {
      throw new Error('Inspector did not fall back after the selected player became invalid');
    }
    if (!primaryAfterInvalidation || !(await page.locator(selectors.inspector).textContent() || '').includes(primaryAfterInvalidation)) {
      throw new Error('Inspector did not fall back to the current pick recommendation');
    }

    const queueCandidate = await findFirstAvailablePlayerByPosition('TE');
    if (!queueCandidate) {
      throw new Error('Could not find a TE candidate for queue testing');
    }
    await clickAction('queue', queueCandidate);
    if ((await rowStatus(queueCandidate)) !== 'queued') {
      throw new Error('Queue action did not mark the player as queued');
    }
    await page.locator(selectors.undoButton).click();
    if ((await rowStatus(queueCandidate)) !== 'available') {
      throw new Error('Undo should restore a queued player to available');
    }
    if (await isDisabled(selectors.redoButton)) {
      throw new Error('Redo should be enabled after undoing Queue');
    }
    await page.locator(selectors.redoButton).click();
    if ((await rowStatus(queueCandidate)) !== 'queued') {
      throw new Error('Redo should restore the queued status');
    }
    await page.locator(selectors.undoButton).click();
    if ((await rowStatus(queueCandidate)) !== 'available') {
      throw new Error('Undo should restore the queued player before redo invalidation testing');
    }

    const redoInvalidationPlayer = await findFirstAvailablePlayerByPosition('QB');
    if (!redoInvalidationPlayer) {
      throw new Error('Could not find a QB candidate for redo invalidation testing');
    }
    await clickAction('taken', redoInvalidationPlayer);
    if (!(await isDisabled(selectors.redoButton))) {
      throw new Error('A new draft action after undo should clear redo history');
    }

    await stubFinalizeConfirm(false);
    const canceledDownloadStart = downloads.length;
    await page.locator(selectors.finalizeButton).click();
    await page.waitForTimeout(150);
    const finalizeCalls = await readFinalizeConfirmCalls();
    if (finalizeCalls.length !== 1 || !finalizeCalls[0].includes('roster is not full yet')) {
      throw new Error(`Incomplete finalize should prompt a warning, got ${finalizeCalls.join(' | ')}`);
    }
    if (downloads.length !== canceledDownloadStart) {
      throw new Error('Canceling finalize should not download artifacts');
    }

    const drafted = [];
    while (true) {
      const rosterNeed = await readRosterNeedMap();
      const nextNeed = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K'].find((position) => Number(rosterNeed[position] || 0) > 0);
      if (!nextNeed) {
        break;
      }
      let playerName = null;
      if (nextNeed === 'FLEX') {
        playerName = await findFirstAvailablePlayerByPosition('RB')
          || await findFirstAvailablePlayerByPosition('WR')
          || await findFirstAvailablePlayerByPosition('TE');
      } else {
        playerName = await findFirstAvailablePlayerByPosition(nextNeed);
      }
      if (!playerName) {
        throw new Error(`Could not find an available ${nextNeed} while filling the roster`);
      }
      drafted.push(playerName);
      await clickAction('mine', playerName);
    }

    const rosterNeed = await readRosterNeedMap();
    if (Object.values(rosterNeed).some((value) => value !== 0)) {
      throw new Error(`Expected roster needs to be fully satisfied, got ${JSON.stringify(rosterNeed)}`);
    }
    const rosterNames = await page.locator(selectors.rosterChips).allTextContents();
    if (rosterNames.length < 7) {
      throw new Error('Expected a full 7-player roster after filling all offensive starter slots');
    }

    await stubFinalizeConfirm(true);
    const downloadStart = downloads.length;
    await page.locator(selectors.finalizeButton).click();
    await waitForDownloadCount(downloadStart + 3);
    const savedDownloads = await saveDownloads(downloads.slice(downloadStart, downloadStart + 3));
    const jsonArtifact = savedDownloads.find((item) => item.filename.endsWith('.json'));
    const lockedHtmlArtifact = savedDownloads.find((item) => item.filename.includes('finalized_draft_') && item.filename.endsWith('.html'));
    const summaryHtmlArtifact = savedDownloads.find((item) => item.filename.includes('finalized_summary_') && item.filename.endsWith('.html'));
    if (!jsonArtifact || !lockedHtmlArtifact || !summaryHtmlArtifact) {
      throw new Error(`Finalize should create JSON + two HTML files, got ${savedDownloads.map((item) => item.filename).join(' | ')}`);
    }

    const finalizedPayload = JSON.parse(await readFile(jsonArtifact.outputPath, 'utf-8'));
    if (finalizedPayload.schema_version !== 'finalized_draft_v1') {
      throw new Error(`Unexpected finalized schema version: ${finalizedPayload.schema_version}`);
    }
    if (!Array.isArray(finalizedPayload.pick_receipts) || finalizedPayload.pick_receipts.length < 1) {
      throw new Error('Finalized JSON should include pick receipts');
    }
    if (!finalizedPayload.final_state?.roster_complete) {
      throw new Error('Finalized JSON should mark a complete roster after filling all starter slots');
    }

    const lockedPage = await context.newPage();
    await lockedPage.goto(pathToFileURL(lockedHtmlArtifact.outputPath).href, { waitUntil: 'domcontentloaded' });
    if (!((await lockedPage.textContent('body')) || '').includes('Read-only snapshot')) {
      throw new Error('Locked HTML should present itself as a read-only snapshot');
    }
    if ((await lockedPage.locator('#undo-button').count()) || (await lockedPage.locator('button[data-action]').count())) {
      throw new Error('Locked HTML should not expose mutating draft controls');
    }
    if (((await lockedPage.textContent('body')) || '').includes('Draft Controls')) {
      throw new Error('Locked HTML should not render the live draft controls section');
    }
    await lockedPage.close();

    const summaryPage = await context.newPage();
    await summaryPage.goto(pathToFileURL(summaryHtmlArtifact.outputPath).href, { waitUntil: 'domcontentloaded' });
    const summaryText = (await summaryPage.textContent('body')) || '';
    if (!summaryText.includes('FFBayes Post-Draft Summary')) {
      throw new Error('Summary HTML should render the post-draft title');
    }
    if (!summaryText.includes('Final Roster Recap') || !summaryText.includes('Team Projection Snapshot') || !summaryText.includes('Risk & Upside Profile') || !summaryText.includes('Draft Value Recap') || !summaryText.includes('Pick-by-Pick Receipts')) {
      throw new Error('Summary HTML is missing one or more required sections');
    }
    await summaryPage.close();

    staticServer = await startStaticServer(siteDir);
    const remotePage = await context.newPage();
    await remotePage.goto(staticServer.url, { waitUntil: 'domcontentloaded' });
    if (await remotePage.locator(selectors.finalizeButton).isVisible()) {
      throw new Error('Finalize button should be hidden on non-local origins');
    }
    const remoteNote = (await remotePage.textContent(selectors.finalizeNote)) || '';
    if (!remoteNote.includes('only supported from the local generated dashboard')) {
      throw new Error('Non-local finalize note did not explain the limitation');
    }
    const remoteProvenance = (await remotePage.textContent('#provenance-panel')) || '';
    if (!remoteProvenance.includes('Publish provenance') && !remoteProvenance.includes('Pages staged')) {
      throw new Error('Dashboard did not expose provenance messaging on the staged site');
    }
    await remotePage.close();

    const currentPills = await readPillTexts();
    console.log(
      JSON.stringify(
        {
          title: 'FFBayes dashboard smoke test',
          draftedPlayers: drafted,
          finalPills: currentPills,
          finalizedFiles: savedDownloads.map((item) => item.filename).sort(),
          rosterComplete: finalizedPayload.final_state.roster_complete,
        },
        null,
        2,
      ),
    );
  } finally {
    if (staticServer) {
      await new Promise((resolveClose) => staticServer.server.close(resolveClose));
    }
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
    await rm(downloadDir, { recursive: true, force: true }).catch(() => {});
  }
}

runSmoke().catch((error) => {
  console.error(error);
  process.exit(1);
});
