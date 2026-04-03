import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');
const repoRoot = resolve(__dirname, '..');
const siteIndexPath = resolve(repoRoot, 'site', 'index.html');

async function runSmoke() {
  const url = pathToFileURL(siteIndexPath).href;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

  const selectors = {
    leagueSize: '#league-size',
    draftPosition: '#draft-position',
    currentPickNumber: '#current-pick-number',
    riskTolerance: '#risk-tolerance',
    search: '#player-search',
    resetButton: '#reset-button',
    statusPills: '#status-pills .pill',
    primaryName: '#primary-card .hero-name',
    primarySummary: '#primary-card .summary-box',
    boardRows: '#board-table tr[data-player-row]',
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

  const readPillTexts = async () => page.locator(selectors.statusPills).allTextContents();
  const readText = async (selector) => page.locator(selector).first().textContent();
  const getInputValue = async (selector) => page.locator(selector).inputValue();
  const getSelectValue = async (selector) => page.locator(selector).inputValue();

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

  const stubConfirm = async (returnValue) => {
    await page.evaluate((nextValue) => {
      window.__ffbayesConfirmCalls = [];
      window.confirm = (message) => {
        window.__ffbayesConfirmCalls.push(message);
        return nextValue;
      };
    }, returnValue);
  };

  const readConfirmCalls = async () => page.evaluate(() => window.__ffbayesConfirmCalls || []);

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    if ((await readText('h1')) !== 'FFBayes Draft War Room') {
      throw new Error('Dashboard did not load the expected title');
    }

    await setNumber(selectors.leagueSize, 3);
    await setNumber(selectors.draftPosition, 2);
    await setNumber(selectors.currentPickNumber, 1);
    await setSelect(selectors.riskTolerance, 'high');

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

    await clickAction('taken', primaryBeforeTaken);
    await waitForPillText('Current pick: 2');
    const primaryAfterTaken = await readText(selectors.primaryName);
    if (primaryAfterTaken === primaryBeforeTaken) {
      throw new Error('Taken did not advance the draft to a new recommendation');
    }

    await clickAction('mine', primaryAfterTaken);
    await waitForPillText('Yours: 1');
    const rosterAfterMine = await page.locator(selectors.rosterChips).allTextContents();
    if (!rosterAfterMine.some((text) => text.includes(primaryAfterTaken))) {
      throw new Error('Mine did not add the selected player to the roster');
    }

    const stickyCandidate = await findFirstAvailablePlayerByPosition('WR');
    if (!stickyCandidate) {
      throw new Error('Could not find a WR candidate for inspector testing');
    }

    await clickBoardRow(stickyCandidate);
    await waitForInspectorText(stickyCandidate);

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

    await stubConfirm(false);
    await page.evaluate(() => document.getElementById('reset-button')?.click());
    await page.waitForTimeout(125);
    const cancelConfirmCalls = await readConfirmCalls();
    if (cancelConfirmCalls.length !== 1 || !cancelConfirmCalls[0].includes('Clear taken players')) {
      throw new Error(`Reset cancel did not trigger the expected confirmation prompt: ${cancelConfirmCalls.join(' | ')}`);
    }

    const afterCancelPills = await readPillTexts();
    if (!afterCancelPills.includes('League: 3-team') || !afterCancelPills.includes('Draft slot: 2')) {
      throw new Error('Reset cancel should preserve league controls');
    }

    await stubConfirm(true);
    await page.evaluate(() => document.getElementById('reset-button')?.click());
    await page.waitForTimeout(150);
    const acceptConfirmCalls = await readConfirmCalls();
    if (acceptConfirmCalls.length !== 1 || !acceptConfirmCalls[0].includes('Clear taken players')) {
      throw new Error(`Reset confirm did not trigger the expected confirmation prompt: ${acceptConfirmCalls.join(' | ')}`);
    }

    const afterResetPills = await readPillTexts();
    if (!afterResetPills.includes('League: 3-team') || !afterResetPills.includes('Draft slot: 2')) {
      throw new Error('Reset confirm should preserve league controls');
    }
    if (!afterResetPills.includes('Taken: 0') || !afterResetPills.includes('Yours: 0')) {
      throw new Error(`Reset should clear draft progress, got ${afterResetPills.join(' | ')}`);
    }
    if ((await getInputValue(selectors.leagueSize)) !== '3') {
      throw new Error('League size control was not preserved after reset');
    }
    if ((await getInputValue(selectors.draftPosition)) !== '2') {
      throw new Error('Draft position control was not preserved after reset');
    }
    if ((await getSelectValue(selectors.riskTolerance)) !== 'high') {
      throw new Error('Risk tolerance control was not preserved after reset');
    }

    const rosterPlan = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'RB'];
    const drafted = [];
    for (const position of rosterPlan) {
      const playerName = await findFirstAvailablePlayerByPosition(position);
      if (!playerName) {
        throw new Error(`Could not find an available ${position} for FLEX test`);
      }
      drafted.push(playerName);
      await clickAction('mine', playerName);
    }

    const rosterNeedTexts = await page.locator(selectors.rosterNeeds).evaluateAll((nodes) => nodes.map((node) => ({
      label: node.querySelector('.label')?.textContent?.trim(),
      value: node.querySelector('.value')?.textContent?.trim(),
    })));
    const flexNeed = rosterNeedTexts.find((item) => item.label === 'FLEX need')?.value;
    if (flexNeed !== '0') {
      throw new Error(`Expected FLEX need to fall to 0, got ${flexNeed}`);
    }

    const rosterNames = await page.locator(selectors.rosterChips).allTextContents();
    if (rosterNames.length < drafted.length) {
      throw new Error('Roster did not retain the drafted players for FLEX coverage');
    }

    console.log(
      JSON.stringify(
        {
          title: 'FFBayes dashboard smoke test',
          draftedPlayers: drafted,
          finalPills: afterResetPills,
          flexNeed,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close().catch(() => {});
  }
}

runSmoke().catch((error) => {
  console.error(error);
  process.exit(1);
});
