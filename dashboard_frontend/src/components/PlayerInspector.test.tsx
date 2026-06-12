import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import { PlayerBoard } from './PlayerBoard';
import { PlayerInspector } from './PlayerInspector';

const decisionRows = [
  {
    player_name: 'Alpha Runner',
    position: 'RB',
    draft_rank: 1,
    simple_vor_rank: 1,
    draft_score: 2.5,
    simple_vor_proxy: 15.0,
    availability_to_next_pick: 0.35,
    expected_regret: 0.4,
    upside_score: 0.8,
    fragility_score: 0.2,
    proj_points_mean: 180,
    status: 'available',
  },
  {
    player_name: 'Beta Receiver',
    position: 'WR',
    draft_rank: 2,
    simple_vor_rank: 2,
    draft_score: 2.1,
    simple_vor_proxy: 12.0,
    availability_to_next_pick: 0.75,
    expected_regret: 0.15,
    upside_score: 0.65,
    fragility_score: 0.25,
    proj_points_mean: 165,
    status: 'available',
  },
];

const payload = {
  ...(fixture as unknown as DashboardPayload),
  decision_table: decisionRows,
  scoring_presets: {
    ...(fixture as unknown as DashboardPayload).scoring_presets,
    half_ppr: {
      ...((fixture as unknown as DashboardPayload).scoring_presets?.half_ppr as object),
      decision_table: decisionRows,
    },
  },
} as unknown as DashboardPayload;

beforeEach(() => {
  window.localStorage.clear();
});

function renderBoardAndInspector(store = createStoreFromPayload(payload)) {
  render(
    <>
      <PlayerBoard payload={payload} store={store} />
      <PlayerInspector payload={payload} store={store} />
    </>,
  );
  return store;
}

it('shows the default selected player from the board state', () => {
  renderBoardAndInspector();
  const inspector = document.getElementById('player-inspector');
  expect(inspector?.textContent).toContain('Alpha Runner');
  expect(inspector?.querySelector('.hero-name')?.textContent).toBe('Alpha Runner');
});

it('updates the inspector when a board row is clicked', async () => {
  const user = userEvent.setup();
  renderBoardAndInspector();

  const row = document.querySelector('tr[data-player-row="Beta Receiver"]') as HTMLElement;
  await user.click(row);

  const inspector = document.getElementById('player-inspector');
  expect(inspector?.querySelector('.hero-name')?.textContent).toBe('Beta Receiver');
});
