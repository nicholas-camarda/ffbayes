import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import { PlayerBoard } from './PlayerBoard';

const decisionRows = [
  {
    player_name: 'Alpha Runner',
    position: 'RB',
    team: 'KC',
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
    team: 'SF',
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

it('renders fixture rows in the board table', () => {
  const store = createStoreFromPayload(payload);
  render(<PlayerBoard payload={payload} store={store} />);

  const rows = document.querySelectorAll('#board-table tr[data-player-row]');
  expect(rows.length).toBe(2);
  expect(screen.getByText('Alpha Runner')).toBeInTheDocument();
  expect(screen.getByText('Beta Receiver')).toBeInTheDocument();
});

it('updates the store when the taken action is clicked', async () => {
  const user = userEvent.setup();
  const store = createStoreFromPayload(payload);
  render(<PlayerBoard payload={payload} store={store} />);

  const takenButton = document.querySelector(
    'button[data-action="taken"][data-player="Alpha Runner"]',
  );
  expect(takenButton).toBeTruthy();
  await user.click(takenButton as HTMLButtonElement);

  expect(store.getState().takenPlayers).toContain('Alpha Runner');
  const row = document.querySelector('tr[data-player-row="Alpha Runner"]');
  expect(row?.querySelector('.status-badge')?.textContent).toBe('taken');
  expect(row?.classList.contains('is-taken')).toBe(true);
});

it('selects a player when a row is clicked', async () => {
  const user = userEvent.setup();
  const store = createStoreFromPayload(payload);
  render(<PlayerBoard payload={payload} store={store} />);

  const row = document.querySelector('tr[data-player-row="Beta Receiver"]') as HTMLElement;
  await user.click(row);

  expect(store.getState().selectedPlayer).toBe('Beta Receiver');
});

it('narrows rows when the search box is used', async () => {
  const user = userEvent.setup();
  const store = createStoreFromPayload(payload);
  render(<PlayerBoard payload={payload} store={store} />);

  await user.type(screen.getByPlaceholderText('Search name, team, or position'), 'alpha');

  const rows = document.querySelectorAll('#board-table tr[data-player-row]');
  expect(rows.length).toBe(1);
  expect(rows[0]?.getAttribute('data-player-row')).toBe('Alpha Runner');
});
