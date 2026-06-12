import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import fixture from '../../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../../payload/load';
import { createStoreFromPayload } from '../../state/createStoreFromPayload';
import { ComparativeExplainer } from './ComparativeExplainer';
import { PositionalCliffs } from './PositionalCliffs';
import { TimingFrontier } from './TimingFrontier';

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
  {
    player_name: 'Gamma Back',
    position: 'RB',
    draft_rank: 3,
    simple_vor_rank: 3,
    draft_score: 1.8,
    simple_vor_proxy: 10.0,
    availability_to_next_pick: 0.8,
    expected_regret: 0.1,
    upside_score: 0.6,
    fragility_score: 0.3,
    proj_points_mean: 150,
    status: 'available',
  },
  {
    player_name: 'Delta Tight',
    position: 'TE',
    draft_rank: 4,
    simple_vor_rank: 5,
    draft_score: 1.5,
    simple_vor_proxy: 8.0,
    availability_to_next_pick: 0.6,
    expected_regret: 0.2,
    upside_score: 0.55,
    fragility_score: 0.35,
    proj_points_mean: 120,
    status: 'available',
  },
];

const basePayload = {
  ...(fixture as unknown as DashboardPayload),
  decision_table: decisionRows,
  scoring_presets: {
    ...(fixture as unknown as DashboardPayload).scoring_presets,
    half_ppr: {
      ...((fixture as unknown as DashboardPayload).scoring_presets?.half_ppr as object),
      decision_table: decisionRows,
    },
  },
  live_state: {
    available: true,
    status: 'available',
    pick_now: { player_name: 'Alpha Runner', position: 'RB' },
    fallbacks: [{ player_name: 'Beta Receiver', position: 'WR' }],
    can_wait: [{ player_name: 'Gamma Back', position: 'RB' }],
  },
} as unknown as DashboardPayload;

beforeEach(() => {
  window.localStorage.clear();
});

function renderWarRoom(payload: DashboardPayload, store = createStoreFromPayload(payload)) {
  render(
    <>
      <TimingFrontier payload={payload} store={store} />
      <PositionalCliffs payload={payload} store={store} />
      <ComparativeExplainer payload={payload} store={store} />
    </>,
  );
  return store;
}

it('renders timing frontier candidates from the fixture board state', () => {
  renderWarRoom(basePayload);
  const frontier = document.getElementById('timing-frontier');
  expect(frontier).toBeTruthy();
  expect(frontier?.querySelectorAll('[data-frontier-player]').length).toBeGreaterThan(0);
  expect(frontier?.textContent).toContain('Alpha Runner');
  expect(frontier?.textContent).toContain('Can I safely wait');
});

it('reflects the derived current pick number in the timing frontier', () => {
  const store = renderWarRoom(basePayload);
  const frontier = document.getElementById('timing-frontier');
  expect(frontier?.getAttribute('data-current-pick')).toBe('1');

  act(() => {
    store.markTaken('Alpha Runner');
  });
  expect(frontier?.getAttribute('data-current-pick')).toBe('2');
  expect(frontier?.textContent).toMatch(/Pick 2/);
});

it('renders positional cliff player chips from the fixture board state', () => {
  renderWarRoom(basePayload);
  const cliffs = document.getElementById('positional-cliffs');
  expect(cliffs).toBeTruthy();
  expect(cliffs?.querySelectorAll('[data-cliff-player]').length).toBeGreaterThan(0);
  expect(cliffs?.textContent).toContain('RB');
});

it('renders the comparative explainer for the selected player', () => {
  renderWarRoom(basePayload);
  const explainer = document.getElementById('comparative-explainer');
  expect(explainer?.textContent).toContain('Board value score');
  expect(explainer?.textContent).toContain('Simple VOR proxy');
  expect(explainer?.textContent).toContain('Alpha Runner');
});

it('degrades timing frontier via SectionGate when unavailable', () => {
  const payload = {
    ...basePayload,
    war_room_visuals: {
      ...(basePayload.war_room_visuals as object),
      timing_frontier: {
        available: false,
        status: 'unavailable',
        reason: 'Timing frontier inputs are incomplete for this payload.',
      },
    },
  } as unknown as DashboardPayload;

  renderWarRoom(payload);
  expect(screen.getByText(/Timing frontier inputs are incomplete/i)).toBeInTheDocument();
  expect(document.querySelector('#timing-frontier [data-frontier-player]')).toBeNull();
});

it('degrades positional cliffs via SectionGate when unavailable', () => {
  const payload = {
    ...basePayload,
    war_room_visuals: {
      ...(basePayload.war_room_visuals as object),
      positional_cliffs: {
        available: false,
        status: 'unavailable',
        reason: 'Positional cliff inputs are incomplete for this payload.',
      },
    },
  } as unknown as DashboardPayload;

  renderWarRoom(payload);
  expect(screen.getByText(/Positional cliff inputs are incomplete/i)).toBeInTheDocument();
  expect(document.querySelector('#positional-cliffs [data-cliff-player]')).toBeNull();
});

it('degrades comparative explainer via SectionGate when unavailable', () => {
  const payload = {
    ...basePayload,
    war_room_visuals: {
      ...(basePayload.war_room_visuals as object),
      comparative_explainer: {
        available: false,
        status: 'unavailable',
        reason: 'Comparative explainer is unavailable for this board.',
      },
    },
  } as unknown as DashboardPayload;

  renderWarRoom(payload);
  expect(screen.getByText(/Comparative explainer is unavailable/i)).toBeInTheDocument();
  expect(document.querySelector('#comparative-explainer table')).toBeNull();
});
