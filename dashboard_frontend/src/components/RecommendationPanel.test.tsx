import { act, render } from '@testing-library/react';
import { expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import {
  RecommendationFallbacks,
  RecommendationPrimary,
  RecommendationWait,
} from './RecommendationPanel';

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

function renderPanels(payload: DashboardPayload, store = createStoreFromPayload(payload)) {
  render(
    <>
      <RecommendationPrimary payload={payload} store={store} />
      <RecommendationFallbacks payload={payload} store={store} />
      <RecommendationWait payload={payload} store={store} />
    </>,
  );
  return store;
}

it('renders the top recommendation from the fixture', () => {
  renderPanels(basePayload);
  const primaryCard = document.querySelector('#primary-card');
  expect(primaryCard?.querySelector('.hero-name')?.textContent).toBe('Alpha Runner');
  expect(primaryCard?.textContent).toMatch(/Board value/);
});

it('renders fallback and can-wait lanes from live_state', () => {
  renderPanels(basePayload);
  expect(document.getElementById('fallback-list')?.textContent).toContain('Beta Receiver');
  expect(document.getElementById('wait-list')?.textContent).toContain('Gamma Back');
});

it('filters out the top player after they are marked taken', () => {
  const store = renderPanels(basePayload);
  act(() => {
    store.markTaken('Alpha Runner');
  });
  expect(document.querySelector('#primary-card .hero-name')?.textContent).toBe('Beta Receiver');
  expect(document.querySelector('#primary-card .hero-name')?.textContent).not.toBe('Alpha Runner');
});
