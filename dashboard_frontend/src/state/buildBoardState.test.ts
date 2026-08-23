import { beforeEach, expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { buildBoardState } from './buildBoardState';
import { createDraftStore } from './draftState';

const decisionRows = [
  {
    player_name: 'Beta RB',
    position: 'RB',
    proj_points_mean: 150,
    proj_points_floor: 130,
    proj_points_ceiling: 170,
    adp: 20,
    adp_std: 2,
    uncertainty_score: 0.1,
    posterior_prob_beats_replacement: 0.6,
    fragility_score: 0.2,
    market_gap: 0,
  },
  {
    player_name: 'Alpha RB',
    position: 'RB',
    proj_points_mean: 200,
    proj_points_floor: 175,
    proj_points_ceiling: 235,
    adp: 5,
    adp_std: 2,
    uncertainty_score: 0.1,
    posterior_prob_beats_replacement: 0.8,
    fragility_score: 0.2,
    market_gap: 0,
  },
];

function buildPayload(rows = decisionRows): DashboardPayload {
  const base = fixture as unknown as DashboardPayload;
  return {
    ...base,
    decision_table: rows,
    scoring_presets: {
      ...base.scoring_presets,
      half_ppr: {
        ...(base.scoring_presets?.half_ppr as object),
        available: true,
        decision_table: rows,
      },
    },
  } as unknown as DashboardPayload;
}

beforeEach(() => {
  window.localStorage.clear();
});

it('sorts the board by recomputed score and preserves signed replacement deltas', () => {
  const payload = buildPayload();
  const store = createDraftStore({ payload });
  const board = buildBoardState(payload, store.getState());

  expect(board.rows.map((row) => row.player_name)).toEqual(['Alpha RB', 'Beta RB']);
  expect(board.rows.map((row) => row.draft_rank)).toEqual([1, 2]);
  expect(board.rows[0]?.replacement_delta).toBe(50);
  expect(board.rows[1]?.replacement_delta).toBe(0);
  expect(board.rows[0]?.availability_to_next_pick).toBeLessThan(
    board.rows[1]?.availability_to_next_pick as number,
  );
});

it('is invariant to source row order and excludes taken players from recommendations', () => {
  const forwardPayload = buildPayload();
  const reversePayload = buildPayload([...decisionRows].reverse());
  const forwardStore = createDraftStore({ payload: forwardPayload });
  const reverseStore = createDraftStore({ payload: reversePayload });

  const forward = buildBoardState(forwardPayload, forwardStore.getState());
  const reverse = buildBoardState(reversePayload, reverseStore.getState());

  expect(reverse.rows.map((row) => row.player_name)).toEqual(
    forward.rows.map((row) => row.player_name),
  );
  expect(reverse.rows.map((row) => row.draft_score)).toEqual(
    forward.rows.map((row) => row.draft_score),
  );

  const takenStore = createDraftStore({
    payload: forwardPayload,
    takenPlayers: ['Alpha RB'],
  });
  const takenBoard = buildBoardState(forwardPayload, takenStore.getState());
  expect(takenBoard.rows.find((row) => row.player_name === 'Alpha RB')?.status).toBe('taken');
  expect(takenBoard.availableRows.map((row) => row.player_name)).not.toContain('Alpha RB');
  expect(takenBoard.pickNow.map((row) => row.player_name)).not.toContain('Alpha RB');
});
