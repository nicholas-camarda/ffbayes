import type { DashboardPayload } from '../payload/load';
import { buildBoardState } from './buildBoardState';
import type { DraftStore } from './draftState';
import { useDraftStore } from './useDraftStore';

export function useBoardState(payload: DashboardPayload, store: DraftStore) {
  const state = useDraftStore(store);
  return buildBoardState(payload, state);
}
