import { useSyncExternalStore } from 'react';
import type { DraftState, DraftStore } from './draftState';

export function useDraftStore(store: DraftStore): DraftState {
  return useSyncExternalStore(
    (listener) => store.subscribe(listener),
    () => store.getState(),
    () => store.getState(),
  );
}
