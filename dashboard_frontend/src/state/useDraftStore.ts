import { useSyncExternalStore } from 'react';
import type { DraftState, DraftStore } from './draftState';

const versions = new WeakMap<DraftStore, number>();

function getVersion(store: DraftStore): number {
  return versions.get(store) ?? 0;
}

function subscribeWithVersion(store: DraftStore, listener: () => void): () => void {
  return store.subscribe(() => {
    versions.set(store, getVersion(store) + 1);
    listener();
  });
}

export function useDraftStore(store: DraftStore): DraftState {
  useSyncExternalStore(
    (listener) => subscribeWithVersion(store, listener),
    () => getVersion(store),
    () => getVersion(store),
  );
  return store.getState();
}
