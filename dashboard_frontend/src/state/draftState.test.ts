import { beforeEach, describe, expect, it } from 'vitest';
import { createDraftStore, STORAGE_KEY } from './draftState';

beforeEach(() => {
  window.localStorage.clear();
});

describe('draft store', () => {
  it('marks a player taken and undoes it', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markTaken('Test Player');
    expect(store.getState().takenPlayers).toContain('Test Player');
    store.undo();
    expect(store.getState().takenPlayers).not.toContain('Test Player');
  });

  it('redo restores an undone action', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markMine('Test Player');
    store.undo();
    store.redo();
    expect(store.getState().yourPlayers).toContain('Test Player');
  });

  it('persists to and rehydrates from localStorage under the legacy key', () => {
    const store = createDraftStore({ initialPickNumber: 5 });
    store.markTaken('Test Player');
    const raw = window.localStorage.getItem(STORAGE_KEY);
    expect(raw).toBeTruthy();
    const rehydrated = createDraftStore({ initialPickNumber: 5 });
    expect(rehydrated.getState().takenPlayers).toContain('Test Player');
  });

  it('rehydrates legacy-format state written by the old dashboard', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 2,
        currentPickNumber: 12,
        draftPosition: 10,
        leagueSize: 10,
        scoringPreset: 'half_ppr',
        riskTolerance: 'medium',
        benchSlots: 6,
        rosterSpots: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, DST: 1, K: 1 },
        takenPlayers: ['Player A', 'Player B'],
        yourPlayers: ['My Pick'],
        queuePlayers: ['Queued Player'],
        history: [],
        redoHistory: [],
        pickLog: [],
        search: 'rb',
        selectedPlayer: 'Player A',
        showAllCliffs: true,
      }),
    );
    const store = createDraftStore({ initialPickNumber: 5 });
    expect(store.getState().takenPlayers).toEqual(['Player A', 'Player B']);
    expect(store.getState().yourPlayers).toEqual(['My Pick']);
    expect(store.getState().queuePlayers).toEqual(['Queued Player']);
    expect(store.getState().currentPickNumber).toBe(12);
    expect(store.getState().search).toBe('rb');
    expect(store.getState().showAllCliffs).toBe(true);
  });
});
