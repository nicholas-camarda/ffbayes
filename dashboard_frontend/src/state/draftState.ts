export const STORAGE_KEY = 'ffbayes-dashboard-state-v2';

const DEFAULT_ROSTER_SPOTS: Record<string, number> = {
  QB: 1,
  RB: 2,
  WR: 2,
  TE: 1,
  FLEX: 1,
  DST: 1,
  K: 1,
};

export function safeLower(value: unknown): string {
  return (value || '').toString().trim().toLowerCase();
}

export interface PickLogEntry {
  pick_number: number;
  player_name: string;
  position?: string;
  team?: string;
  adp?: number;
  market_rank?: number;
  draft_score?: number;
  simple_vor_proxy?: number;
  fragility_score?: number;
  upside_score?: number;
  top_recommendation?: string;
  recommended_draft_score?: number | null;
  top_wait_candidate?: Record<string, unknown> | null;
  followed_model?: boolean;
  decision_label?: string;
}

export interface DraftSnapshot {
  currentPickNumber: number;
  takenPlayers: string[];
  yourPlayers: string[];
  queuePlayers: string[];
  pickLog: PickLogEntry[];
  selectedPlayer: string;
}

export interface DraftState {
  version: number;
  currentPickNumber: number;
  draftPosition: number;
  leagueSize: number;
  scoringPreset: string;
  riskTolerance: string;
  benchSlots: number;
  rosterSpots: Record<string, number>;
  takenPlayers: string[];
  yourPlayers: string[];
  queuePlayers: string[];
  history: DraftSnapshot[];
  redoHistory: DraftSnapshot[];
  pickLog: PickLogEntry[];
  search: string;
  selectedPlayer: string;
  showAllCliffs: boolean;
}

export interface CreateDraftStoreOptions {
  initialPickNumber?: number;
  draftPosition?: number;
  leagueSize?: number;
  scoringPreset?: string;
  riskTolerance?: string;
  benchSlots?: number;
  rosterSpots?: Record<string, number>;
  takenPlayers?: string[];
  yourPlayers?: string[];
  queuePlayers?: string[];
  selectedPlayer?: string;
  storage?: Storage;
}

export interface DraftStore {
  markTaken(playerName: string): void;
  markMine(playerName: string): void;
  toggleQueue(playerName: string): void;
  selectPlayer(playerName: string): void;
  setSearch(value: string): void;
  setScoringPreset(value: string): void;
  setRiskTolerance(value: string): void;
  setLeagueSize(value: number): void;
  setDraftPosition(value: number): void;
  setBenchSlots(value: number): void;
  undo(): void;
  redo(): void;
  getState(): Readonly<DraftState>;
  subscribe(listener: () => void): () => void;
}

export function nextPickNumber(
  currentPickNumber: number,
  draftPosition: number,
  leagueSize: number,
): number {
  const current = Math.max(1, Number(currentPickNumber) || 1);
  const draft = Math.max(1, Number(draftPosition) || 1);
  const size = Math.max(1, Number(leagueSize) || 1);
  const rounds = Math.max(1, Math.ceil(current / size) + 2);
  for (let round = 1; round <= rounds; round += 1) {
    const pick = round % 2 === 1 ? (round - 1) * size + draft : round * size - draft + 1;
    if (pick > current) {
      return pick;
    }
  }
  return current + size;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function buildDefaultState(options: CreateDraftStoreOptions): DraftState {
  return {
    version: 2,
    currentPickNumber: options.initialPickNumber ?? options.draftPosition ?? 10,
    draftPosition: options.draftPosition ?? 10,
    leagueSize: options.leagueSize ?? 10,
    scoringPreset: options.scoringPreset ?? 'half_ppr',
    riskTolerance: (options.riskTolerance ?? 'medium').toLowerCase(),
    benchSlots: options.benchSlots ?? 6,
    rosterSpots: { ...DEFAULT_ROSTER_SPOTS, ...(options.rosterSpots ?? {}) },
    takenPlayers: (options.takenPlayers ?? []).slice(),
    yourPlayers: (options.yourPlayers ?? []).slice(),
    queuePlayers: (options.queuePlayers ?? []).slice(),
    history: [],
    redoHistory: [],
    pickLog: [],
    search: '',
    selectedPlayer: options.selectedPlayer ?? '',
    showAllCliffs: false,
  };
}

function loadState(
  defaultState: DraftState,
  storage: Storage,
): DraftState {
  try {
    const parsed = JSON.parse(storage.getItem(STORAGE_KEY) || 'null') as Partial<DraftState> | null;
    if (!parsed || parsed.version !== defaultState.version) {
      return clone(defaultState);
    }
    return {
      ...clone(defaultState),
      ...parsed,
      rosterSpots: { ...clone(defaultState).rosterSpots, ...(parsed.rosterSpots || {}) },
      takenPlayers: Array.isArray(parsed.takenPlayers)
        ? parsed.takenPlayers
        : clone(defaultState).takenPlayers,
      yourPlayers: Array.isArray(parsed.yourPlayers)
        ? parsed.yourPlayers
        : clone(defaultState).yourPlayers,
      queuePlayers: Array.isArray(parsed.queuePlayers) ? parsed.queuePlayers : [],
      history: Array.isArray(parsed.history) ? parsed.history : [],
      redoHistory: Array.isArray(parsed.redoHistory) ? parsed.redoHistory : [],
      pickLog: Array.isArray(parsed.pickLog) ? parsed.pickLog : [],
      showAllCliffs: Boolean(parsed.showAllCliffs),
    };
  } catch {
    return clone(defaultState);
  }
}

function captureDraftSnapshot(state: DraftState): DraftSnapshot {
  return {
    currentPickNumber: state.currentPickNumber,
    takenPlayers: state.takenPlayers.slice(),
    yourPlayers: state.yourPlayers.slice(),
    queuePlayers: state.queuePlayers.slice(),
    pickLog: state.pickLog.slice(),
    selectedPlayer: state.selectedPlayer,
  };
}

function pushSnapshot(stack: DraftSnapshot[], snapshot: DraftSnapshot): void {
  stack.push(snapshot);
  if (stack.length > 60) {
    stack.shift();
  }
}

function restoreDraftSnapshot(state: DraftState, snapshot: DraftSnapshot | undefined): void {
  if (!snapshot) {
    return;
  }
  state.currentPickNumber = snapshot.currentPickNumber;
  state.takenPlayers = snapshot.takenPlayers;
  state.yourPlayers = snapshot.yourPlayers;
  state.queuePlayers = snapshot.queuePlayers;
  state.pickLog = snapshot.pickLog || [];
  state.selectedPlayer = snapshot.selectedPlayer;
}

export function createDraftStore(options: CreateDraftStoreOptions = {}): DraftStore {
  const storage = options.storage ?? window.localStorage;
  const defaultState = buildDefaultState(options);
  const state = loadState(defaultState, storage);
  const listeners = new Set<() => void>();

  function notify(): void {
    for (const listener of listeners) {
      listener();
    }
  }

  function persistState(): void {
    storage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  function commit(): void {
    persistState();
    notify();
  }

  function pushHistory(): void {
    pushSnapshot(state.history, captureDraftSnapshot(state));
    state.redoHistory = [];
  }

  function applyQueue(playerName: string): void {
    const normalized = safeLower(playerName);
    state.queuePlayers = state.queuePlayers.some((item) => safeLower(item) === normalized)
      ? state.queuePlayers.filter((item) => safeLower(item) !== normalized)
      : [...state.queuePlayers, playerName];
  }

  function applyTaken(playerName: string): void {
    const normalized = safeLower(playerName);
    const alreadyTaken = state.takenPlayers.some((item) => safeLower(item) === normalized);
    state.takenPlayers = alreadyTaken
      ? state.takenPlayers.filter((item) => safeLower(item) !== normalized)
      : [...state.takenPlayers, playerName];
    if (alreadyTaken) {
      state.yourPlayers = state.yourPlayers.filter((item) => safeLower(item) !== normalized);
    } else {
      state.queuePlayers = state.queuePlayers.filter((item) => safeLower(item) !== normalized);
    }
  }

  function applyMine(playerName: string): void {
    const normalized = safeLower(playerName);
    const alreadyMine = state.yourPlayers.some((item) => safeLower(item) === normalized);
    state.yourPlayers = alreadyMine
      ? state.yourPlayers.filter((item) => safeLower(item) !== normalized)
      : [...state.yourPlayers, playerName];
    state.takenPlayers = alreadyMine
      ? state.takenPlayers.filter((item) => safeLower(item) !== normalized)
      : Array.from(new Set([...state.takenPlayers, playerName]));
    state.queuePlayers = state.queuePlayers.filter((item) => safeLower(item) !== normalized);
    state.pickLog = (state.pickLog || []).filter((entry) =>
      state.yourPlayers.some((item) => safeLower(item) === safeLower(entry.player_name)),
    );
  }

  return {
    markTaken(playerName: string): void {
      pushHistory();
      applyTaken(playerName);
      commit();
    },

    markMine(playerName: string): void {
      pushHistory();
      applyMine(playerName);
      commit();
    },

    toggleQueue(playerName: string): void {
      pushHistory();
      applyQueue(playerName);
      commit();
    },

    selectPlayer(playerName: string): void {
      state.selectedPlayer = playerName;
      commit();
    },

    setSearch(value: string): void {
      state.search = value;
      commit();
    },

    setScoringPreset(value: string): void {
      state.scoringPreset = value;
      commit();
    },

    setRiskTolerance(value: string): void {
      state.riskTolerance = safeLower(value);
      commit();
    },

    setLeagueSize(value: number): void {
      state.leagueSize = Math.max(2, Number(value) || 10);
      state.draftPosition = Math.min(state.draftPosition, state.leagueSize);
      commit();
    },

    setDraftPosition(value: number): void {
      state.draftPosition = Math.max(1, Math.min(Number(value) || 1, state.leagueSize));
      commit();
    },

    setBenchSlots(value: number): void {
      state.benchSlots = Math.max(0, Number(value) || 0);
      commit();
    },

    undo(): void {
      const snapshot = state.history.pop();
      if (!snapshot) {
        return;
      }
      pushSnapshot(state.redoHistory, captureDraftSnapshot(state));
      restoreDraftSnapshot(state, snapshot);
      commit();
    },

    redo(): void {
      const snapshot = state.redoHistory.pop();
      if (!snapshot) {
        return;
      }
      pushSnapshot(state.history, captureDraftSnapshot(state));
      restoreDraftSnapshot(state, snapshot);
      commit();
    },

    getState(): Readonly<DraftState> {
      return state;
    },

    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
