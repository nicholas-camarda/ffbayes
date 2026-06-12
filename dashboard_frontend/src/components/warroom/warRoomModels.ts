import {
  buildPlayerSummary,
  formatNumber,
  type BoardState,
  type DecisionRow,
} from '../../state/buildBoardState';
import { safeLower } from '../../state/draftState';
import type { GatedSection } from '../SectionGate';

export interface WarRoomVisualsConfig {
  schema_version?: string;
  contextual?: { key?: string; label?: string };
  baseline?: { key?: string; label?: string };
  timing_frontier?: TimingFrontierConfig;
  positional_cliffs?: PositionalCliffsConfig;
  comparative_explainer?: ComparativeExplainerConfig;
}

export interface TimingFrontierConfig extends GatedSection {
  question?: string;
  contextual_label?: string;
  baseline_label?: string;
  current_pick_number?: number | null;
  next_pick_number?: number | null;
  candidates?: unknown[];
}

export interface PositionalCliffsConfig extends GatedSection {
  question?: string;
  default_positions?: string[];
  positions?: unknown[];
}

export interface ComparativeExplainerConfig extends GatedSection {
  question?: string;
  contextual_label?: string;
  baseline_label?: string;
  top_disagreements?: unknown[];
}

export interface VisualRow {
  player_name: string;
  position: string;
  status?: string;
  lane: string;
  contextual_score: number;
  baseline_score: number;
  contextual_rank: number;
  baseline_rank: number;
  rank_gap: number;
  timing_survival: number;
  wait_regret: number;
  timing_pressure: number;
  contextual_label: string;
  baseline_label: string;
  rationale: string;
}

export interface TimingFrontierCandidate extends VisualRow {
  survival_percent: number;
  regret_percent: number;
}

export interface CliffPlayer {
  row: DecisionRow;
  cliffDistance: number;
  isCliffEdge: boolean;
}

export interface CliffGroup {
  position: string;
  cliffStrength: number;
  strongestCliffIndex: number;
  strongestCliffDistance: number;
  players: CliffPlayer[];
}

export function getWarRoomVisualsConfig(
  warRoomVisuals: WarRoomVisualsConfig | undefined,
): Required<
  Pick<WarRoomVisualsConfig, 'timing_frontier' | 'positional_cliffs' | 'comparative_explainer'>
> & {
  contextual: { key: string; label: string };
  baseline: { key: string; label: string };
} {
  const raw = warRoomVisuals ?? {};
  const fallbackContextual = raw.contextual?.label || 'Board value score';
  const fallbackBaseline = raw.baseline?.label || 'Simple VOR proxy';
  return {
    contextual: {
      key: raw.contextual?.key || 'contextual_score',
      label: fallbackContextual,
    },
    baseline: {
      key: raw.baseline?.key || 'baseline_score',
      label: fallbackBaseline,
    },
    timing_frontier: raw.timing_frontier ?? {
      available: true,
      status: 'available',
      question: 'Can I safely wait on this value, or do I need to pick now?',
      reason: '',
      contextual_label: fallbackContextual,
      baseline_label: fallbackBaseline,
      candidates: [],
    },
    positional_cliffs: raw.positional_cliffs ?? {
      available: true,
      status: 'available',
      question: 'Which positions are about to fall off if I wait?',
      reason: '',
      default_positions: [],
      positions: [],
    },
    comparative_explainer: raw.comparative_explainer ?? {
      available: true,
      status: 'available',
      question: 'Why does the contextual board differ from the baseline value view?',
      reason: '',
      contextual_label: fallbackContextual,
      baseline_label: fallbackBaseline,
      top_disagreements: [],
    },
  };
}

export function clamp01(value: unknown): number {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

export function formatSignedNumber(value: unknown): string {
  const numeric = Number(value || 0);
  return `${numeric > 0 ? '+' : ''}${formatNumber(numeric)}`;
}

function uniquePlayerRows(rows: DecisionRow[]): DecisionRow[] {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = safeLower(row?.player_name);
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

export function normalizeVisualRow(row: DecisionRow, config: ReturnType<typeof getWarRoomVisualsConfig>): VisualRow {
  return {
    player_name: row.player_name,
    position: row.position,
    status: row.status,
    lane: String(row.recommendation_lane || 'watch'),
    contextual_score: Number(row.draft_score || row.board_value_score || 0),
    baseline_score: Number(row.simple_vor_proxy || row.replacement_delta || 0),
    contextual_rank: Number(row.draft_rank || 0),
    baseline_rank: Number(row.simple_vor_rank || 0),
    rank_gap: Number(row.rank_gap_vs_vor || 0),
    timing_survival: clamp01(row.availability_to_next_pick),
    wait_regret: Math.max(0, Number(row.expected_regret || 0)),
    timing_pressure: clamp01(row.position_run_risk),
    contextual_label: config.contextual.label,
    baseline_label: config.baseline.label,
    rationale: String(row.rationale_live || buildPlayerSummary(row)),
  };
}

export function buildTimingFrontierModel(
  boardState: BoardState,
  config: TimingFrontierConfig,
  visualsConfig: ReturnType<typeof getWarRoomVisualsConfig>,
):
  | { available: false; status: string; reason: string }
  | {
      available: true;
      status: string;
      question?: string;
      reason: string;
      candidates: TimingFrontierCandidate[];
    } {
  if (config.available === false) {
    return {
      available: false,
      status: config.status || 'unavailable',
      reason: config.reason || 'Timing frontier is unavailable.',
    };
  }
  const candidates = uniquePlayerRows([
    ...boardState.pickNow,
    ...boardState.fallbacks.slice(0, 3),
    ...boardState.canWait.slice(0, 4),
  ]).map((row) => normalizeVisualRow(row, visualsConfig));
  if (!candidates.length) {
    return {
      available: false,
      status: 'unavailable',
      reason: config.reason || 'No current timing candidates are available.',
    };
  }
  const regrets = candidates.map((row) => row.wait_regret);
  const minRegret = Math.min(...regrets);
  const maxRegret = Math.max(...regrets);
  const normalizeRegret = (value: number): number => {
    if (!Number.isFinite(value)) {
      return 0.5;
    }
    if (Math.abs(maxRegret - minRegret) < 1e-9) {
      return 0.5;
    }
    return (value - minRegret) / (maxRegret - minRegret);
  };
  return {
    available: true,
    status: config.status || 'available',
    question: config.question,
    reason: config.reason || '',
    candidates: candidates.map((row) => ({
      ...row,
      survival_percent: clamp01(row.timing_survival),
      regret_percent: normalizeRegret(row.wait_regret),
    })),
  };
}

export function buildCliffGroups(boardState: BoardState): CliffGroup[] {
  const groups = new Map<string, DecisionRow[]>();
  (boardState.availableRows || []).forEach((row) => {
    if (!groups.has(row.position)) {
      groups.set(row.position, []);
    }
    groups.get(row.position)?.push(row);
  });
  return Array.from(groups.entries())
    .map(([position, rows]) => {
      const ordered = [...rows]
        .sort(
          (a, b) =>
            Number(b.proj_points_mean) - Number(a.proj_points_mean)
            || Number(b.draft_score) - Number(a.draft_score),
        )
        .slice(0, 7);
      const diffs = ordered.map((row, index) => {
        const next = ordered[index + 1];
        return next
          ? Math.max(0, Number(row.proj_points_mean || 0) - Number(next.proj_points_mean || 0))
          : 0;
      });
      const sortedDiffs = [...diffs].filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      const threshold =
        sortedDiffs.length > 3
          ? sortedDiffs[Math.min(sortedDiffs.length - 1, Math.floor(sortedDiffs.length * 0.75))]
          : Math.max(...sortedDiffs, 0);
      const players = ordered.map((row, index) => ({
        row,
        cliffDistance: Number(diffs[index] || 0),
        isCliffEdge: Number(diffs[index] || 0) >= Number(threshold || 0) && Number(diffs[index] || 0) > 0,
      }));
      const strongest = [...players].sort(
        (a, b) => Number(b.cliffDistance) - Number(a.cliffDistance),
      )[0] ?? null;
      const strongestIndex = strongest
        ? players.findIndex((player) => player.row.player_name === strongest.row.player_name)
        : -1;
      return {
        position,
        cliffStrength: Number(strongest?.cliffDistance || 0),
        strongestCliffIndex: strongestIndex,
        strongestCliffDistance: Number(strongest?.cliffDistance || 0),
        players,
      };
    })
    .sort((a, b) => Number(b.cliffStrength) - Number(a.cliffStrength));
}

export function deriveRelevantCliffPositions(
  boardState: BoardState,
  groups: CliffGroup[],
  config: PositionalCliffsConfig,
): string[] {
  const preferred = [
    ...(Array.isArray(config.default_positions) ? config.default_positions : []),
    boardState.selectedRow?.position,
    ...boardState.pickNow.map((row) => row.position),
    ...boardState.fallbacks.map((row) => row.position),
    ...boardState.canWait.map((row) => row.position),
  ]
    .map((value) => (value || '').toString().toUpperCase())
    .filter(Boolean);
  const seen = new Set<string>();
  const filtered = preferred.filter((value) => {
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return groups.some((group) => group.position === value);
  });
  return filtered.length ? filtered : groups.slice(0, 3).map((group) => group.position);
}

export function buildComparativeModel(
  boardState: BoardState,
  row: DecisionRow | null | undefined,
  config: ComparativeExplainerConfig,
  visualsConfig: ReturnType<typeof getWarRoomVisualsConfig>,
):
  | { available: false; status: string; reason: string }
  | {
      available: true;
      status: string;
      reason: string;
      headline: string;
      row: VisualRow;
      peers: VisualRow[];
      contextualLabel: string;
      baselineLabel: string;
    } {
  if (!row) {
    return {
      available: false,
      status: 'unavailable',
      reason: 'Select a player to inspect the contextual-versus-baseline comparison.',
    };
  }
  const visualRow = normalizeVisualRow(row, visualsConfig);
  const peerRows = uniquePlayerRows(
    (boardState.availableRows || [])
      .filter((candidate) => candidate.player_name !== row.player_name)
      .sort(
        (a, b) =>
          Math.abs(Number(b.rank_gap_vs_vor || 0)) - Math.abs(Number(a.rank_gap_vs_vor || 0)),
      )
      .slice(0, 4),
  ).map((candidate) => normalizeVisualRow(candidate, visualsConfig));
  const status =
    config.available === false
      ? 'unavailable'
      : config.status || (config.reason ? 'degraded' : 'available');
  const available = status !== 'unavailable';
  if (!available) {
    return {
      available: false,
      status,
      reason: config.reason || 'No contextual-versus-baseline explainer is available for this player.',
    };
  }
  let headline = `${config.contextual_label || visualRow.contextual_label} and ${config.baseline_label || visualRow.baseline_label} largely agree on this player.`;
  if (visualRow.rank_gap > 0) {
    headline = `${config.contextual_label || visualRow.contextual_label} likes ${row.player_name} ${Math.abs(visualRow.rank_gap)} rank${Math.abs(visualRow.rank_gap) === 1 ? '' : 's'} more than ${config.baseline_label || visualRow.baseline_label}.`;
  } else if (visualRow.rank_gap < 0) {
    headline = `${config.baseline_label || visualRow.baseline_label} likes ${row.player_name} ${Math.abs(visualRow.rank_gap)} rank${Math.abs(visualRow.rank_gap) === 1 ? '' : 's'} more than ${config.contextual_label || visualRow.contextual_label}.`;
  }
  return {
    available: true,
    status,
    reason: config.reason || '',
    headline,
    row: visualRow,
    peers: peerRows,
    contextualLabel: config.contextual_label || visualRow.contextual_label,
    baselineLabel: config.baseline_label || visualRow.baseline_label,
  };
}
