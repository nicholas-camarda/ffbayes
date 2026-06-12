import type { DashboardPayload } from '../payload/load';
import { nextPickNumber, safeLower, type DraftState } from './draftState';

const POSITION_KEYS = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K'] as const;
const FANTASY_DRAFT_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'DST', 'K'] as const;
const OFFENSIVE_POSITIONS = ['QB', 'RB', 'WR', 'TE'] as const;
const LATE_SPECIALIST_BUFFER_ROUNDS = 2;
const CONSERVATIVE_WAIT_SURVIVAL_THRESHOLD = 0.74;
const CONSERVATIVE_WAIT_LINEUP_LOSS_THRESHOLD = 0.28;
const SECONDARY_QB_TE_VALUE_EDGE = 0.45;

export interface DecisionRow {
  player_name: string;
  position: string;
  status?: string;
  draft_score?: number;
  simple_vor_rank?: number;
  availability_to_next_pick?: number;
  expected_regret?: number;
  position_run_risk?: number;
  roster_fit_score?: number;
  rationale_live?: string;
  [key: string]: unknown;
}

export interface BoardState {
  rows: DecisionRow[];
  availableRows: DecisionRow[];
  nextPick: number;
  rosterCounts: Record<string, number>;
  rosterNeed: Record<string, number>;
  pickNow: DecisionRow[];
  fallbacks: DecisionRow[];
  canWait: DecisionRow[];
  selectedRow: DecisionRow | null;
}

export function formatNumber(value: unknown, digits = 2): string {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : 'n/a';
}

export function formatPercent(value: unknown): string {
  const num = Number(value);
  return Number.isFinite(num) ? `${Math.round(num * 100)}%` : 'n/a';
}

export function buildPlayerSummary(row: DecisionRow | null | undefined): string {
  if (!row) {
    return 'No player selected.';
  }
  const reasons: string[] = [];
  if (Number(row.starter_delta) > 0) {
    reasons.push(
      `adds ${formatNumber(row.starter_delta)} points over a typical ${row.position} starter`,
    );
  }
  if (Number(row.rank_gap_vs_vor) > 0) {
    reasons.push('the contextual model likes this profile more than the simple VOR baseline');
  } else if (Number(row.rank_gap_vs_vor) < 0) {
    reasons.push('simple VOR likes this player a bit more than the contextual score');
  }
  if (Number(row.availability_to_next_pick) < 0.35) {
    reasons.push('is unlikely to reach your next pick');
  } else if (Number(row.availability_to_next_pick) > 0.7) {
    reasons.push('has a realistic chance to survive to your next turn');
  }
  if (Number(row.fragility_score) > 0.6) {
    reasons.push('comes with elevated fragility risk');
  } else if (Number(row.upside_score) > 0.7) {
    reasons.push('brings strong upside if you want ceiling');
  }
  if (Number(row.posterior_games_mean) > 0 && Number(row.posterior_rate_mean) > 0) {
    reasons.push(
      `projects for about ${formatNumber(row.posterior_rate_mean)} points when active across ${formatNumber(row.posterior_games_mean)} games`,
    );
  }
  if (Number(row.rookie_draft_pick) > 0) {
    reasons.push(
      'has current/prior draft-year rookie context shaped by draft capital and current depth-chart opportunity',
    );
  }
  return `${row.player_name} is recommended because ${reasons.slice(0, 3).join(', ')}.`.replace(
    ' because .',
    '.',
  );
}

function derivedCurrentPickNumber(state: DraftState): number {
  const drafted = new Set(
    [...state.takenPlayers, ...state.yourPlayers].map(safeLower).filter(Boolean),
  );
  return drafted.size + 1;
}

function currentRoundNumber(currentPickNumber: number, leagueSize: number): number {
  return Math.max(
    1,
    Math.ceil(Math.max(1, Number(currentPickNumber) || 1) / Math.max(1, Number(leagueSize) || 1)),
  );
}

function specialistWindowOpen(state: DraftState): boolean {
  const round = currentRoundNumber(state.currentPickNumber, state.leagueSize);
  const totalRounds =
    Object.values(state.rosterSpots || {}).reduce((sum, value) => sum + (Number(value) || 0), 0)
    + (Number(state.benchSlots) || 0);
  return round >= Math.max(1, totalRounds - LATE_SPECIALIST_BUFFER_ROUNDS);
}

function availabilityProbability(
  adp: unknown,
  targetPick: number,
  adpStd: unknown,
  uncertaintyScore: unknown,
): number {
  const adpValue = Number(adp);
  if (!Number.isFinite(adpValue)) {
    return 0.5;
  }
  let spread = Number(adpStd);
  if (!Number.isFinite(spread) || spread <= 0) {
    spread = 2.5;
  }
  const uncertainty = Number(uncertaintyScore);
  if (Number.isFinite(uncertainty)) {
    spread += 2.0 * uncertainty;
  }
  const z = (adpValue - Number(targetPick)) / Math.max(1, spread);
  return Math.max(0, Math.min(1, 1 / (1 + Math.exp(-z))));
}

function getPresetEntry(payload: DashboardPayload, state: DraftState) {
  const scoringPresets = payload.scoring_presets ?? {};
  const requested = scoringPresets[state.scoringPreset] as
    | { available?: boolean; decision_table?: DecisionRow[]; key?: string; label?: string }
    | undefined;
  if (requested?.available) {
    return requested;
  }
  const fallback = Object.values(scoringPresets).find(
    (entry) => entry && typeof entry === 'object' && (entry as { available?: boolean }).available,
  ) as { decision_table?: DecisionRow[]; key?: string; label?: string } | undefined;
  return (
    fallback ?? {
      decision_table: payload.decision_table ?? [],
      available: true,
      key: state.scoringPreset,
      label: 'Half PPR (0.5)',
    }
  );
}

function activeRows(payload: DashboardPayload, state: DraftState): DecisionRow[] {
  const presetEntry = getPresetEntry(payload, state);
  if (Array.isArray(presetEntry.decision_table) && presetEntry.decision_table.length) {
    return presetEntry.decision_table.map((row) => ({ ...row }));
  }
  return Array.isArray(payload.decision_table)
    ? payload.decision_table.map((row) => ({ ...row }))
    : [];
}

function rosterCounts(rows: DecisionRow[], state: DraftState): Record<string, number> {
  const byPlayer = new Map(rows.map((row) => [safeLower(row.player_name), row]));
  return state.yourPlayers.reduce<Record<string, number>>((counts, playerName) => {
    const row = byPlayer.get(safeLower(playerName));
    if (row) {
      counts[row.position] = (counts[row.position] || 0) + 1;
    }
    return counts;
  }, {});
}

function rosterNeed(
  counts: Record<string, number>,
  state: DraftState,
): Record<string, number> {
  const need = POSITION_KEYS.reduce<Record<string, number>>((acc, position) => {
    acc[position] = Math.max(
      0,
      Number(state.rosterSpots[position] || 0) - Number(counts[position] || 0),
    );
    return acc;
  }, {});
  const flexEligibleExtras = ['RB', 'WR', 'TE'].reduce((sum, position) => {
    const baseSlots = Number(state.rosterSpots[position] || 0);
    const filled = Number(counts[position] || 0);
    return sum + Math.max(0, filled - baseSlots);
  }, 0);
  need.FLEX = Math.max(0, Number(state.rosterSpots.FLEX || 0) - flexEligibleExtras);
  return need;
}

function offensiveNeedByPosition(need: Record<string, number>, position: string): number {
  const baseNeed = Number(need[position] || 0);
  if (position === 'RB' || position === 'WR' || position === 'TE') {
    const flexWeights = { RB: 0.45, WR: 0.45, TE: 0.1 };
    return baseNeed + Number(need.FLEX || 0) * Number(flexWeights[position as keyof typeof flexWeights] || 0);
  }
  return baseNeed;
}

function startersByPosition(state: DraftState): Record<string, number> {
  return POSITION_KEYS.reduce<Record<string, number>>((acc, position) => {
    acc[position] = Number(state.rosterSpots[position] || 0) * Math.max(1, Number(state.leagueSize || 1));
    return acc;
  }, {});
}

function replacementSlots(state: DraftState, flexWeights: Record<string, number>): Record<string, number> {
  const starters = startersByPosition(state);
  const flexSlots = Number(state.rosterSpots.FLEX || 0) * Math.max(1, Number(state.leagueSize || 1));
  const replacement = { ...starters };
  replacement.RB = (replacement.RB || 0) + Math.round(flexSlots * (flexWeights.RB || 0));
  replacement.WR = (replacement.WR || 0) + Math.round(flexSlots * (flexWeights.WR || 0));
  replacement.TE = (replacement.TE || 0) + Math.round(flexSlots * (flexWeights.TE || 0));
  return replacement;
}

function positionBaseline(
  rows: DecisionRow[],
  position: string,
  slotCount: number,
  fallback: number,
): number {
  const values = rows
    .filter((row) => row.position === position)
    .map((row) => Number(row.proj_points_mean))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => b - a);
  if (!values.length) {
    return fallback;
  }
  const index = Math.max(0, Math.min(values.length - 1, Math.round(slotCount || 1) - 1));
  return values[index];
}

function rankPercentiles(values: unknown[]): number[] {
  const finite = values
    .map((value, index) => ({ value: Number(value), index }))
    .filter((entry) => Number.isFinite(entry.value));
  if (!finite.length) {
    return values.map(() => 0.5);
  }
  const sorted = [...finite].sort((a, b) => a.value - b.value);
  const result = values.map(() => 0.5);
  sorted.forEach((entry, idx) => {
    result[entry.index] = sorted.length === 1 ? 0.5 : idx / (sorted.length - 1);
  });
  return result;
}

function zScores(values: unknown[]): number[] {
  const numeric = values.map((value) => Number(value));
  const finite = numeric.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return numeric.map(() => 0);
  }
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
  const std = Math.sqrt(variance);
  if (!Number.isFinite(std) || std === 0) {
    return numeric.map(() => 0);
  }
  return numeric.map((value) => (Number.isFinite(value) ? (value - mean) / std : 0));
}

function assignTiers(rows: DecisionRow[]): void {
  const total = rows.length || 1;
  rows.forEach((row, index) => {
    const bucket = Math.min(5, Math.floor((index / total) * 5) + 1);
    row.draft_tier = `Tier ${bucket}`;
  });
}

function getStatus(
  row: DecisionRow,
  takenSet: Set<string>,
  yoursSet: Set<string>,
  queueSet: Set<string>,
): string {
  const key = safeLower(row.player_name);
  if (yoursSet.has(key)) {
    return 'mine';
  }
  if (takenSet.has(key)) {
    return 'taken';
  }
  if (queueSet.has(key)) {
    return 'queued';
  }
  return 'available';
}

function isInspectableStatus(status: string | undefined): boolean {
  return status === 'available' || status === 'queued';
}

export function filterBoardRows(
  rows: DecisionRow[],
  search: string,
): DecisionRow[] {
  const query = safeLower(search);
  return rows.filter((row) => {
    if (query) {
      const haystack = [row.player_name, row.position, row.team].map(safeLower).join(' ');
      if (!haystack.includes(query)) {
        return false;
      }
    }
    return true;
  });
}

export function buildBoardState(payload: DashboardPayload, state: DraftState): BoardState {
  const draftState = {
    ...state,
    currentPickNumber: derivedCurrentPickNumber(state),
  };
  const flexWeights = {
    RB: 0.45,
    WR: 0.45,
    TE: 0.1,
    ...((payload.league_settings?.flex_weights as Record<string, number> | undefined) ?? {}),
  };

  const rows = activeRows(payload, draftState).filter((row) => {
    const name = (row.player_name || '').toString().trim();
    return (
      name
      && name.toUpperCase() !== 'UNKNOWN'
      && (FANTASY_DRAFT_POSITIONS as readonly string[]).includes(row.position)
    );
  });

  const takenSet = new Set((draftState.takenPlayers || []).map(safeLower));
  const yoursSet = new Set((draftState.yourPlayers || []).map(safeLower));
  const queueSet = new Set((draftState.queuePlayers || []).map(safeLower));
  const nextPick = nextPickNumber(
    draftState.currentPickNumber,
    draftState.draftPosition,
    draftState.leagueSize,
  );
  const roundNumber = currentRoundNumber(draftState.currentPickNumber, draftState.leagueSize);
  const lateSpecialistsOk = specialistWindowOpen(draftState);
  const counts = rosterCounts(rows, draftState);
  const need = rosterNeed(counts, draftState);
  const effectiveNeed = OFFENSIVE_POSITIONS.reduce<Record<string, number>>((acc, position) => {
    acc[position] = offensiveNeedByPosition(need, position);
    return acc;
  }, {});
  const totalRounds =
    Object.values(draftState.rosterSpots || {}).reduce((sum, value) => sum + (Number(value) || 0), 0)
    + (Number(draftState.benchSlots) || 0);
  const remainingRounds = Math.max(1, totalRounds - roundNumber + 1);
  const openSpecialists = ['DST', 'K'].filter((position) => Number(need[position] || 0) > 0);
  const starterCounts = startersByPosition(draftState);
  const replacementCounts = replacementSlots(draftState, flexWeights);
  const overallMeanProjection =
    rows
      .map((row) => Number(row.proj_points_mean))
      .filter((value) => Number.isFinite(value))
      .reduce((sum, value, _, values) => sum + value / values.length, 0) || 0;

  const baselines: Record<
    string,
    { starter: number; replacement: number; scarcity: number }
  > = {};
  ['QB', 'RB', 'WR', 'TE', 'DST', 'K'].forEach((position) => {
    baselines[position] = {
      starter: positionBaseline(
        rows,
        position,
        starterCounts[position] || 1,
        overallMeanProjection,
      ),
      replacement: positionBaseline(
        rows,
        position,
        replacementCounts[position] || 1,
        overallMeanProjection,
      ),
      scarcity: 1 / Math.max(1, rows.filter((row) => row.position === position).length),
    };
  });

  const totalNeed =
    OFFENSIVE_POSITIONS.reduce((sum, position) => sum + (effectiveNeed[position] || 0), 0) || 1;
  const openOffensiveSlots = OFFENSIVE_POSITIONS.reduce(
    (sum, position) => sum + (effectiveNeed[position] || 0),
    0,
  );
  const projectionValues: number[] = [];
  const starterValues: number[] = [];
  const replacementValues: number[] = [];
  const posteriorConfidenceValues: number[] = [];
  const fragilityValues: number[] = [];
  const upsideValues: number[] = [];
  const starterNeedValues: number[] = [];
  const marketGapValues: number[] = [];
  const availableCounts: Record<string, number> = {};

  rows.forEach((row) => {
    const baseline = baselines[row.position] || {
      starter: overallMeanProjection,
      replacement: overallMeanProjection,
      scarcity: 0,
    };
    row.availability_at_pick = availabilityProbability(
      row.adp,
      nextPick,
      row.adp_std,
      row.uncertainty_score,
    );
    row.availability_to_next_pick = row.availability_at_pick as number;
    row.starter_baseline = baseline.starter;
    row.replacement_baseline = baseline.replacement;
    row.starter_delta = Number(row.proj_points_mean || 0) - baseline.starter;
    row.replacement_delta = Number(row.proj_points_mean || 0) - baseline.replacement;
    row.simple_vor_proxy = row.replacement_delta as number;
    row.position_scarcity = baseline.scarcity;
    row.starter_need = (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position)
      ? (effectiveNeed[row.position] || 0) / totalNeed
      : 0;
    row.status = getStatus(row, takenSet, yoursSet, queueSet);
    if (row.status === 'available' || row.status === 'queued') {
      availableCounts[row.position] = (availableCounts[row.position] || 0) + 1;
    }
    projectionValues.push(Number(row.proj_points_mean || 0));
    starterValues.push(Number(row.starter_delta || 0));
    replacementValues.push(Number(row.replacement_delta || 0));
    posteriorConfidenceValues.push(Number(row.posterior_prob_beats_replacement || 0));
    fragilityValues.push(Number(row.fragility_score || 0));
    starterNeedValues.push(Number(row.starter_need || 0));
    marketGapValues.push(Number(row.market_gap || 0));
    upsideValues.push(Number(row.proj_points_ceiling || 0) - Number(row.proj_points_mean || 0));
  });

  const upsidePercentiles = rankPercentiles(upsideValues);
  const availabilityPercentiles = rankPercentiles(
    rows.map((row) => Number(row.availability_to_next_pick || 0)),
  );
  const projectionPercentiles = rankPercentiles(projectionValues);
  rows.forEach((row, index) => {
    row.upside_score = Math.max(
      0,
      Math.min(
        1,
        upsidePercentiles[index]
          + 0.35 * availabilityPercentiles[index]
          + 0.15 * projectionPercentiles[index]
          + 0.45 * Number(row.posterior_prob_beats_replacement || 0),
      ),
    );
  });
  const riskMultiplier =
    ({ low: 0.8, medium: 1.0, high: 1.18 })[
      (draftState.riskTolerance || 'medium').toLowerCase() as 'low' | 'medium' | 'high'
    ] || 1.0;
  const componentZ = {
    proj_points_mean: zScores(projectionValues),
    starter_delta: zScores(starterValues),
    replacement_delta: zScores(replacementValues),
    posterior_prob_beats_replacement: zScores(posteriorConfidenceValues),
    starter_need: zScores(starterNeedValues),
    fragility_score: zScores(fragilityValues),
    market_gap: zScores(marketGapValues),
  };

  rows.forEach((row, index) => {
    row.component_terms = {
      proj_points_mean: 0.4 * componentZ.proj_points_mean[index],
      starter_delta: 0.24 * componentZ.starter_delta[index],
      replacement_delta: 0.18 * componentZ.replacement_delta[index],
      posterior_prob_beats_replacement: 0.1 * componentZ.posterior_prob_beats_replacement[index],
      starter_need: 0.03 * componentZ.starter_need[index],
      fragility_score: -(0.06 * riskMultiplier) * componentZ.fragility_score[index],
      market_gap: 0.05 * componentZ.market_gap[index],
    };
    row.board_value_score = Object.values(row.component_terms as Record<string, number>).reduce(
      (sum, value) => sum + value,
      0,
    );
    row.draft_score = row.board_value_score as number;
  });

  rows.sort(
    (a, b) =>
      Number(b.draft_score) - Number(a.draft_score)
      || Number(b.proj_points_mean) - Number(a.proj_points_mean),
  );
  assignTiers(rows);
  rows.forEach((row, index) => {
    row.draft_rank = index + 1;
  });
  const simpleVorSorted = [...rows].sort(
    (a, b) =>
      Number(b.simple_vor_proxy) - Number(a.simple_vor_proxy)
      || Number(b.proj_points_mean) - Number(a.proj_points_mean),
  );
  simpleVorSorted.forEach((row, index) => {
    row.simple_vor_rank = index + 1;
  });

  const availableRows = rows.filter((row) => row.status === 'available' || row.status === 'queued');
  const bestOffensiveValue = availableRows
    .filter((row) => (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position))
    .reduce((best, row) => Math.max(best, Number(row.draft_score) || 0), Number.NEGATIVE_INFINITY);
  const lineupGainValues: number[] = [];
  availableRows.forEach((row) => {
    const posNeed = effectiveNeed[row.position] || 0;
    const demand = (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position)
      ? Math.max(1, Number(draftState.leagueSize || 1) * Math.max(1, posNeed))
      : Math.max(1, Number(draftState.leagueSize || 1));
    row.position_run_risk = Math.max(
      0,
      Math.min(1, 1 - (availableCounts[row.position] || 0) / demand),
    );
    row.starter_slot_urgency = (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position)
      ? posNeed / totalNeed
      : 0;
    row.specialist_need_bonus =
      (row.position === 'DST' || row.position === 'K')
      && lateSpecialistsOk
      && Number(need[row.position] || 0) > 0
        ? 2.5
        : 0;
    if (
      (row.position === 'DST' || row.position === 'K')
      && openSpecialists.includes(row.position)
      && remainingRounds <= openSpecialists.length + 1
    ) {
      row.specialist_need_bonus = Number(row.specialist_need_bonus) + 3.0;
    }
    row.specialist_urgency =
      (row.position === 'DST' || row.position === 'K')
      && lateSpecialistsOk
      && Number(need[row.position] || 0) > 0
        ? 1.25 * (1 - Number(row.availability_to_next_pick || 0))
          + 0.75 * Math.max(0, Math.min(1, 1 / Math.max(1, availableCounts[row.position] || 1)))
        : 0;
    row.roster_fit_score = Number(row.starter_slot_urgency || 0);
    row.lineup_gain_now = Math.max(
      0,
      (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position)
        ? Number(row.starter_delta || 0)
        : Number(row.replacement_delta || 0) * 0.35,
      (OFFENSIVE_POSITIONS as readonly string[]).includes(row.position)
        ? Number(row.replacement_delta || 0) * 0.7
        : Number(row.replacement_delta || 0) * 0.2,
    );
    lineupGainValues.push(Number(row.lineup_gain_now || 0));
    row.policy_eligible = true;
    row.policy_eligibility_reason = 'eligible';
    if ((row.position === 'DST' || row.position === 'K') && !lateSpecialistsOk) {
      row.policy_eligible = false;
      row.policy_eligibility_reason = 'late_round_only';
    } else if (
      (row.position === 'QB' || row.position === 'TE')
      && openOffensiveSlots > 0
      && Number(counts[row.position] || 0) >= Number(draftState.rosterSpots[row.position] || 0)
      && Number(row.draft_score || 0) < bestOffensiveValue + SECONDARY_QB_TE_VALUE_EDGE
    ) {
      row.policy_eligible = false;
      row.policy_eligibility_reason = 'starter_priority';
    }
  });
  const lineupGainPercentiles = rankPercentiles(lineupGainValues);
  availableRows.forEach((row, index) => {
    row.expected_regret =
      (0.55 * lineupGainPercentiles[index]
        + 0.25 * Number(row.starter_slot_urgency || 0)
        + 0.2 * Number(row.position_run_risk || 0))
      * (1 - Number(row.availability_to_next_pick || 0));
    if ((row.position === 'DST' || row.position === 'K') && lateSpecialistsOk) {
      row.wait_signal = 'late_round_stash_ok';
    } else if (Number(row.availability_to_next_pick || 0) < CONSERVATIVE_WAIT_SURVIVAL_THRESHOLD) {
      row.wait_signal = 'low_survival';
    } else if (Number(row.expected_regret || 0) > CONSERVATIVE_WAIT_LINEUP_LOSS_THRESHOLD) {
      row.wait_signal = 'too_much_lineup_loss';
    } else {
      row.wait_signal = 'safe_to_wait';
    }
    const riskBias =
      ({ low: -0.03, medium: 0.0, high: 0.03 })[
        (draftState.riskTolerance || 'medium').toLowerCase() as 'low' | 'medium' | 'high'
      ] || 0;
    row.current_pick_utility =
      Number(row.draft_score || 0)
      + Number(row.specialist_need_bonus || 0)
      + Number(row.specialist_urgency || 0)
      + 0.32 * Number(row.starter_slot_urgency || 0)
      + 0.22 * lineupGainPercentiles[index]
      + 0.08 * Number(row.posterior_prob_beats_replacement || 0)
      + 0.06 * Number(row.position_run_risk || 0)
      + riskBias * Number(row.upside_score || 0);
    if (!row.policy_eligible) {
      row.current_pick_utility = Number(row.current_pick_utility) - 2.5;
    }
    row.wait_utility =
      Number(row.draft_score || 0) * Number(row.availability_to_next_pick || 0)
      + 0.06 * Number(row.upside_score || 0)
      - 0.85 * Number(row.expected_regret || 0);
    row.rank_gap_vs_vor = Number(row.simple_vor_rank || 0) - Number(row.draft_rank || 0);
    row.rationale_live = buildPlayerSummary(row);
  });

  const recommendedNow = [...availableRows]
    .filter((row) => row.policy_eligible)
    .sort(
      (a, b) =>
        Number(b.current_pick_utility) - Number(a.current_pick_utility)
        || Number(b.draft_score) - Number(a.draft_score),
    );
  const recommendedWait = [...availableRows]
    .filter((row) => row.wait_signal === 'safe_to_wait' || row.wait_signal === 'late_round_stash_ok')
    .sort(
      (a, b) =>
        Number(b.wait_utility) - Number(a.wait_utility)
        || Number(b.availability_to_next_pick) - Number(a.availability_to_next_pick),
    );
  const pickNow = recommendedNow.slice(0, 1).map((row, index) => ({
    ...row,
    recommendation_lane: 'pick_now',
    lane_rank: index + 1,
  }));
  const fallbacks = (
    recommendedNow.length
      ? recommendedNow
      : [...availableRows].sort(
          (a, b) =>
            Number(b.current_pick_utility) - Number(a.current_pick_utility)
            || Number(b.draft_score) - Number(a.draft_score),
        )
  )
    .slice(1, 6)
    .map((row, index) => ({
      ...row,
      recommendation_lane: 'fallback',
      lane_rank: index + 1,
    }));
  const canWait = (
    recommendedWait.length
      ? recommendedWait
      : [...availableRows].sort(
          (a, b) =>
            Number(b.availability_to_next_pick) - Number(a.availability_to_next_pick)
            || Number(b.draft_score) - Number(a.draft_score),
        )
  )
    .filter(
      (row) =>
        !pickNow.some((candidate) => safeLower(candidate.player_name) === safeLower(row.player_name)),
    )
    .slice(0, 5)
    .map((row, index) => ({
      ...row,
      recommendation_lane: 'can_wait',
      lane_rank: index + 1,
    }));

  const selectedKey = safeLower(draftState.selectedPlayer);
  const selectedCandidate = rows.find((row) => safeLower(row.player_name) === selectedKey) || null;
  const selectedRow =
    selectedCandidate && isInspectableStatus(selectedCandidate.status)
      ? selectedCandidate
      : pickNow[0] || availableRows[0] || rows[0] || null;

  return {
    rows,
    availableRows,
    nextPick,
    rosterCounts: counts,
    rosterNeed: need,
    pickNow,
    fallbacks,
    canWait,
    selectedRow,
  };
}
