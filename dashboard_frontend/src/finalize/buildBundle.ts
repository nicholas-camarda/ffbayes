import type { DashboardPayload } from '../payload/load';
import {
  buildBoardState,
  formatNumber,
  formatPercent,
  type BoardState,
  type DecisionRow,
} from '../state/buildBoardState';
import { safeLower, type DraftState, type PickLogEntry } from '../state/draftState';

export const FINALIZED_SCHEMA_VERSION = 'finalized_draft_v1';

export interface FinalizedDraftPayload {
  schema_version: string;
  season_year: number;
  exported_at: string;
  source_payload_generated_at: string;
  title: string;
  league_settings: {
    league_size: number;
    draft_position: number;
    scoring_preset: string;
    risk_tolerance: string;
    bench_slots: number;
    roster_spots: Record<string, number>;
  };
  final_state: {
    current_pick_number: number;
    next_pick_number: number;
    taken_players: string[];
    your_players: string[];
    queue_players: string[];
    roster_capacity: number;
    drafted_player_count: number;
    roster_complete: boolean;
  };
  drafted_players: Array<Record<string, unknown>>;
  starters: Array<Record<string, unknown>>;
  bench: Array<Record<string, unknown>>;
  summary_metrics: Record<string, unknown>;
  value_recap: Array<Record<string, unknown>>;
  pick_receipts: PickLogEntry[];
  selected_player: Record<string, unknown> | null;
}

function escapeHtml(value: unknown): string {
  return (value ?? '')
    .toString()
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function rosterCapacity(state: DraftState): number {
  return (
    Object.values(state.rosterSpots || {}).reduce((sum, value) => sum + (Number(value) || 0), 0)
    + (Number(state.benchSlots) || 0)
  );
}

function draftYearLabel(payload: DashboardPayload): string {
  return String((payload.generated_at || '').slice(0, 4) || new Date().getFullYear());
}

function summarizeValueIndicator(row: DecisionRow): string {
  const draftRank = Number(row.draft_rank || 0);
  const marketRank = Number(row.market_rank || row.adp || 0);
  if (!Number.isFinite(draftRank) || !Number.isFinite(marketRank) || !draftRank || !marketRank) {
    return 'Fair';
  }
  const edge = marketRank - draftRank;
  if (edge >= 12) {
    return 'Value';
  }
  if (edge <= -12) {
    return 'Reach';
  }
  return 'Fair';
}

function summarizeRiskStyle(avgFragility: number, avgUpside: number): string {
  if (avgFragility >= 0.55 && avgUpside >= 0.65) {
    return 'Ceiling-heavy and fragile';
  }
  if (avgFragility >= 0.55) {
    return 'Fragile';
  }
  if (avgUpside >= 0.65) {
    return 'Ceiling-heavy';
  }
  return 'Balanced';
}

function getDraftedRows(boardState: BoardState, state: DraftState): DecisionRow[] {
  const draftedSet = new Set((state.yourPlayers || []).map(safeLower));
  return boardState.rows
    .filter((row) => draftedSet.has(safeLower(row.player_name)))
    .map((row) => ({ ...row }));
}

export function isRosterComplete(boardState: BoardState, state: DraftState): boolean {
  const draftedRows = getDraftedRows(boardState, state);
  const openNeed = Object.values(boardState.rosterNeed || {}).reduce(
    (sum, value) => sum + Number(value || 0),
    0,
  );
  return draftedRows.length >= rosterCapacity(state) && openNeed === 0;
}

function sortRowsForLineup(rows: DecisionRow[]): DecisionRow[] {
  return [...rows].sort(
    (a, b) =>
      Number(b.proj_points_mean || 0) - Number(a.proj_points_mean || 0)
      || Number(b.draft_score || 0) - Number(a.draft_score || 0),
  );
}

type LineupRow = DecisionRow & { lineup_slot: string };

function buildStarterBenchSplit(draftedRows: DecisionRow[], state: DraftState) {
  const used = new Set<string>();
  const starters: Array<DecisionRow & { lineup_slot: string }> = [];
  const addSlot = (position: string, slotCount: number, labelPrefix: string) => {
    const pool = sortRowsForLineup(
      draftedRows.filter(
        (row) => row.position === position && !used.has(safeLower(row.player_name)),
      ),
    );
    for (let index = 0; index < Math.max(0, slotCount); index += 1) {
      const row = pool[index];
      if (!row) {
        break;
      }
      used.add(safeLower(row.player_name));
      starters.push({ ...row, lineup_slot: `${labelPrefix}${index + 1}` });
    }
  };

  addSlot('QB', Number(state.rosterSpots.QB || 0), 'QB');
  addSlot('RB', Number(state.rosterSpots.RB || 0), 'RB');
  addSlot('WR', Number(state.rosterSpots.WR || 0), 'WR');
  addSlot('TE', Number(state.rosterSpots.TE || 0), 'TE');
  addSlot('DST', Number(state.rosterSpots.DST || 0), 'DST');
  addSlot('K', Number(state.rosterSpots.K || 0), 'K');

  const flexPool = sortRowsForLineup(
    draftedRows.filter(
      (row) =>
        ['RB', 'WR', 'TE'].includes(row.position)
        && !used.has(safeLower(row.player_name)),
    ),
  );
  for (let index = 0; index < Math.max(0, Number(state.rosterSpots.FLEX || 0)); index += 1) {
    const row = flexPool[index];
    if (!row) {
      break;
    }
    used.add(safeLower(row.player_name));
    starters.push({ ...row, lineup_slot: `FLEX${index + 1}` });
  }

  const bench: LineupRow[] = sortRowsForLineup(
    draftedRows.filter((row) => !used.has(safeLower(row.player_name))),
  ).map((row, index) => ({ ...row, lineup_slot: `BENCH${index + 1}` }));

  return { starters, bench };
}

function sumMetric(rows: DecisionRow[], key: string): number {
  return rows.reduce((sum, row) => sum + Number(row[key] || 0), 0);
}

function tableRowsHtml(rows: Array<Record<string, unknown>>, columns: Array<{ key: string }>) {
  return rows.length
    ? rows
        .map(
          (row) => `
            <tr>${columns.map((column) => `<td>${escapeHtml(row[column.key] ?? '')}</td>`).join('')}</tr>
          `,
        )
        .join('')
    : `<tr><td colspan="${columns.length}" class="empty">No rows available.</td></tr>`;
}

export function buildFinalizedDraftPayload(
  boardState: BoardState,
  state: DraftState,
  payload: DashboardPayload,
): FinalizedDraftPayload {
  const draftedRows = getDraftedRows(boardState, state);
  const split = buildStarterBenchSplit(draftedRows, state);
  const starterFragility = split.starters.length
    ? split.starters.reduce((sum, row) => sum + Number(row.fragility_score || 0), 0)
      / split.starters.length
    : 0;
  const fullFragility = draftedRows.length
    ? draftedRows.reduce((sum, row) => sum + Number(row.fragility_score || 0), 0)
      / draftedRows.length
    : 0;
  const starterUpside = split.starters.length
    ? split.starters.reduce((sum, row) => sum + Number(row.upside_score || 0), 0)
      / split.starters.length
    : 0;
  const fullUpside = draftedRows.length
    ? draftedRows.reduce((sum, row) => sum + Number(row.upside_score || 0), 0) / draftedRows.length
    : 0;
  const capacity = rosterCapacity(state);
  const rosterComplete = isRosterComplete(boardState, state);
  const projectedStarterMean = sumMetric(split.starters, 'proj_points_mean');
  const projectedStarterFloor = sumMetric(split.starters, 'proj_points_floor');
  const projectedStarterCeiling = sumMetric(split.starters, 'proj_points_ceiling');
  const benchProjection = sumMetric(split.bench, 'proj_points_mean');
  const draftedPlayerRows = draftedRows.map((row) => ({
    player_name: row.player_name,
    position: row.position,
    team: row.team || '',
    adp: row.adp,
    market_rank: row.market_rank,
    draft_rank: row.draft_rank,
    draft_score: row.draft_score,
    simple_vor_proxy: row.simple_vor_proxy,
    fragility_score: row.fragility_score,
    upside_score: row.upside_score,
    proj_points_mean: row.proj_points_mean,
    proj_points_floor: row.proj_points_floor,
    proj_points_ceiling: row.proj_points_ceiling,
    lineup_slot:
      [...split.starters, ...split.bench].find(
        (item) => safeLower(item.player_name) === safeLower(row.player_name),
      )?.lineup_slot || '',
    value_indicator: summarizeValueIndicator(row),
  }));
  const followedCount = (state.pickLog || []).filter((entry) => entry.followed_model).length;
  const latestPickedPlayer = state.pickLog.length
    ? draftedPlayerRows.find(
        (row) =>
          safeLower(row.player_name) === safeLower(state.pickLog[state.pickLog.length - 1].player_name),
      )
    : null;

  return {
    schema_version: FINALIZED_SCHEMA_VERSION,
    season_year: Number(draftYearLabel(payload)),
    exported_at: new Date().toISOString(),
    source_payload_generated_at: payload.generated_at || '',
    title: 'FFBayes Finalized Draft Snapshot',
    league_settings: {
      league_size: state.leagueSize,
      draft_position: state.draftPosition,
      scoring_preset: state.scoringPreset,
      risk_tolerance: state.riskTolerance,
      bench_slots: state.benchSlots,
      roster_spots: { ...(state.rosterSpots || {}) },
    },
    final_state: {
      current_pick_number: state.currentPickNumber,
      next_pick_number: boardState.nextPick,
      taken_players: state.takenPlayers.slice(),
      your_players: state.yourPlayers.slice(),
      queue_players: state.queuePlayers.slice(),
      roster_capacity: capacity,
      drafted_player_count: draftedRows.length,
      roster_complete: rosterComplete,
    },
    drafted_players: draftedPlayerRows,
    starters: split.starters.map((row) => ({
      player_name: row.player_name,
      position: row.position,
      lineup_slot: row.lineup_slot,
      proj_points_mean: Number(row['proj_points_mean'] || 0),
    })),
    bench: split.bench.map((row) => ({
      player_name: row.player_name,
      position: row.position,
      lineup_slot: row.lineup_slot,
      proj_points_mean: Number(row['proj_points_mean'] || 0),
    })),
    summary_metrics: {
      starter_lineup_mean: projectedStarterMean,
      starter_lineup_floor: projectedStarterFloor,
      starter_lineup_ceiling: projectedStarterCeiling,
      bench_depth_mean: benchProjection,
      starter_fragility_avg: starterFragility,
      full_roster_fragility_avg: fullFragility,
      starter_upside_avg: starterUpside,
      full_roster_upside_avg: fullUpside,
      risk_style: summarizeRiskStyle(starterFragility, starterUpside),
      model_follow_count: followedCount,
      model_pivot_count: Math.max(0, (state.pickLog || []).length - followedCount),
    },
    value_recap: draftedPlayerRows.map((row) => ({
      player_name: row.player_name,
      position: row.position,
      adp: row.adp,
      market_rank: row.market_rank,
      draft_rank: row.draft_rank,
      draft_score: row.draft_score,
      value_indicator: row.value_indicator,
    })),
    pick_receipts: (state.pickLog || []).slice(),
    selected_player: latestPickedPlayer || draftedPlayerRows[0] || null,
  };
}

export function buildFinalizedSnapshotHtml(payload: FinalizedDraftPayload): string {
  const draftedRows = payload.drafted_players || [];
  const starters = payload.starters || [];
  const bench = payload.bench || [];
  const selectedPlayer = payload.selected_player || draftedRows[0] || null;
  const metrics = payload.summary_metrics || {};
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FFBayes Finalized Draft Snapshot</title>
  <style>
    :root { color-scheme: dark; --bg:#08111e; --panel:#0f172a; --border:rgba(148,163,184,0.18); --text:#f8fafc; --muted:#94a3b8; --accent:#38bdf8; --good:#34d399; --warn:#f59e0b; }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(180deg,#030712 0%,#0b1324 100%); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,sans-serif; padding:20px; }
    .shell { max-width:1480px; margin:0 auto; display:grid; gap:16px; }
    .panel { background:rgba(15,23,42,0.92); border:1px solid var(--border); border-radius:20px; padding:18px; }
    .layout { display:grid; grid-template-columns:1.1fr 0.9fr; gap:16px; }
    .pill { display:inline-flex; align-items:center; padding:6px 10px; border-radius:999px; border:1px solid rgba(56,189,248,0.24); background:rgba(56,189,248,0.14); color:#d7f5ff; font-size:12px; margin-right:8px; margin-bottom:8px; }
    .metric-grid { display:grid; gap:10px; grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .metric { background:rgba(255,255,255,0.04); border:1px solid rgba(148,163,184,0.12); border-radius:14px; padding:10px 12px; }
    .label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
    .value { display:block; font-size:16px; font-weight:600; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { text-align:left; padding:10px; border-bottom:1px solid rgba(148,163,184,0.12); vertical-align:top; }
    th { color:#9bdcff; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }
    .summary-box { padding:12px; border-radius:14px; background:rgba(255,255,255,0.04); border:1px solid rgba(148,163,184,0.12); line-height:1.5; }
    .roster-grid { display:grid; gap:12px; grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .empty { color:var(--muted); }
    @media (max-width: 980px) { .layout, .metric-grid, .roster-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <div class="pill">Read-only snapshot</div>
      <div class="pill">Schema ${escapeHtml(payload.schema_version || '')}</div>
      <div class="pill">League ${escapeHtml(payload.league_settings?.league_size || '')}-team</div>
      <h1 style="margin:8px 0 6px 0;">FFBayes Finalized Draft Snapshot</h1>
      <p style="color:#94a3b8; margin:0;">Saved from the local live draft dashboard. This snapshot is locked and contains no draft controls.</p>
    </section>
    <section class="panel">
      <div class="metric-grid">
        <div class="metric"><span class="label">Starter lineup mean</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_mean || 0))}</span></div>
        <div class="metric"><span class="label">Starter lineup floor</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_floor || 0))}</span></div>
        <div class="metric"><span class="label">Starter lineup ceiling</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_ceiling || 0))}</span></div>
        <div class="metric"><span class="label">Bench depth mean</span><span class="value">${escapeHtml(formatNumber(metrics.bench_depth_mean || 0))}</span></div>
        <div class="metric"><span class="label">Risk style</span><span class="value">${escapeHtml(String(metrics.risk_style || 'Balanced'))}</span></div>
        <div class="metric"><span class="label">Model pivots</span><span class="value">${escapeHtml(String(metrics.model_pivot_count || 0))}</span></div>
      </div>
    </section>
    <section class="layout">
      <section class="panel">
        <h2 style="margin-top:0;">Final Roster</h2>
        <div class="roster-grid">
          <div>
            <h3>Starters</h3>
            <table>
              <thead><tr><th>Slot</th><th>Player</th><th>Pos</th><th>Proj</th></tr></thead>
              <tbody>${tableRowsHtml(starters, [{ key: 'lineup_slot' }, { key: 'player_name' }, { key: 'position' }, { key: 'proj_points_mean' }])}</tbody>
            </table>
          </div>
          <div>
            <h3>Bench</h3>
            <table>
              <thead><tr><th>Slot</th><th>Player</th><th>Pos</th><th>Proj</th></tr></thead>
              <tbody>${tableRowsHtml(bench, [{ key: 'lineup_slot' }, { key: 'player_name' }, { key: 'position' }, { key: 'proj_points_mean' }])}</tbody>
            </table>
          </div>
        </div>
      </section>
      <section class="panel">
        <h2 style="margin-top:0;">Player Inspector</h2>
        ${
          selectedPlayer
            ? `
          <div class="summary-box">
            <strong>${escapeHtml(selectedPlayer.player_name || '')}</strong><br />
            ${escapeHtml(selectedPlayer.position || '')}${selectedPlayer.team ? ` • ${escapeHtml(selectedPlayer.team)}` : ''}<br />
            Draft rank ${escapeHtml(String(selectedPlayer.draft_rank || 'n/a'))} • ADP ${escapeHtml(formatNumber(selectedPlayer.adp || 0))}<br />
            Board value score ${escapeHtml(formatNumber(selectedPlayer.draft_score || 0))} • Simple VOR proxy ${escapeHtml(formatNumber(selectedPlayer.simple_vor_proxy || 0))}<br />
            Fragility ${escapeHtml(formatPercent(selectedPlayer.fragility_score || 0))} • Upside ${escapeHtml(formatPercent(selectedPlayer.upside_score || 0))}
          </div>
        `
            : '<div class="empty">No drafted player available.</div>'
        }
        <div style="margin-top:12px;">
          <h3>Pick Receipts</h3>
          <table>
            <thead><tr><th>Pick</th><th>Player</th><th>Top recommendation</th><th>Decision</th></tr></thead>
            <tbody>${tableRowsHtml((payload.pick_receipts || []) as unknown as Array<Record<string, unknown>>, [{ key: 'pick_number' }, { key: 'player_name' }, { key: 'top_recommendation' }, { key: 'decision_label' }])}</tbody>
          </table>
        </div>
      </section>
    </section>
  </div>
</body>
</html>`;
}

export function buildFinalizedSummaryHtml(payload: FinalizedDraftPayload): string {
  const metrics = payload.summary_metrics || {};
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FFBayes Post-Draft Summary</title>
  <style>
    :root { color-scheme: dark; --bg:#08111e; --panel:#0f172a; --border:rgba(148,163,184,0.18); --text:#f8fafc; --muted:#94a3b8; }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(180deg,#030712 0%,#0b1324 100%); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,sans-serif; padding:20px; }
    .shell { max-width:1280px; margin:0 auto; display:grid; gap:16px; }
    .panel { background:rgba(15,23,42,0.92); border:1px solid var(--border); border-radius:20px; padding:18px; }
    .metric-grid { display:grid; gap:10px; grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .metric { background:rgba(255,255,255,0.04); border:1px solid rgba(148,163,184,0.12); border-radius:14px; padding:10px 12px; }
    .label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
    .value { display:block; font-size:16px; font-weight:600; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { text-align:left; padding:10px; border-bottom:1px solid rgba(148,163,184,0.12); vertical-align:top; }
    th { color:#9bdcff; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }
    .summary-box { padding:12px; border-radius:14px; background:rgba(255,255,255,0.04); border:1px solid rgba(148,163,184,0.12); line-height:1.5; }
    @media (max-width: 980px) { .metric-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <h1 style="margin:0 0 6px 0;">FFBayes Post-Draft Summary</h1>
      <p style="color:#94a3b8; margin:0;">A compact recap of the roster you drafted and how it looked through the draft-board model.</p>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Final Roster Recap</h2>
      <div class="summary-box">
        Drafted ${escapeHtml(String(payload.final_state?.drafted_player_count || 0))} of ${escapeHtml(String(payload.final_state?.roster_capacity || 0))} total roster spots.
        Roster complete: ${escapeHtml(payload.final_state?.roster_complete ? 'Yes' : 'No')}.
      </div>
      <table style="margin-top:12px;">
        <thead><tr><th>Player</th><th>Pos</th><th>Slot</th><th>Proj</th></tr></thead>
        <tbody>${tableRowsHtml(payload.drafted_players || [], [{ key: 'player_name' }, { key: 'position' }, { key: 'lineup_slot' }, { key: 'proj_points_mean' }])}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Team Projection Snapshot</h2>
      <div class="metric-grid">
        <div class="metric"><span class="label">Starter lineup mean</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_mean || 0))}</span></div>
        <div class="metric"><span class="label">Starter lineup floor</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_floor || 0))}</span></div>
        <div class="metric"><span class="label">Starter lineup ceiling</span><span class="value">${escapeHtml(formatNumber(metrics.starter_lineup_ceiling || 0))}</span></div>
        <div class="metric"><span class="label">Bench depth mean</span><span class="value">${escapeHtml(formatNumber(metrics.bench_depth_mean || 0))}</span></div>
      </div>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Risk & Upside Profile</h2>
      <div class="summary-box">
        This roster profiles as <strong>${escapeHtml(String(metrics.risk_style || 'Balanced'))}</strong>.
        Starter fragility averages ${escapeHtml(formatPercent(metrics.starter_fragility_avg || 0))} with starter upside averaging ${escapeHtml(formatPercent(metrics.starter_upside_avg || 0))}.
      </div>
      <div class="metric-grid" style="margin-top:12px;">
        <div class="metric"><span class="label">Starter fragility avg</span><span class="value">${escapeHtml(formatPercent(metrics.starter_fragility_avg || 0))}</span></div>
        <div class="metric"><span class="label">Full-roster fragility avg</span><span class="value">${escapeHtml(formatPercent(metrics.full_roster_fragility_avg || 0))}</span></div>
        <div class="metric"><span class="label">Starter upside avg</span><span class="value">${escapeHtml(formatPercent(metrics.starter_upside_avg || 0))}</span></div>
        <div class="metric"><span class="label">Full-roster upside avg</span><span class="value">${escapeHtml(formatPercent(metrics.full_roster_upside_avg || 0))}</span></div>
      </div>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Draft Value Recap</h2>
      <table>
        <thead><tr><th>Player</th><th>Pos</th><th>ADP</th><th>Draft rank</th><th>Indicator</th></tr></thead>
        <tbody>${tableRowsHtml(payload.value_recap || [], [{ key: 'player_name' }, { key: 'position' }, { key: 'adp' }, { key: 'draft_rank' }, { key: 'value_indicator' }])}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Pick-by-Pick Receipts</h2>
      <table>
        <thead><tr><th>Pick</th><th>Player</th><th>Top recommendation</th><th>Decision</th><th>Board value score</th></tr></thead>
        <tbody>${tableRowsHtml((payload.pick_receipts || []) as unknown as Array<Record<string, unknown>>, [{ key: 'pick_number' }, { key: 'player_name' }, { key: 'top_recommendation' }, { key: 'decision_label' }, { key: 'draft_score' }])}</tbody>
      </table>
    </section>
  </div>
</body>
</html>`;
}

export function buildBundle(state: DraftState, payload: DashboardPayload) {
  const boardState = buildBoardState(payload, state);
  const json = buildFinalizedDraftPayload(boardState, state, payload);
  return {
    json,
    snapshotHtml: buildFinalizedSnapshotHtml(json),
    summaryHtml: buildFinalizedSummaryHtml(json),
  };
}

export function timestampLabel(date = new Date()): string {
  return date.toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '_');
}

export function buildFinalizeFilenames(payload: DashboardPayload, date = new Date()) {
  const year = draftYearLabel(payload);
  const stamp = timestampLabel(date);
  return {
    year,
    stamp,
    json: `ffbayes_finalized_draft_${year}_${stamp}.json`,
    snapshotHtml: `ffbayes_finalized_draft_${year}_${stamp}.html`,
    summaryHtml: `ffbayes_finalized_summary_${year}_${stamp}.html`,
  };
}

export function downloadTextFile(filename: string, text: string, mimeType: string): void {
  const blob = new Blob([text], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => window.URL.revokeObjectURL(url), 0);
}

export function isLocalFinalizeSupported(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return window.location.protocol === 'file:';
}
