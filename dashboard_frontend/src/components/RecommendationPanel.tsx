import { useMemo } from 'react';
import type { DashboardPayload } from '../payload/load';
import {
  buildPlayerSummary,
  formatNumber,
  formatPercent,
  type DecisionRow,
} from '../state/buildBoardState';
import type { DraftStore } from '../state/draftState';
import { useBoardState } from '../state/useBoardState';
import { SectionGate, type GatedSection } from './SectionGate';

type LiveStateSection = GatedSection & {
  pick_now?: DecisionRow;
  fallbacks?: DecisionRow[];
  can_wait?: DecisionRow[];
};

function recommendationsSection(payload: DashboardPayload): GatedSection | undefined {
  const live = payload.live_state as LiveStateSection | undefined;
  if (live) {
    if (live.available === false) {
      return live;
    }
    return { available: true, status: 'available' };
  }
  if ((payload.decision_table?.length ?? 0) > 0) {
    return { available: true, status: 'available' };
  }
  return undefined;
}

function laneSection(
  live: LiveStateSection | undefined,
  laneKey: 'pick_now' | 'fallbacks' | 'can_wait',
  title: string,
): GatedSection | undefined {
  if (!live) {
    return { available: true, status: 'available' };
  }
  if (live.available === false) {
    return live;
  }
  if (laneKey === 'pick_now') {
    return live.pick_now ? { available: true } : { available: false, reason: `${title} is not available.` };
  }
  const laneRows = live[laneKey];
  if (Array.isArray(laneRows) && laneRows.length > 0) {
    return { available: true };
  }
  return { available: true, status: 'available' };
}

function PrimaryCard(props: { primary: DecisionRow | undefined }) {
  const { primary } = props;
  if (!primary) {
    return <div className="empty">No available recommendation.</div>;
  }
  return (
    <>
      <div className="pill-row">
        <span className="pill">{primary.position}</span>
        <span className="pill">Board value {formatNumber(primary.draft_score)}</span>
        <span className="pill">Simple VOR rank {primary.simple_vor_rank}</span>
      </div>
      <div className="hero-name">{primary.player_name}</div>
      <div className="summary-box">{buildPlayerSummary(primary)}</div>
      <div className="metric-grid">
        <div className="metric">
          <span className="label">Availability to next pick</span>
          <span className="value">{formatPercent(primary.availability_to_next_pick)}</span>
        </div>
        <div className="metric">
          <span className="label">Expected regret</span>
          <span className="value">{formatNumber(primary.expected_regret)}</span>
        </div>
        <div className="metric">
          <span className="label">Position run risk</span>
          <span className="value">{formatNumber(primary.position_run_risk)}</span>
        </div>
        <div className="metric">
          <span className="label">Roster fit</span>
          <span className="value">{formatNumber(primary.roster_fit_score)}</span>
        </div>
      </div>
    </>
  );
}

function LaneList(props: { rows: DecisionRow[]; emptyMessage: string }) {
  const { rows, emptyMessage } = props;
  if (!rows.length) {
    return <div className="empty">{emptyMessage}</div>;
  }
  return (
    <>
      {rows.map((row) => (
        <div key={row.player_name} className="lane-item">
          <div className="lane-item-header">
            <div className="item-title">
              {row.player_name} <span className="tiny">• {row.position}</span>
            </div>
            <span className={`status-badge ${row.status}`}>{row.status}</span>
          </div>
          <div className="item-meta">
            Board value {formatNumber(row.draft_score)} • Survival{' '}
            {formatPercent(row.availability_to_next_pick)} • Regret{' '}
            {formatNumber(row.expected_regret)}
          </div>
          <div className="tiny">{buildPlayerSummary(row)}</div>
        </div>
      ))}
    </>
  );
}

export function RecommendationPrimary(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const boardState = useBoardState(payload, store);
  const live = payload.live_state as LiveStateSection | undefined;
  const section = useMemo(
    () => laneSection(live, 'pick_now', 'Pick now'),
    [live],
  );
  const gate = recommendationsSection(payload);

  return (
    <section className="panel strong">
      <h2>Pick Now</h2>
      <p className="subtle">
        Best current move after adjusting for your league, roster, and next-pick survival.
      </p>
      <SectionGate section={gate} title="Recommendations">
        <SectionGate section={section} title="Pick now">
          <div className="hero-pick" id="primary-card">
            <PrimaryCard primary={boardState.pickNow[0]} />
          </div>
        </SectionGate>
      </SectionGate>
    </section>
  );
}

export function RecommendationFallbacks(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const boardState = useBoardState(payload, store);
  const live = payload.live_state as LiveStateSection | undefined;
  const section = useMemo(
    () => laneSection(live, 'fallbacks', 'Fallback ladder'),
    [live],
  );

  return (
    <section className="panel">
      <div className="split">
        <div>
          <h2>Fallback Ladder</h2>
          <p className="subtle">If the top target goes, pivot here.</p>
        </div>
        <span className="pill">Immediate pivots</span>
      </div>
      <SectionGate section={section} title="Fallback ladder">
        <div className="lane-list" id="fallback-list">
          <LaneList
            rows={boardState.fallbacks}
            emptyMessage="No fallback options right now."
          />
        </div>
      </SectionGate>
    </section>
  );
}

export function RecommendationWait(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const boardState = useBoardState(payload, store);
  const live = payload.live_state as LiveStateSection | undefined;
  const section = useMemo(
    () => laneSection(live, 'can_wait', 'Can wait'),
    [live],
  );

  return (
    <section className="panel">
      <div className="split">
        <div>
          <h2>Can Wait</h2>
          <p className="subtle">Strong values with the best shot to survive to your next turn.</p>
        </div>
        <span className="pill">Patience lane</span>
      </div>
      <SectionGate section={section} title="Can wait">
        <div className="lane-list" id="wait-list">
          <LaneList rows={boardState.canWait} emptyMessage="No wait candidates right now." />
        </div>
      </SectionGate>
    </section>
  );
}
