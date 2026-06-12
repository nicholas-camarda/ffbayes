import type { DashboardPayload } from '../payload/load';
import {
  buildPlayerSummary,
  formatNumber,
  formatPercent,
  type DecisionRow,
} from '../state/buildBoardState';
import type { DraftStore } from '../state/draftState';
import { useBoardState } from '../state/useBoardState';
import { ComparativeExplainer } from './warroom/ComparativeExplainer';

const COMPONENT_LABELS: Record<string, string> = {
  starter_delta: 'Starter edge',
  replacement_delta: 'Simple VOR',
  proj_points_mean: 'Projection',
  availability_to_next_pick: 'Draft timing',
  upside_score: 'Upside',
  starter_need: 'Roster need',
  position_scarcity: 'Scarcity',
  fragility_score: 'Fragility',
  market_gap: 'Market gap',
};

function ContributionBars(props: { row: DecisionRow }) {
  const { row } = props;
  const contributions = Object.entries(row.component_terms || {}).map(([key, value]) => ({
    key,
    value: Number(value || 0),
    magnitude: Math.abs(Number(value || 0)),
  }));
  const totalContribution = contributions.reduce((sum, item) => sum + item.magnitude, 0) || 1;

  if (!contributions.length) {
    return null;
  }

  return (
    <div className="bar-stack">
      {contributions
        .sort((a, b) => b.magnitude - a.magnitude)
        .map((item) => (
          <div key={item.key} className="bar-row">
            <div className="bar-head">
              <span>{COMPONENT_LABELS[item.key] || item.key}</span>
              <span>{formatNumber(item.value)}</span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${
                  item.key === 'fragility_score' ? 'bad' : item.value >= 0 ? 'good' : 'warn'
                }`}
                style={{
                  width: `${Math.max(6, Math.round((item.magnitude / totalContribution) * 100))}%`,
                }}
              />
            </div>
          </div>
        ))}
    </div>
  );
}

export function PlayerInspector(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const boardState = useBoardState(payload, store);
  const row = boardState.selectedRow;

  return (
    <section className="panel strong" id="player-inspector">
      <h2>Selected Player</h2>
      {!row ? (
        <div className="empty">Select a player from the board.</div>
      ) : (
        <>
          <div className="inspector-title">
            <div className="pill-row">
              <span className="pill">{row.position}</span>
              <span className="pill">{String(row.status ?? 'available')}</span>
              <span className="pill">Draft rank {String(row.draft_rank ?? 'n/a')}</span>
              <span className="pill">VOR rank {String(row.simple_vor_rank ?? 'n/a')}</span>
            </div>
            <div className="hero-name" style={{ fontSize: '24px' }}>
              {row.player_name}
            </div>
            <div className="summary-box">{buildPlayerSummary(row)}</div>
          </div>
          <div className="metric-grid">
            <div className="metric">
              <span className="label">Board value score</span>
              <span className="value">{formatNumber(row.draft_score)}</span>
            </div>
            <div className="metric">
              <span className="label">Simple VOR proxy</span>
              <span className="value">{formatNumber(row.simple_vor_proxy)}</span>
            </div>
            <div className="metric">
              <span className="label">Availability to next pick</span>
              <span className="value">{formatPercent(row.availability_to_next_pick)}</span>
            </div>
            <div className="metric">
              <span className="label">Expected regret</span>
              <span className="value">{formatNumber(row.expected_regret)}</span>
            </div>
            <div className="metric">
              <span className="label">Upside score</span>
              <span className="value">{formatPercent(row.upside_score)}</span>
            </div>
            <div className="metric">
              <span className="label">Fragility score</span>
              <span className="value">{formatPercent(row.fragility_score)}</span>
            </div>
          </div>
          <details open>
            <summary>Why this player / why not wait?</summary>
            <div className="details-body">
              <ContributionBars row={row} />
            </div>
          </details>
          <ComparativeExplainer payload={payload} store={store} />
          <details>
            <summary>Projection breakdown</summary>
            <div className="details-body">
              <div className="metric-grid">
                <div className="metric">
                  <span className="label">Season total mean</span>
                  <span className="value">
                    {formatNumber(row.posterior_mean ?? row.proj_points_mean)}
                  </span>
                </div>
                <div className="metric">
                  <span className="label">Rate when active</span>
                  <span className="value">{formatNumber(row.posterior_rate_mean)}</span>
                </div>
                <div className="metric">
                  <span className="label">Expected games</span>
                  <span className="value">{formatNumber(row.posterior_games_mean)}</span>
                </div>
                <div className="metric">
                  <span className="label">Availability rate</span>
                  <span className="value">{formatPercent(row.availability_rate_projection)}</span>
                </div>
                <div className="metric">
                  <span className="label">Current team</span>
                  <span className="value">{String(row.current_team || row.team || 'n/a')}</span>
                </div>
                <div className="metric">
                  <span className="label">Team change</span>
                  <span className="value">{Number(row.team_change) > 0 ? 'yes' : 'no'}</span>
                </div>
              </div>
              {Number(row.rookie_draft_pick) > 0
              || Number(row.depth_chart_rank) > 0
              || Number(row.rookie_combine_score) ? (
                <div className="summary-box">
                  Current/prior draft-year rookie context: draft pick{' '}
                  {String(row.rookie_draft_pick || 'n/a')}, depth rank{' '}
                  {String(row.depth_chart_rank || 'n/a')}, combine signal{' '}
                  {formatNumber(row.rookie_combine_score)}.
                </div>
              ) : null}
            </div>
          </details>
        </>
      )}
    </section>
  );
}
