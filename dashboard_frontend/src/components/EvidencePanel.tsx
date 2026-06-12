import type { DashboardPayload } from '../payload/load';
import { formatNumber } from '../state/buildBoardState';
import { SectionGate } from './SectionGate';

type StrategyRow = {
  strategy: string;
  mean_lineup_points?: number;
  season_count?: number;
};

type SeasonRow = {
  holdout_year: number;
  draft_score_lineup_points?: number;
  historical_vor_proxy_lineup_points?: number;
  delta_lineup_points?: number;
};

type DecisionEvidence = {
  available?: boolean;
  headline?: string;
  reason_unavailable?: string;
  strategy_summary?: StrategyRow[];
  season_rows?: SeasonRow[];
  limitations?: string[];
};

export function EvidencePanel(props: { payload: DashboardPayload }) {
  const evidence = props.payload.decision_evidence as DecisionEvidence;
  const strategyRows = evidence.strategy_summary ?? [];
  const seasonRows = evidence.season_rows ?? [];
  const limitations = evidence.limitations ?? [];

  return (
    <section className="panel" id="decision-evidence">
      <h2>Decision evidence</h2>
      <p className="subtle">
        Internal holdout comparison between contextual board value and the simple VOR baseline.
      </p>
      <SectionGate
        section={{
          available: evidence.available,
          reason_unavailable:
            evidence.reason_unavailable ||
            evidence.headline ||
            'No decision evidence is available in this payload.',
        }}
        title="Decision evidence"
      >
        <div className="summary-box">
          {evidence.headline || 'Decision evidence is available for this board.'}
        </div>
        <div className="board-table-wrap" style={{ maxHeight: '220px' }}>
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Mean lineup points</th>
                <th>Seasons</th>
              </tr>
            </thead>
            <tbody>
              {strategyRows.length ? (
                strategyRows.map((row) => (
                  <tr key={row.strategy}>
                    <td>{row.strategy}</td>
                    <td>{formatNumber(row.mean_lineup_points)}</td>
                    <td>{row.season_count ?? 'n/a'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3} className="empty">
                    No strategy rows available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="board-table-wrap" style={{ maxHeight: '220px' }}>
          <table>
            <thead>
              <tr>
                <th>Year</th>
                <th>Board value score</th>
                <th>Simple VOR proxy</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {seasonRows.length ? (
                seasonRows.map((row) => (
                  <tr key={row.holdout_year}>
                    <td>{row.holdout_year}</td>
                    <td>{formatNumber(row.draft_score_lineup_points)}</td>
                    <td>{formatNumber(row.historical_vor_proxy_lineup_points)}</td>
                    <td>{formatNumber(row.delta_lineup_points)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="empty">
                    No season-level rows available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {limitations.length ? (
          <div className="notice">{limitations.join(' ')}</div>
        ) : null}
      </SectionGate>
    </section>
  );
}
