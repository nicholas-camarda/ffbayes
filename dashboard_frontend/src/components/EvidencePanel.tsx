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
        <div className="evidence-tables">
          <div className="panel-table-block">
            <h3 className="panel-table-title">Strategy comparison</h3>
            <div className="panel-table-wrap">
              <table className="panel-table panel-table-strategy">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th className="num">Mean lineup pts</th>
                    <th className="num">Seasons</th>
                  </tr>
                </thead>
                <tbody>
                  {strategyRows.length ? (
                    strategyRows.map((row) => (
                      <tr key={row.strategy}>
                        <td data-label="Strategy">{row.strategy}</td>
                        <td className="num" data-label="Mean lineup pts">
                          {formatNumber(row.mean_lineup_points)}
                        </td>
                        <td className="num" data-label="Seasons">
                          {row.season_count ?? 'n/a'}
                        </td>
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
          </div>
          <div className="panel-table-block">
            <h3 className="panel-table-title">Season holdouts</h3>
            <div className="panel-table-wrap">
              <table className="panel-table panel-table-seasons">
                <thead>
                  <tr>
                    <th className="num">Year</th>
                    <th className="num">Board value</th>
                    <th className="num">VOR proxy</th>
                    <th className="num">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {seasonRows.length ? (
                    seasonRows.map((row) => (
                      <tr key={row.holdout_year}>
                        <td className="num" data-label="Year">
                          {row.holdout_year}
                        </td>
                        <td className="num" data-label="Board value">
                          {formatNumber(row.draft_score_lineup_points)}
                        </td>
                        <td className="num" data-label="VOR proxy">
                          {formatNumber(row.historical_vor_proxy_lineup_points)}
                        </td>
                        <td className="num" data-label="Delta">
                          {formatNumber(row.delta_lineup_points)}
                        </td>
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
          </div>
        </div>
        {limitations.length ? (
          <div className="notice">{limitations.join(' ')}</div>
        ) : null}
      </SectionGate>
    </section>
  );
}
