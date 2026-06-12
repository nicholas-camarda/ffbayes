import type { DashboardPayload } from '../payload/load';
import { FreshnessNotice, type FreshnessRow } from './freshnessShared';

type AnalysisProvenance = {
  overall_freshness?: {
    status?: string;
    override_used?: boolean;
    warnings?: string[];
  };
  sources?: Record<string, FreshnessRow>;
};

export function FreshnessPanel(props: { payload: DashboardPayload }) {
  const { payload } = props;
  const analysisProvenance = (payload.analysis_provenance ?? {}) as AnalysisProvenance;
  const analysisFreshness = analysisProvenance.overall_freshness ?? {};
  const sourceRows = Object.values(analysisProvenance.sources ?? {});
  const fallbackRows = Array.isArray(payload.source_freshness)
    ? (payload.source_freshness as FreshnessRow[])
    : [];
  const freshnessRows = sourceRows.length ? sourceRows : fallbackRows;

  return (
    <section className="panel" id="freshness-panel">
      <h2>Freshness and provenance</h2>
      <p className="subtle">
        Source freshness and analysis-window status for this board.
      </p>
      <div className="metric-grid">
        <div className="metric">
          <span className="label">Analysis freshness</span>
          <span className="value">{analysisFreshness.status || 'unknown'}</span>
        </div>
        <div className="metric">
          <span className="label">Override used</span>
          <span className="value">{analysisFreshness.override_used ? 'yes' : 'no'}</span>
        </div>
      </div>
      <FreshnessNotice freshness={analysisFreshness} />
      <div className="board-table-wrap" style={{ maxHeight: '220px' }}>
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Status</th>
              <th>Override</th>
              <th>Expected / found</th>
            </tr>
          </thead>
          <tbody id="freshness-table">
            {freshnessRows.length ? (
              freshnessRows.map((row, index) => (
                <tr key={`${row.source_name || row.label || 'source'}-${index}`}>
                  <td>{row.source_name || row.label || 'unknown'}</td>
                  <td>
                    {row.status ||
                      (row.freshness_days != null ? `${row.freshness_days}d` : 'unknown')}
                  </td>
                  <td>{row.override_used ? 'yes' : 'no'}</td>
                  <td>
                    {row.latest_expected_year ?? '-'} / {row.latest_found_year ?? '-'}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} className="empty">
                  No freshness rows available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
