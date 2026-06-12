import type { DashboardPayload } from '../payload/load';
import { FreshnessNotice, formatTimestamp } from './freshnessShared';

type PublishProvenance = {
  published_at?: string;
  source_html?: string;
  dashboard_generated_at?: string;
  source_payload?: string;
  surface_sync?: {
    status?: string;
    detail?: string;
  };
};

type AnalysisProvenance = {
  overall_freshness?: {
    status?: string;
    override_used?: boolean;
    warnings?: string[];
  };
};

type DecisionEvidence = {
  freshness?: {
    status?: string;
    override_used?: boolean;
    warnings?: string[];
  };
};

export function ProvenanceBanner(props: { payload: DashboardPayload }) {
  const { payload } = props;
  const publishProvenance = payload.publish_provenance as PublishProvenance;
  const analysisProvenance = (payload.analysis_provenance ?? {}) as AnalysisProvenance;
  const decisionEvidence = (payload.decision_evidence ?? {}) as DecisionEvidence;
  const analysisFreshness =
    analysisProvenance.overall_freshness || decisionEvidence.freshness || {};

  return (
    <section className="panel" id="provenance-panel">
      <h2>Publish provenance</h2>
      <div className="summary-box">
        {publishProvenance.published_at
          ? `Pages staged ${formatTimestamp(publishProvenance.published_at)} from ${publishProvenance.source_html || 'index.html'}.`
          : 'Publish provenance will appear after `ffbayes pre-draft --stage-pages`, `ffbayes stage-dashboard`, or `ffbayes publish` stages this dashboard.'}
      </div>
      <div className="tiny">Dashboard generated: {formatTimestamp(payload.generated_at)}</div>
      <div className="tiny">
        Analysis freshness: {analysisFreshness.status || 'unknown'}
        {analysisFreshness.override_used ? ' (explicit override used)' : ''}
      </div>
      <div className="tiny">
        Staged surface sync:{' '}
        {publishProvenance.surface_sync?.status ||
          (publishProvenance.published_at ? 'status detail unavailable' : 'local-only')}
      </div>
      {publishProvenance.dashboard_generated_at || publishProvenance.source_payload ? (
        <div className="tiny">
          Staged payload: {publishProvenance.source_payload || 'dashboard_payload.json'} • generated{' '}
          {formatTimestamp(publishProvenance.dashboard_generated_at)}
        </div>
      ) : null}
      <FreshnessNotice freshness={analysisFreshness} />
      {publishProvenance.surface_sync?.detail ? (
        <div className="tiny">{publishProvenance.surface_sync.detail}</div>
      ) : null}
    </section>
  );
}
