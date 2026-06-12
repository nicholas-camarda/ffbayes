import type { DashboardPayload } from '../payload/load';
import type { DraftStore } from '../state/draftState';
import { useDraftStore } from '../state/useDraftStore';
import { PickStatus } from './PickStatus';
import { PlayerBoard } from './PlayerBoard';
import { PlayerInspector } from './PlayerInspector';
import {
  RecommendationFallbacks,
  RecommendationPrimary,
  RecommendationWait,
} from './RecommendationPanel';
import { EvidencePanel } from './EvidencePanel';
import { FreshnessPanel } from './FreshnessPanel';
import { ProvenanceBanner } from './ProvenanceBanner';
import { SettingsPanel } from './SettingsPanel';
import { FinalizePanel } from './FinalizePanel';
import { PositionalCliffs } from './warroom/PositionalCliffs';
import { TimingFrontier } from './warroom/TimingFrontier';

export function AppShell(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);

  return (
    <div className="shell">
      <section className="topbar">
        <div className="topbar-head">
          <div className="title-wrap">
            <div className="title-row">
              <h1>FFBayes Draft War Room</h1>
              <span className="pill">Live draft mode</span>
              <span className="pill" id="storage-pill">
                Saved locally
              </span>
            </div>
            <p className="subtitle">
              Operate the draft from this board: update league shape, scoring preset, queue,
              taken players, and your roster without leaving the dashboard.
            </p>
          </div>
          <div className="toolbar-stack">
            <div className="toolbar-row">
              <button type="button" id="undo-button" disabled={!state.history.length} onClick={() => store.undo()}>
                Undo
              </button>
              <button type="button" id="redo-button" disabled={!state.redoHistory.length} onClick={() => store.redo()}>
                Redo
              </button>
            </div>
            <FinalizePanel payload={payload} store={store} />
          </div>
        </div>
        <PickStatus payload={payload} store={store} />
        <div className="tiny">Generated {payload.generated_at}</div>
      </section>

      <section className="layout">
        <div className="column">
          <RecommendationPrimary payload={payload} store={store} />
          <TimingFrontier payload={payload} store={store} />
        </div>

        <div className="column">
          <PositionalCliffs payload={payload} store={store} />
          <PlayerBoard payload={payload} store={store} />
          <RecommendationFallbacks payload={payload} store={store} />
          <RecommendationWait payload={payload} store={store} />
        </div>

        <div className="column">
          <SettingsPanel payload={payload} store={store} />
          <section className="panel placeholder-panel">
            <h2>Queue &amp; Roster</h2>
            <p className="subtle">Queue and roster coming soon.</p>
            <div className="roster-chip-row" id="my-roster" />
            <div className="metric-grid" id="roster-need-grid" />
          </section>
          <PlayerInspector payload={payload} store={store} />
          <EvidencePanel payload={payload} />
          <FreshnessPanel payload={payload} />
          {payload.publish_provenance ? <ProvenanceBanner payload={payload} /> : null}
        </div>
      </section>
    </div>
  );
}
