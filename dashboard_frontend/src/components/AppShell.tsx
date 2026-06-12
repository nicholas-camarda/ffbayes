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
import { SettingsPanel } from './SettingsPanel';

function isLocalFinalizeSupported(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return window.location.protocol === 'file:';
}

export function AppShell(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const showFinalize = isLocalFinalizeSupported();

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
              <button type="button" id="finalize-button" hidden={!showFinalize}>
                Finalize Draft
              </button>
            </div>
            <div className="tiny toolbar-note" id="finalize-note">
              {showFinalize
                ? 'Finalize downloads a locked HTML snapshot, JSON export, and post-draft summary from this local dashboard.'
                : 'Finalize downloads are only supported from the local generated dashboard opened as a file.'}
            </div>
          </div>
        </div>
        <PickStatus payload={payload} store={store} />
        <div className="tiny">Generated {payload.generated_at}</div>
      </section>

      <section className="layout">
        <div className="column">
          <RecommendationPrimary payload={payload} store={store} />
          <section className="panel placeholder-panel" id="timing-frontier">
            <h2>Wait vs Pick Frontier</h2>
            <p className="subtle">Timing frontier coming soon.</p>
          </section>
        </div>

        <div className="column">
          <section className="panel strong placeholder-panel" id="positional-cliffs">
            <h2>Positional Cliffs</h2>
            <p className="subtle">Positional cliffs coming soon.</p>
          </section>
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
          <section className="panel placeholder-panel">
            <h2>Decision evidence</h2>
            <p className="subtle">Evidence panels coming soon.</p>
          </section>
          {payload.publish_provenance ? (
            <section className="panel placeholder-panel" id="provenance-panel">
              <h2>Publish provenance</h2>
              <p className="subtle">Provenance banner coming soon.</p>
            </section>
          ) : null}
        </div>
      </section>
    </div>
  );
}
