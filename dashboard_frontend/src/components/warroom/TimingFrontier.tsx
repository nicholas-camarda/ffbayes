import type { DashboardPayload } from '../../payload/load';
import type { DraftStore } from '../../state/draftState';
import { safeLower } from '../../state/draftState';
import { formatNumber, formatPercent } from '../../state/buildBoardState';
import { useBoardState } from '../../state/useBoardState';
import { useDraftStore } from '../../state/useDraftStore';
import { SectionGate } from '../SectionGate';
import {
  buildTimingFrontierModel,
  getWarRoomVisualsConfig,
  type WarRoomVisualsConfig,
} from './warRoomModels';

function derivedCurrentPickNumber(takenPlayers: string[], yourPlayers: string[]): number {
  return new Set(
    [...takenPlayers, ...yourPlayers].map(safeLower).filter(Boolean),
  ).size + 1;
}

export function TimingFrontier(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const boardState = useBoardState(payload, store);
  const visualsConfig = getWarRoomVisualsConfig(
    payload.war_room_visuals as WarRoomVisualsConfig | undefined,
  );
  const config = visualsConfig.timing_frontier;
  const model = buildTimingFrontierModel(boardState, config, visualsConfig);
  const currentPick = derivedCurrentPickNumber(state.takenPlayers, state.yourPlayers);
  const selectedKey = safeLower(state.selectedPlayer);
  const primaryKey = safeLower(boardState.pickNow[0]?.player_name);

  return (
    <section className="panel">
      <h2>Wait vs Pick Frontier</h2>
      <p className="subtle">
        A compact timing view of who you need to take now versus who can plausibly survive to your
        next turn.
      </p>
      <SectionGate section={config} title="Wait vs Pick Frontier">
        <div
          className="visual-stack"
          id="timing-frontier"
          data-current-pick={currentPick}
          data-next-pick={boardState.nextPick}
        >
          {!model.available ? (
            <div className="notice">
              {model.reason || 'Timing frontier is unavailable for this dashboard state.'}
            </div>
          ) : (
            <>
              <div className="summary-box">
                {model.question || 'Can I safely wait on this value, or do I need to pick now?'}
                <div className="tiny" style={{ marginTop: '8px' }}>
                  Pick {currentPick} · Next {boardState.nextPick}. Rows compare pass regret and
                  next-pick survival directly. Red-tagged rows are take-now pressure; green-tagged
                  rows are safer to wait on.
                </div>
              </div>
              {model.status !== 'available' && model.reason ? (
                <div className="notice">{model.reason}</div>
              ) : null}
              <div className="frontier-board">
                <div className="frontier-scale">
                  <span>Candidate</span>
                  <span>Higher regret if you pass</span>
                  <span>More likely to survive</span>
                </div>
                {model.candidates.map((row) => (
                  <button
                    key={row.player_name}
                    type="button"
                    className={`frontier-row ${row.lane} ${safeLower(row.player_name) === selectedKey ? 'is-selected' : ''}`}
                    data-frontier-player={row.player_name}
                    onClick={() => store.selectPlayer(row.player_name)}
                  >
                    <span className="frontier-list-head">
                      <span className={`frontier-swatch ${row.lane}`} />
                      <span className="frontier-list-copy">
                        <span className="frontier-list-name">
                          <span className="frontier-player-name">{row.player_name}</span>
                          <span className="frontier-player-position">• {row.position}</span>
                        </span>
                        <span className="frontier-list-meta">{row.rationale}</span>
                        <span className="frontier-tag-row">
                          {safeLower(row.player_name) === primaryKey ? (
                            <span className="frontier-tag primary">Recommended now</span>
                          ) : null}
                          {safeLower(row.player_name) === selectedKey ? (
                            <span className="frontier-tag selected">Selected below</span>
                          ) : null}
                        </span>
                      </span>
                    </span>
                    <div className="frontier-meters">
                      <span className="frontier-meter">
                        <span className="frontier-meter-head">
                          <span>Pass regret</span>
                          <strong>{formatNumber(row.wait_regret)}</strong>
                        </span>
                        <span className="frontier-meter-track">
                          <span
                            className="frontier-meter-fill regret"
                            style={{
                              width: `${Math.max(8, Math.round(row.regret_percent * 100))}%`,
                            }}
                          />
                        </span>
                      </span>
                      <span className="frontier-meter">
                        <span className="frontier-meter-head">
                          <span>Survival</span>
                          <strong>{formatPercent(row.timing_survival)}</strong>
                        </span>
                        <span className="frontier-meter-track">
                          <span
                            className="frontier-meter-fill survival"
                            style={{
                              width: `${Math.max(8, Math.round(row.survival_percent * 100))}%`,
                            }}
                          />
                        </span>
                      </span>
                    </div>
                  </button>
                ))}
              </div>
              <div className="frontier-legend">
                <span className="legend-pill">
                  <span
                    className="legend-dot"
                    style={{ background: 'rgba(239, 68, 68, 0.95)' }}
                  />
                  Pick now pressure
                </span>
                <span className="legend-pill">
                  <span
                    className="legend-dot"
                    style={{ background: 'rgba(245, 158, 11, 0.95)' }}
                  />
                  Fallback pivot
                </span>
                <span className="legend-pill">
                  <span
                    className="legend-dot"
                    style={{ background: 'rgba(16, 185, 129, 0.95)' }}
                  />
                  Can wait
                </span>
              </div>
            </>
          )}
        </div>
      </SectionGate>
    </section>
  );
}
