import type { DashboardPayload } from '../../payload/load';
import { formatNumber } from '../../state/buildBoardState';
import type { DraftStore } from '../../state/draftState';
import { useBoardState } from '../../state/useBoardState';
import { SectionGate } from '../SectionGate';
import {
  buildComparativeModel,
  formatSignedNumber,
  getWarRoomVisualsConfig,
  type WarRoomVisualsConfig,
} from './warRoomModels';

export function ComparativeExplainer(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const boardState = useBoardState(payload, store);
  const row = boardState.selectedRow;
  const visualsConfig = getWarRoomVisualsConfig(
    payload.war_room_visuals as WarRoomVisualsConfig | undefined,
  );
  const config = visualsConfig.comparative_explainer;
  const comparative = buildComparativeModel(boardState, row, config, visualsConfig);

  return (
    <details>
      <summary>Contextual vs baseline</summary>
      <div className="details-body" id="comparative-explainer">
        <SectionGate section={config} title="Contextual vs baseline">
          {!comparative.available ? (
            <div className="notice">
              {comparative.reason
                || 'No contextual-versus-baseline explainer is available for this player.'}
            </div>
          ) : (
            <>
              <div className="summary-box">{comparative.headline}</div>
              {comparative.status !== 'available' && comparative.reason ? (
                <div className="notice">{comparative.reason}</div>
              ) : null}
              <div className="metric-grid">
                <div className="metric">
                  <span className="label">{comparative.contextualLabel} rank</span>
                  <span className="value">{comparative.row.contextual_rank || 'n/a'}</span>
                </div>
                <div className="metric">
                  <span className="label">{comparative.baselineLabel} rank</span>
                  <span className="value">{comparative.row.baseline_rank || 'n/a'}</span>
                </div>
                <div className="metric">
                  <span className="label">{comparative.contextualLabel}</span>
                  <span className="value">{formatNumber(comparative.row.contextual_score)}</span>
                </div>
                <div className="metric">
                  <span className="label">{comparative.baselineLabel}</span>
                  <span className="value">{formatNumber(comparative.row.baseline_score)}</span>
                </div>
              </div>
              <div className="board-table-wrap" style={{ maxHeight: '220px' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Pos</th>
                      <th>{comparative.contextualLabel} rank</th>
                      <th>{comparative.baselineLabel} rank</th>
                      <th>Gap</th>
                    </tr>
                  </thead>
                  <tbody>
                    {row ? (
                      <tr>
                        <td>{row.player_name}</td>
                        <td>{row.position}</td>
                        <td>{comparative.row.contextual_rank || 'n/a'}</td>
                        <td>{comparative.row.baseline_rank || 'n/a'}</td>
                        <td>{formatSignedNumber(comparative.row.rank_gap)}</td>
                      </tr>
                    ) : null}
                    {comparative.peers.length ? (
                      comparative.peers.map((peer) => (
                        <tr key={peer.player_name}>
                          <td>{peer.player_name}</td>
                          <td>{peer.position}</td>
                          <td>{peer.contextual_rank || 'n/a'}</td>
                          <td>{peer.baseline_rank || 'n/a'}</td>
                          <td>{formatSignedNumber(peer.rank_gap)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={5} className="empty">
                          No nearby disagreement peers are available.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </SectionGate>
      </div>
    </details>
  );
}
