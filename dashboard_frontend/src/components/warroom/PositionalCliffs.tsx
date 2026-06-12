import type { DashboardPayload } from '../../payload/load';
import { formatNumber } from '../../state/buildBoardState';
import type { DraftStore } from '../../state/draftState';
import { safeLower } from '../../state/draftState';
import { useBoardState } from '../../state/useBoardState';
import { useDraftStore } from '../../state/useDraftStore';
import { SectionGate } from '../SectionGate';
import {
  buildCliffGroups,
  deriveRelevantCliffPositions,
  getWarRoomVisualsConfig,
  type WarRoomVisualsConfig,
} from './warRoomModels';

function cliffSummary(groups: ReturnType<typeof buildCliffGroups>): string {
  const summaryBits = groups.slice(0, 2).map((group) => {
    const nextPlayer = group.players[group.strongestCliffIndex + 1];
    if (group.strongestCliffIndex >= 0 && nextPlayer) {
      return `${group.position}: after ${group.players[group.strongestCliffIndex].row.player_name}`;
    }
    return `${group.position}: flat`;
  });
  return summaryBits.join(' · ');
}

export function PositionalCliffs(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const boardState = useBoardState(payload, store);
  const visualsConfig = getWarRoomVisualsConfig(
    payload.war_room_visuals as WarRoomVisualsConfig | undefined,
  );
  const config = visualsConfig.positional_cliffs;
  const groups = buildCliffGroups(boardState);
  const defaultPositions = deriveRelevantCliffPositions(boardState, groups, config);
  const visibleGroups = state.showAllCliffs
    ? groups
    : groups.filter((group) => defaultPositions.includes(group.position));
  const primaryKey = safeLower(boardState.pickNow[0]?.player_name);
  const selectedKey = safeLower(state.selectedPlayer);

  return (
    <section className="panel strong">
      <details>
        <summary>
          Positional Cliffs{' '}
          <span className="tiny" id="positional-cliffs-summary" style={{ marginLeft: '8px' }}>
            {config.available === false
              ? 'unavailable'
              : !groups.length
                ? 'no active cliffs'
                : cliffSummary(visibleGroups)}
          </span>
        </summary>
        <div className="details-body">
          <SectionGate section={config} title="Positional Cliffs">
            <div className="visual-stack" id="positional-cliffs">
              {!groups.length ? (
                <div className="notice">
                  {config.reason || 'No current positional cliff data is available.'}
                </div>
              ) : (
                <>
                  <div className="summary-box">
                    {config.question || 'Which positions are about to fall off if I wait?'}
                  </div>
                  {config.status !== 'available' && config.reason ? (
                    <div className="notice">{config.reason}</div>
                  ) : null}
                  <div className="split">
                    <div className="tiny">
                      Default view follows the active recommendation lanes and selected-player
                      context.
                    </div>
                    <button type="button" id="toggle-cliffs" onClick={() => store.toggleShowAllCliffs()}>
                      {state.showAllCliffs ? 'Focus relevant positions' : 'Show all positions'}
                    </button>
                  </div>
                  <div className="cliff-stack">
                    {visibleGroups.map((group) => (
                      <div key={group.position} className="cliff-row">
                        <div className="lane-item-header">
                          <div className="item-title">{group.position}</div>
                          <span className="pill">
                            Strongest drop {formatNumber(group.cliffStrength)}
                          </span>
                        </div>
                        <div className="cliff-strip">
                          {group.players.map((player, index) => (
                            <span key={player.row.player_name} style={{ display: 'contents' }}>
                              <button
                                type="button"
                                className={`cliff-node ${
                                  safeLower(player.row.player_name) === selectedKey ? 'is-selected' : ''
                                } ${
                                  safeLower(player.row.player_name) === primaryKey ? 'is-primary' : ''
                                } ${index === group.strongestCliffIndex ? 'is-cliff-edge' : ''}`}
                                data-cliff-player={player.row.player_name}
                                onClick={() => store.selectPlayer(player.row.player_name)}
                              >
                                {player.row.player_name}
                              </button>
                              {index === group.strongestCliffIndex && group.players[index + 1] ? (
                                <div className="cliff-break">
                                  <span className="cliff-break-value">
                                    drop {formatNumber(group.strongestCliffDistance)}
                                  </span>
                                  <span className="cliff-break-line" />
                                  <span>cliff</span>
                                </div>
                              ) : null}
                            </span>
                          ))}
                        </div>
                        <div className="cliff-strip-note">
                          {group.strongestCliffIndex >= 0 && group.players[group.strongestCliffIndex + 1]
                            ? `Main break after ${group.players[group.strongestCliffIndex].row.player_name} before ${group.players[group.strongestCliffIndex + 1].row.player_name}.`
                            : 'No sharp break detected right now; position remains relatively flat.'}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </SectionGate>
        </div>
      </details>
    </section>
  );
}
