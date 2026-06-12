import type { DashboardPayload } from '../payload/load';
import type { DraftStore } from '../state/draftState';
import { safeLower } from '../state/draftState';
import { useBoardState } from '../state/useBoardState';
import { useDraftStore } from '../state/useDraftStore';

const POSITION_KEYS = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'K'] as const;

export function QueueRosterPanel(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const boardState = useBoardState(payload, store);
  const queueKeys = new Set(state.queuePlayers.map(safeLower));
  const yourKeys = new Set(state.yourPlayers.map(safeLower));
  const queueRows = boardState.rows.filter((row) => queueKeys.has(safeLower(row.player_name)));
  const yourRows = boardState.rows.filter((row) => yourKeys.has(safeLower(row.player_name)));

  return (
    <section className="panel">
      <h2>Queue &amp; Roster</h2>
      <p className="subtle">Track queued targets and the roster you are building live.</p>
      <div id="queue-list">
        {queueRows.length ? (
          queueRows.map((row) => (
            <div key={row.player_name} className="mini-item">
              <div className="mini-item-header">
                <div className="item-title">
                  {row.player_name} <span className="tiny">• {row.position}</span>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty">Queue players with the “Queue” action in the board.</div>
        )}
      </div>
      <div className="roster-chip-row" id="my-roster">
        {yourRows.length ? (
          yourRows.map((row) => (
            <span key={row.player_name} className="pill good">
              {row.player_name} • {row.position}
            </span>
          ))
        ) : (
          <span className="empty">Your roster will appear here as you mark picks.</span>
        )}
      </div>
      <div className="metric-grid" id="roster-need-grid">
        {POSITION_KEYS.map((position) => (
          <div key={position} className="metric">
            <span className="label">{position} need</span>
            <span className="value">{boardState.rosterNeed[position] || 0}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
