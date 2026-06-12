import { useEffect, useMemo } from 'react';
import type { DashboardPayload } from '../payload/load';
import {
  buildPlayerSummary,
  filterBoardRows,
  formatNumber,
  formatPercent,
  type DecisionRow,
} from '../state/buildBoardState';
import { safeLower, type DraftStore } from '../state/draftState';
import { useBoardState } from '../state/useBoardState';
import { useDraftStore } from '../state/useDraftStore';

declare global {
  interface Window {
    __ffbayesBoardState?: {
      scoringPreset: string;
      selectedPlayer: string;
      rows: Array<{
        player_name: string;
        position: string;
        proj_points_mean?: number;
        draft_score?: number;
        status?: string;
      }>;
    };
  }
}

function presetSummary(payload: DashboardPayload, scoringPreset: string): string {
  const entry = payload.scoring_presets?.[scoringPreset] as
    | { label?: string; key?: string }
    | undefined;
  const label = entry?.label || entry?.key || 'Current';
  return label;
}

function rowClassName(row: DecisionRow, selectedPlayer: string): string {
  return [
    safeLower(row.player_name) === safeLower(selectedPlayer) ? 'is-selected' : '',
    row.status === 'taken' ? 'is-taken' : '',
    row.status === 'mine' ? 'is-mine' : '',
    row.status === 'queued' ? 'is-queued' : '',
  ]
    .filter(Boolean)
    .join(' ');
}

function BoardRow(props: {
  row: DecisionRow;
  selectedPlayer: string;
  onSelect: (playerName: string) => void;
  onAction: (action: 'queue' | 'taken' | 'mine', playerName: string) => void;
}) {
  const { row, selectedPlayer, onSelect, onAction } = props;
  return (
    <tr
      className={rowClassName(row, selectedPlayer)}
      data-player-row={row.player_name}
      onClick={(event) => {
        if ((event.target as HTMLElement).closest('button')) {
          return;
        }
        onSelect(row.player_name);
      }}
    >
      <td>
        <div className="item-title">{row.player_name}</div>
        <div className="tiny">
          {row.position}
          {row.team ? ` • ${String(row.team)}` : ''} • {String(row.draft_tier || '')}
        </div>
      </td>
      <td>
        <span className={`status-badge ${row.status}`}>{String(row.status ?? 'available')}</span>
      </td>
      <td>{formatNumber(row.draft_score)}</td>
      <td>
        {formatNumber(row.simple_vor_proxy)}{' '}
        <span className="tiny">(rank {row.simple_vor_rank || '-'})</span>
      </td>
      <td>{formatPercent(row.availability_to_next_pick)}</td>
      <td className="tiny">{buildPlayerSummary(row)}</td>
      <td>
        <div className="action-group">
          <button
            type="button"
            data-action="queue"
            data-player={row.player_name}
            onClick={(event) => {
              event.stopPropagation();
              onAction('queue', row.player_name);
            }}
          >
            {row.status === 'queued' ? 'Unqueue' : 'Queue'}
          </button>
          <button
            type="button"
            data-action="taken"
            data-player={row.player_name}
            onClick={(event) => {
              event.stopPropagation();
              onAction('taken', row.player_name);
            }}
          >
            {row.status === 'taken' ? 'Clear' : 'Taken'}
          </button>
          <button
            type="button"
            data-action="mine"
            data-player={row.player_name}
            onClick={(event) => {
              event.stopPropagation();
              onAction('mine', row.player_name);
            }}
          >
            {row.status === 'mine' ? 'Unmark' : 'Mine'}
          </button>
        </div>
      </td>
    </tr>
  );
}

export function PlayerBoard(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const boardState = useBoardState(payload, store);
  const filteredRows = useMemo(
    () => filterBoardRows(boardState.rows, state.search),
    [boardState.rows, state.search],
  );

  useEffect(() => {
    window.__ffbayesBoardState = {
      scoringPreset: state.scoringPreset,
      selectedPlayer: state.selectedPlayer,
      rows: boardState.rows.map((row) => ({
        player_name: row.player_name,
        position: row.position,
        proj_points_mean: Number(row.proj_points_mean),
        draft_score: Number(row.draft_score),
        status: row.status,
      })),
    };
  }, [boardState.rows, state.scoringPreset, state.selectedPlayer]);

  function handleAction(action: 'queue' | 'taken' | 'mine', playerName: string): void {
    if (action === 'queue') {
      store.toggleQueue(playerName);
      return;
    }
    if (action === 'taken') {
      store.markTaken(playerName);
      return;
    }
    store.markMine(playerName);
  }

  return (
    <section className="panel strong">
      <div className="split">
        <div>
          <h2>Full Player Board</h2>
          <p className="subtle">
            Search every player, keep taken rows visible if you want, and click a row to inspect
            the model reasoning.
          </p>
        </div>
        <span className="pill" id="board-count-pill">
          {filteredRows.length} shown
        </span>
      </div>
      <div className="board-controls">
        <div className="search-wrap">
          <input
            id="player-search"
            type="search"
            placeholder="Search name, team, or position"
            value={state.search}
            onChange={(event) => store.setSearch(event.target.value)}
          />
        </div>
        <div className="board-summary">
          <span id="board-summary-text">
            Showing {filteredRows.length} of {boardState.rows.length} players
          </span>
          <span id="preset-summary-text">
            Preset: {presetSummary(payload, state.scoringPreset)} • Risk:{' '}
            {(state.riskTolerance || 'medium').toUpperCase()}
          </span>
          <span>Marking Taken or Mine advances the draft automatically.</span>
        </div>
      </div>
      <div className="board-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Status</th>
              <th>Board value score</th>
              <th>VOR proxy</th>
              <th>Next-pick survival</th>
              <th>Why now</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="board-table">
            {!filteredRows.length ? (
              <tr>
                <td colSpan={7} className="empty">
                  No players match the current filters.
                </td>
              </tr>
            ) : (
              filteredRows.map((row) => (
                <BoardRow
                  key={row.player_name}
                  row={row}
                  selectedPlayer={state.selectedPlayer}
                  onSelect={(playerName) => store.selectPlayer(playerName)}
                  onAction={handleAction}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
