import { beforeEach, describe, expect, it, vi } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createDraftStore } from '../state/draftState';
import {
  FINALIZED_SCHEMA_VERSION,
  buildBundle,
  buildFinalizedDraftPayload,
  buildFinalizedSnapshotHtml,
  buildFinalizedSummaryHtml,
} from './buildBundle';
import { buildBoardState } from '../state/buildBoardState';

const payload = fixture as unknown as DashboardPayload;

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-06-12T15:30:00.000Z'));
});

describe('buildBundle', () => {
  it('returns finalized_draft_v1 JSON and HTML with legacy titles', () => {
    const store = createDraftStore({ yourPlayers: ['Test Player'] });
    const state = store.getState();
    const bundle = buildBundle(state, payload);

    expect(bundle.json.schema_version).toBe(FINALIZED_SCHEMA_VERSION);
    expect(bundle.json.schema_version).toBe('finalized_draft_v1');
    expect(bundle.json.exported_at).toBe('2026-06-12T15:30:00.000Z');
    expect(bundle.snapshotHtml).toContain('<title>FFBayes Finalized Draft Snapshot</title>');
    expect(bundle.snapshotHtml).toContain('FFBayes Finalized Draft Snapshot</h1>');
    expect(bundle.summaryHtml).toContain('<title>FFBayes Post-Draft Summary</title>');
    expect(bundle.summaryHtml).toContain('FFBayes Post-Draft Summary</h1>');
  });

  it('includes drafted roster players in both HTML exports', () => {
    const store = createDraftStore({ yourPlayers: ['Test Player'] });
    const state = store.getState();
    const bundle = buildBundle(state, payload);

    expect(bundle.json.drafted_players).toHaveLength(1);
    expect(bundle.json.drafted_players[0]?.player_name).toBe('Test Player');
    expect(bundle.snapshotHtml).toContain('Test Player');
    expect(bundle.summaryHtml).toContain('Test Player');
  });

  it('builds starter/bench split and summary metrics from drafted rows', () => {
    const store = createDraftStore({ yourPlayers: ['Test Player'] });
    const state = store.getState();
    const boardState = buildBoardState(payload, state);
    const json = buildFinalizedDraftPayload(boardState, state, payload);

    expect(json.starters).toHaveLength(1);
    expect(json.starters[0]?.lineup_slot).toBe('RB1');
    expect(json.summary_metrics.risk_style).toBe('Ceiling-heavy');
    expect(buildFinalizedSnapshotHtml(json)).toContain('RB1');
    expect(buildFinalizedSummaryHtml(json)).toContain('Final Roster Recap');
  });
});
