import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { EvidencePanel } from './EvidencePanel';
import { FreshnessPanel } from './FreshnessPanel';
import { ProvenanceBanner } from './ProvenanceBanner';

const basePayload = fixture as unknown as DashboardPayload;

it('renders strategy summary and season rows from the fixture', () => {
  render(<EvidencePanel payload={basePayload} />);
  const panel = document.getElementById('decision-evidence');
  expect(panel?.textContent).toContain('Decision evidence');
  expect(panel?.textContent).toContain('draft_score');
  expect(panel?.textContent).toContain('2025');
  expect(panel?.textContent).toContain('10.00');
  expect(panel?.querySelectorAll('td').length).toBeGreaterThanOrEqual(4);
});

it('shows a SectionGate notice when decision evidence is unavailable', () => {
  const unavailablePayload = {
    ...basePayload,
    decision_evidence: {
      available: false,
      status: 'unavailable',
      reason_unavailable: 'Backtest evidence is unavailable for this board.',
      strategy_summary: [],
      season_rows: [],
    },
  } as unknown as DashboardPayload;

  render(<EvidencePanel payload={unavailablePayload} />);
  expect(screen.queryByText('draft_score')).not.toBeInTheDocument();
  expect(screen.getByText(/Backtest evidence is unavailable/i)).toBeInTheDocument();
});

it('renders ProvenanceBanner only when publish_provenance exists', () => {
  const { rerender } = render(
    <>
      <FreshnessPanel payload={basePayload} />
      {basePayload.publish_provenance ? <ProvenanceBanner payload={basePayload} /> : null}
    </>,
  );
  expect(screen.getByText('Freshness and provenance')).toBeInTheDocument();
  expect(screen.queryByText('Publish provenance')).not.toBeInTheDocument();

  const stagedPayload = {
    ...basePayload,
    publish_provenance: {
      schema_version: 'publish_provenance_v1',
      published_at: '2026-04-24T12:45:18',
      dashboard_generated_at: '2026-04-24T12:42:12',
      source_html: 'draft_board_2026.html',
      source_payload: 'dashboard_payload_2026.json',
      surface_sync: {
        status: 'synchronized',
        detail: 'The staged HTML and staged payload were written together during dashboard staging.',
      },
    },
  } as unknown as DashboardPayload;

  rerender(
    <>
      <FreshnessPanel payload={stagedPayload} />
      {stagedPayload.publish_provenance ? <ProvenanceBanner payload={stagedPayload} /> : null}
    </>,
  );
  expect(screen.getByText('Publish provenance')).toBeInTheDocument();
  expect(screen.getByText(/Pages staged/i)).toBeInTheDocument();
});
