import { fireEvent, render } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import * as buildBundleModule from '../finalize/buildBundle';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import { FinalizePanel } from './FinalizePanel';

const basePayload = fixture as unknown as DashboardPayload;

beforeEach(() => {
  window.localStorage.clear();
});

it('does not render finalize controls when publish_provenance exists', () => {
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
  const store = createStoreFromPayload(stagedPayload);

  render(<FinalizePanel payload={stagedPayload} store={store} />);

  expect(document.getElementById('finalize-button')).not.toBeInTheDocument();
  expect(document.getElementById('finalize-note')).not.toBeInTheDocument();
});

it('renders finalize controls for local payloads without publish_provenance', () => {
  const store = createStoreFromPayload(basePayload);

  render(<FinalizePanel payload={basePayload} store={store} />);

  expect(document.getElementById('finalize-button')).toBeInTheDocument();
  expect(document.getElementById('finalize-note')).toBeInTheDocument();
});

it('downloads three finalized artifacts when finalize is clicked', () => {
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: { protocol: 'file:' },
  });
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  const downloadTextFile = vi.spyOn(buildBundleModule, 'downloadTextFile').mockImplementation(() => {});
  const store = createStoreFromPayload(basePayload);
  store.markMine('Test Player');

  render(<FinalizePanel payload={basePayload} store={store} />);
  const button = document.getElementById('finalize-button');
  fireEvent.click(button!);

  expect(downloadTextFile).toHaveBeenCalledTimes(3);
  expect(downloadTextFile.mock.calls[0]?.[2]).toBe('application/json');
  expect(downloadTextFile.mock.calls[1]?.[2]).toBe('text/html');
  expect(downloadTextFile.mock.calls[2]?.[2]).toBe('text/html');
});
