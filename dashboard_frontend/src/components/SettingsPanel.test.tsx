import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import { SettingsPanel } from './SettingsPanel';

const payload = fixture as unknown as DashboardPayload;

beforeEach(() => {
  window.localStorage.clear();
});

it('renders league size, draft position, and scoring preset from the fixture', () => {
  const store = createStoreFromPayload(payload);
  render(<SettingsPanel payload={payload} store={store} />);

  expect(screen.getByLabelText('League size')).toHaveValue(10);
  expect(screen.getByLabelText('Draft position')).toHaveValue(10);
  expect(screen.getByLabelText('Scoring preset')).toHaveValue('half_ppr');
});

it('updates the store when the scoring preset changes', async () => {
  const user = userEvent.setup();
  const store = createStoreFromPayload(payload);
  const setScoringPreset = vi.spyOn(store, 'setScoringPreset');

  render(<SettingsPanel payload={payload} store={store} />);
  await user.selectOptions(screen.getByLabelText('Scoring preset'), 'ppr');

  expect(setScoringPreset).toHaveBeenCalledWith('ppr');
  expect(store.getState().scoringPreset).toBe('ppr');
});
