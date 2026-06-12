import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import fixture from '../../../tests/fixtures/dashboard_payload_minimal.json';
import type { DashboardPayload } from '../payload/load';
import { createStoreFromPayload } from '../state/createStoreFromPayload';
import { PickStatus } from './PickStatus';

const payload = {
  ...(fixture as unknown as DashboardPayload),
  current_pick_number: 10,
  next_pick_number: 11,
};

it('shows derived current pick and next pick numbers from draft state', () => {
  const store = createStoreFromPayload(payload);
  render(<PickStatus payload={payload} store={store} />);

  expect(screen.getByText('Current pick: 1')).toBeInTheDocument();
  expect(screen.getByText('Next pick: 11')).toBeInTheDocument();
});
