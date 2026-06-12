import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { SectionGate } from './SectionGate';

it('renders children when the section is available', () => {
  render(
    <SectionGate section={{ available: true, status: 'available' }} title="Evidence">
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.getByText('panel-body')).toBeInTheDocument();
});

it('renders an unavailable notice with the reason when absent', () => {
  render(
    <SectionGate
      section={{ available: false, status: 'unavailable', reason_unavailable: 'no data' }}
      title="Evidence"
    >
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.queryByText('panel-body')).not.toBeInTheDocument();
  expect(screen.getByText(/no data/)).toBeInTheDocument();
});

it('treats a missing section object as unavailable without crashing', () => {
  render(
    <SectionGate section={undefined} title="Evidence">
      <div>panel-body</div>
    </SectionGate>
  );
  expect(screen.queryByText('panel-body')).not.toBeInTheDocument();
  expect(screen.getByText(/not available/i)).toBeInTheDocument();
});
