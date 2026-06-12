import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

function BrokenChild(): never {
  throw new Error('payload parse failed');
}

it('shows a fallback with a regeneration hint when a child throws', () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

  render(
    <ErrorBoundary>
      <BrokenChild />
    </ErrorBoundary>,
  );

  expect(screen.getByRole('alert')).toBeInTheDocument();
  expect(screen.getByText(/payload parse failed/)).toBeInTheDocument();
  expect(screen.getByText(/ffbayes stage-dashboard --year/i)).toBeInTheDocument();

  consoleError.mockRestore();
});
