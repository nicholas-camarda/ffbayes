import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadPayload } from './load';

afterEach(() => {
  delete window.FFBAYES_DASHBOARD;
  vi.restoreAllMocks();
});

describe('loadPayload', () => {
  it('returns the embedded payload when present', async () => {
    (window as any).FFBAYES_DASHBOARD = { generated_at: 'x', decision_table: [] };
    const payload = await loadPayload();
    expect(payload.generated_at).toBe('x');
  });

  it('falls back to fetching dashboard_payload.json', async () => {
    (window as any).FFBAYES_DASHBOARD = null;
    const fake = { generated_at: 'fetched', decision_table: [] };
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(fake))));
    const payload = await loadPayload();
    expect(payload.generated_at).toBe('fetched');
  });

  it('throws a clear error when neither source is available', async () => {
    (window as any).FFBAYES_DASHBOARD = null;
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    await expect(loadPayload()).rejects.toThrow(/dashboard payload/i);
  });
});
