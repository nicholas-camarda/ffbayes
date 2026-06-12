import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadPayload, PAYLOAD_SCRIPT_ID } from './load';

afterEach(() => {
  delete window.FFBAYES_DASHBOARD;
  document.getElementById(PAYLOAD_SCRIPT_ID)?.remove();
  vi.restoreAllMocks();
});

describe('loadPayload', () => {
  it('returns the payload from the embedded JSON script tag', async () => {
    const script = document.createElement('script');
    script.id = PAYLOAD_SCRIPT_ID;
    script.type = 'application/json';
    script.textContent = JSON.stringify({ generated_at: 'script', decision_table: [] });
    document.head.appendChild(script);

    const payload = await loadPayload();
    expect(payload.generated_at).toBe('script');
  });

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
