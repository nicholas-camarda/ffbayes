import type { FfbayesDashboardPayload } from './types.generated';

export type DashboardPayload = FfbayesDashboardPayload;

export const PAYLOAD_SCRIPT_ID = 'ffbayes-dashboard-payload';

declare global {
  interface Window {
    FFBAYES_DASHBOARD?: unknown;
  }
}

function readEmbeddedPayloadScript(): DashboardPayload | null {
  const element = document.getElementById(PAYLOAD_SCRIPT_ID);
  const raw = element?.textContent?.trim();
  if (!raw || raw === '__PAYLOAD_JSON__') {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === 'object') {
      return parsed as DashboardPayload;
    }
  } catch {
    return null;
  }
  return null;
}

export async function loadPayload(): Promise<DashboardPayload> {
  const fromScript = readEmbeddedPayloadScript();
  if (fromScript) {
    return fromScript;
  }

  const embedded = window.FFBAYES_DASHBOARD;
  if (embedded && typeof embedded === 'object') {
    return embedded as DashboardPayload;
  }

  const response = await fetch('./dashboard_payload.json');
  if (!response.ok) {
    throw new Error(
      `Could not load the dashboard payload: no embedded payload and ` +
        `fetching ./dashboard_payload.json returned HTTP ${response.status}.`
    );
  }
  return (await response.json()) as DashboardPayload;
}
