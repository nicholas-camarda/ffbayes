import type { FfbayesDashboardPayload } from './types.generated';

export type DashboardPayload = FfbayesDashboardPayload;

declare global {
  interface Window {
    FFBAYES_DASHBOARD?: unknown;
  }
}

export async function loadPayload(): Promise<DashboardPayload> {
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
