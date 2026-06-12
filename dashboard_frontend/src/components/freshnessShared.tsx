export type FreshnessRow = {
  source_name?: string;
  label?: string;
  status?: string;
  override_used?: boolean;
  freshness_days?: number;
  latest_expected_year?: number;
  latest_found_year?: number;
};

type FreshnessContext = {
  status?: string;
  override_used?: boolean;
  warnings?: string[];
};

function explainFreshnessState(status: string | undefined, overrideUsed: boolean | undefined): string {
  const normalized = (status || '').trim().toLowerCase();
  if (!normalized || normalized === 'fresh') {
    return '';
  }
  if (normalized === 'mixed') {
    return 'Some analysis inputs are current while others are degraded, so trust the board with extra caution.';
  }
  if (normalized === 'stale') {
    return 'The rendered dashboard surface is out of sync with its paired payload or expected source artifacts.';
  }
  if (normalized === 'degraded') {
    return 'The board was generated with degraded inputs, so some recommendation support may be weaker than usual.';
  }
  if (overrideUsed) {
    return 'A freshness override was used to force a degraded analysis window.';
  }
  return `Freshness status is ${status}.`;
}

export function FreshnessNotice(props: { freshness: FreshnessContext | undefined }) {
  const { freshness } = props;
  const warnings = Array.isArray(freshness?.warnings) ? freshness.warnings : [];
  if (warnings.length) {
    return <div className="notice">{warnings.join(' ')}</div>;
  }
  const explanation = explainFreshnessState(freshness?.status, freshness?.override_used);
  if (!explanation) {
    return null;
  }
  return (
    <div className="notice">
      {explanation} Detailed warning text is unavailable in this payload.
    </div>
  );
}

export function formatTimestamp(value: string | undefined): string {
  if (!value) {
    return 'n/a';
  }
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? String(value) : dt.toLocaleString();
}
