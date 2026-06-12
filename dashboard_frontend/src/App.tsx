import { useEffect, useMemo, useState } from 'react';
import { AppShell } from './components/AppShell';
import { DashboardProvider } from './context/DashboardContext';
import { loadPayload, type DashboardPayload } from './payload/load';
import { createStoreFromPayload } from './state/createStoreFromPayload';

export default function App() {
  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const store = useMemo(() => (payload ? createStoreFromPayload(payload) : null), [payload]);

  useEffect(() => {
    let active = true;
    loadPayload()
      .then((loaded) => {
        if (!active) {
          return;
        }
        setPayload(loaded);
        setError(null);
      })
      .catch((loadError: unknown) => {
        if (!active) {
          return;
        }
        const message = loadError instanceof Error ? loadError.message : String(loadError);
        setError(message);
        setPayload(null);
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="shell">
        <h1>FFBayes Draft War Room</h1>
        <p className="subtle">Loading dashboard payload…</p>
      </div>
    );
  }

  if (error || !payload || !store) {
    return (
      <div className="shell error-boundary" role="alert">
        <h1>FFBayes Draft War Room</h1>
        <p className="error-message">{error || 'Dashboard payload is unavailable.'}</p>
        <p className="error-hint">
          The dashboard payload may be malformed — regenerate with{' '}
          <code>ffbayes stage-dashboard --year &lt;year&gt;</code>
        </p>
      </div>
    );
  }

  return (
    <DashboardProvider payload={payload} store={store}>
      <AppShell payload={payload} store={store} />
    </DashboardProvider>
  );
}
