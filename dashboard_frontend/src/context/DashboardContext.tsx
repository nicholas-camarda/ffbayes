import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import type { DashboardPayload } from '../payload/load';
import type { DraftStore } from '../state/draftState';

export interface DashboardContextValue {
  payload: DashboardPayload;
  store: DraftStore;
}

const DashboardContext = createContext<DashboardContextValue | null>(null);

export function DashboardProvider(props: {
  payload: DashboardPayload;
  store: DraftStore;
  children: ReactNode;
}) {
  const { payload, store, children } = props;
  return (
    <DashboardContext.Provider value={{ payload, store }}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard(): DashboardContextValue {
  const value = useContext(DashboardContext);
  if (!value) {
    throw new Error('useDashboard must be used within DashboardProvider');
  }
  return value;
}
