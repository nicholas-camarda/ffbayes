import type { DashboardPayload } from '../payload/load';
import type { DraftStore } from '../state/draftState';
import { useDraftStore } from '../state/useDraftStore';

interface ScoringPresetEntry {
  key?: string;
  label?: string;
  available?: boolean;
}

export function SettingsPanel(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const runtime = payload.runtime_controls ?? {};
  const supportedPresets = (runtime.supported_scoring_presets as string[] | undefined) ?? [
    'standard',
    'half_ppr',
    'ppr',
  ];
  const scoringPresets = (payload.scoring_presets ?? {}) as Record<string, ScoringPresetEntry>;
  const currentPreset = scoringPresets[state.scoringPreset];
  const riskOptions = (runtime.risk_tolerance_options as string[] | undefined) ?? [
    'low',
    'medium',
    'high',
  ];

  return (
    <section className="panel">
      <h2>Draft Controls</h2>
      <p className="subtle">
        These settings are dashboard-first. Changes update the board immediately.
      </p>
      <div className="settings-grid">
        <div className="field">
          <label htmlFor="scoring-preset">Scoring preset</label>
          <select
            id="scoring-preset"
            value={state.scoringPreset}
            onChange={(event) => store.setScoringPreset(event.target.value)}
          >
            {supportedPresets.map((key) => {
              const entry = scoringPresets[key];
              return (
                <option key={key} value={key}>
                  {entry?.label || key}
                </option>
              );
            })}
          </select>
        </div>
        <div className="field">
          <label htmlFor="risk-tolerance">Risk tolerance</label>
          <select
            id="risk-tolerance"
            value={state.riskTolerance}
            onChange={(event) => store.setRiskTolerance(event.target.value)}
          >
            {riskOptions.map((option) => (
              <option key={option} value={option}>
                {option.charAt(0).toUpperCase() + option.slice(1)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="league-size">League size</label>
          <input
            id="league-size"
            type="number"
            min={2}
            max={20}
            value={state.leagueSize}
            onChange={(event) => store.setLeagueSize(Number(event.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="draft-position">Draft position</label>
          <input
            id="draft-position"
            type="number"
            min={1}
            max={20}
            value={state.draftPosition}
            onChange={(event) => store.setDraftPosition(Number(event.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="bench-slots">Bench slots</label>
          <input
            id="bench-slots"
            type="number"
            min={0}
            max={12}
            value={state.benchSlots}
            onChange={(event) => store.setBenchSlots(Number(event.target.value))}
          />
        </div>
        <div className="field full">
          <div className="notice" id="preset-notice">
            {`Starting from ${currentPreset?.label || state.scoringPreset}. The config-selected preset seeds this board's default view, and you can switch among Standard (0.0 PPR), Half PPR (0.5), and Full PPR (1.0) here without regeneration.`}
          </div>
        </div>
      </div>
    </section>
  );
}
