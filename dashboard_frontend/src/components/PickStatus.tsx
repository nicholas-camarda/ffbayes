import type { DashboardPayload } from '../payload/load';
import { nextPickNumber } from '../state/draftState';
import type { DraftStore } from '../state/draftState';
import { useDraftStore } from '../state/useDraftStore';

export function PickStatus(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const state = useDraftStore(store);
  const currentPick = state.currentPickNumber;
  const nextPick =
    payload.next_pick_number
    ?? nextPickNumber(currentPick, state.draftPosition, state.leagueSize);
  const presetLabel =
    payload.scoring_presets?.[state.scoringPreset] as { label?: string } | undefined;

  const pills = [
    ['Current pick', currentPick],
    ['Next pick', nextPick],
    ['League', `${state.leagueSize}-team`],
    ['Draft slot', state.draftPosition],
    ['Preset', presetLabel?.label || state.scoringPreset],
    ['Taken', state.takenPlayers.length],
    ['Yours', state.yourPlayers.length],
  ] as const;

  return (
    <div className="pill-row" id="status-pills">
      {pills.map(([label, value]) => (
        <span key={label} className="pill">
          {label}: {value}
        </span>
      ))}
    </div>
  );
}
