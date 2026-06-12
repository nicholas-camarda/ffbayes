import type { DashboardPayload } from '../payload/load';
import { createDraftStore, type CreateDraftStoreOptions } from './draftState';

export function createStoreFromPayload(payload: DashboardPayload) {
  const league = payload.league_settings;
  const defaults = payload.current_draft_context_defaults ?? {};
  const runtime = payload.runtime_controls ?? {};
  const rosterSpots = league.roster_spots as Record<string, number> | undefined;

  const options: CreateDraftStoreOptions = {
    initialPickNumber:
      payload.current_pick_number
      ?? (defaults.current_pick_number as number | undefined)
      ?? league.draft_position,
    draftPosition: league.draft_position,
    leagueSize: league.league_size,
    scoringPreset:
      (runtime.active_scoring_preset as string | undefined) ?? 'half_ppr',
    riskTolerance: (league.risk_tolerance as string | undefined) ?? 'medium',
    benchSlots: (league.bench_slots as number | undefined) ?? 6,
    rosterSpots,
    takenPlayers: (defaults.taken_players as string[] | undefined) ?? [],
    yourPlayers: (defaults.your_players as string[] | undefined) ?? [],
    selectedPlayer: payload.selected_player ?? '',
  };

  return createDraftStore({ ...options, payload });
}
