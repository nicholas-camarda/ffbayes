/* AUTO-GENERATED from dashboard_payload.schema.json — run npm run generate:types */

export interface FfbayesDashboardPayload {
  dashboard_schema_version: 1;
  generated_at: string;
  league_settings: {
    league_size: number;
    draft_position: number;
    scoring_type: string;
    roster_spots: {
      [k: string]: unknown;
    };
    [k: string]: unknown;
  };
  /**
   * @minItems 1
   */
  decision_table: [
    {
      player_name: string;
      position: string;
      [k: string]: unknown;
    },
    ...{
      player_name: string;
      position: string;
      [k: string]: unknown;
    }[]
  ];
  recommendation_summary: unknown[];
  decision_evidence: {
    available: boolean;
    status: string;
    [k: string]: unknown;
  };
  runtime_controls?: {
    [k: string]: unknown;
  };
  current_pick_number?: number;
  next_pick_number?: number;
  current_draft_context_defaults?: {
    [k: string]: unknown;
  };
  selected_player?: string | null;
  recommendation_inputs?: unknown[];
  live_state?: {
    [k: string]: unknown;
  };
  scoring_presets?: {
    [k: string]: unknown;
  };
  position_summary?:
    | unknown[]
    | {
        [k: string]: unknown;
      };
  tier_cliffs?: unknown[];
  roster_scenarios?: unknown[];
  source_freshness?: unknown[];
  analysis_provenance?: {
    [k: string]: unknown;
  };
  player_forecast_validation?: {
    [k: string]: unknown;
  };
  war_room_visuals?: {
    [k: string]: unknown;
  };
  backtest?: {
    [k: string]: unknown;
  };
  supporting_math?: {
    [k: string]: unknown;
  };
  metric_glossary?: {
    [k: string]: unknown;
  };
  model_overview?: {
    [k: string]: unknown;
  };
  bayesian_vor_summary?: {
    [k: string]: unknown;
  } | null;
  publish_provenance?: {
    [k: string]: unknown;
  };
  [k: string]: unknown;
}
