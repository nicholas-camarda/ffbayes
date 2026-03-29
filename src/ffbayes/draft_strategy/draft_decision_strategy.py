#!/usr/bin/env python3
"""
Draft decision strategy entrypoint.

This module is the thin compatibility layer that turns the new draft decision
engine into the public CLI/script interface used by the repo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    DraftContext,
    DraftDecisionArtifacts,
    LeagueSettings,
    build_draft_decision_artifacts,
    save_draft_decision_artifacts,
)
from ffbayes.utils.path_constants import (
    get_dashboard_payload_path,
    get_draft_board_path,
    get_draft_decision_backtest_path,
    get_draft_strategy_dir,
    get_unified_dataset_csv_path,
    get_unified_dataset_path,
)
from ffbayes.utils.strategy_path_generator import get_bayesian_strategy_path


logger = logging.getLogger(__name__)


@dataclass
class DraftConfig:
    """Compatibility configuration object for older callers."""

    league_size: int = 10
    draft_position: int = 10
    scoring_type: str = 'PPR'
    roster_positions: dict[str, int] = field(default_factory=lambda: {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1})
    risk_tolerance: str = 'medium'

    def to_league_settings(self) -> LeagueSettings:
        return LeagueSettings(
            league_size=self.league_size,
            draft_position=self.draft_position,
            scoring_type=self.scoring_type,
            risk_tolerance=self.risk_tolerance,
            roster_spots=self.roster_positions,
        )


class TierBasedStrategy:
    """Compatibility facade that uses the new draft decision table."""

    def __init__(self, predictions: pd.DataFrame):
        self.predictions = predictions.copy()

    def create_tiers(self, num_tiers: int = 5) -> dict[str, list[str]]:
        settings = LeagueSettings()
        context = DraftContext(current_pick_number=settings.draft_position)
        artifacts = build_draft_decision_artifacts(self.predictions, settings, context)
        tiers: dict[str, list[str]] = {}
        for tier_name, group in artifacts.decision_table.groupby('draft_tier'):
            tiers[tier_name] = group.sort_values('draft_score', ascending=False)['player_name'].tolist()
        return tiers

    def generate_pick_options(self, draft_position: int, league_size: int, config: DraftConfig) -> dict[str, Any]:
        settings = config.to_league_settings()
        context = DraftContext(current_pick_number=draft_position, drafted_players=set())
        artifacts = build_draft_decision_artifacts(self.predictions, settings, context)
        top = artifacts.recommendations.head(7)
        return {
            'primary_targets': top.head(3)['player_name'].tolist(),
            'backup_options': top.iloc[3:7]['player_name'].tolist(),
            'position_priority': _position_priority(top),
            'reasoning': top.iloc[0]['rationale'] if not top.empty else 'No recommendations available',
            'uncertainty_analysis': {
                'risk_tolerance': settings.risk_tolerance,
                'primary_avg_uncertainty': float(top.head(3)['fragility_score'].mean()) if not top.empty else 0.0,
                'backup_avg_uncertainty': float(top.iloc[3:7]['fragility_score'].mean()) if len(top) > 3 else 0.0,
                'overall_uncertainty': float(top['fragility_score'].mean()) if not top.empty else 0.0,
            },
            'confidence_intervals': {
                row['player_name']: {
                    'floor': float(row['proj_points_floor']),
                    'ceiling': float(row['proj_points_ceiling']),
                }
                for _, row in top.iterrows()
            },
        }


class BayesianDraftStrategy:
    """Compatibility wrapper around the new draft decision engine."""

    def __init__(self, predictions: pd.DataFrame | None = None, season_history: pd.DataFrame | None = None):
        self.predictions = predictions
        self.season_history = season_history

    def build(self, config: DraftConfig | None = None, current_pick_number: int | None = None) -> DraftDecisionArtifacts:
        config = config or DraftConfig()
        settings = config.to_league_settings()
        context = DraftContext(current_pick_number=current_pick_number or settings.draft_position)
        if self.predictions is None:
            self.predictions = _load_player_frame()
        return build_draft_decision_artifacts(self.predictions, settings, context, season_history=self.season_history)

    def save(self, output_dir: Path | str | None = None, year: int | None = None, config: DraftConfig | None = None) -> dict[str, Path]:
        artifacts = self.build(config=config)
        output_dir = Path(output_dir) if output_dir is not None else get_draft_strategy_dir(year or datetime.now().year)
        return save_draft_decision_artifacts(artifacts, output_dir, year=year)


class TeamConstructionOptimizer:
    """Compatibility optimizer that returns the top roster scenarios."""

    def __init__(self, predictions: pd.DataFrame | None = None):
        self.predictions = predictions

    def optimize_team_construction(self, config: DraftConfig | None = None) -> dict[str, Any]:
        config = config or DraftConfig()
        artifacts = BayesianDraftStrategy(self.predictions).build(config=config)
        top = artifacts.roster_scenarios.head(3).to_dict(orient='records')
        return {
            'recommended_scenarios': top,
            'league_settings': artifacts.league_settings.to_dict(),
            'decision_summary': artifacts.metadata,
        }


class UncertaintyAwareSelector:
    """Compatibility selector that surfaces the current top choices."""

    def __init__(self, predictions: pd.DataFrame | None = None):
        self.predictions = predictions

    def select(self, config: DraftConfig | None = None) -> pd.DataFrame:
        config = config or DraftConfig()
        artifacts = BayesianDraftStrategy(self.predictions).build(config=config)
        return artifacts.recommendations.copy()


def _load_player_frame() -> pd.DataFrame:
    csv_path = get_unified_dataset_csv_path()
    json_path = get_unified_dataset_path()
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.read_json(json_path)
    raise FileNotFoundError(
        f'No unified player dataset found at {csv_path} or {json_path}. '
        'Run the data pipeline first.'
    )


def _load_season_history() -> pd.DataFrame | None:
    from ffbayes.utils.path_constants import SEASON_DATASETS_DIR

    files = sorted(SEASON_DATASETS_DIR.glob('*season.csv'))
    if not files:
        return None
    return pd.concat((pd.read_csv(file_path) for file_path in files), ignore_index=True)


def _position_priority(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return 'Balanced'
    top_positions = recommendations.head(5)['position'].value_counts()
    if top_positions.empty:
        return 'Balanced'
    return top_positions.index[0]


def _compatibility_payload(artifacts: DraftDecisionArtifacts) -> dict[str, Any]:
    picks = {}
    recommendation_window = artifacts.recommendations.copy()
    if recommendation_window.empty:
        recommendation_window = artifacts.decision_table.head(20).copy()
        recommendation_window['rationale'] = recommendation_window['why_flags']
        recommendation_window['availability_to_next_pick'] = recommendation_window['availability_at_pick']

    round_count = artifacts.league_settings.round_count()
    for round_number in range(1, round_count + 1):
        if round_number <= 5:
            window = recommendation_window.sort_values(['draft_score', 'proj_points_mean'], ascending=[False, False]).head(7)
        elif round_number <= 10:
            window = recommendation_window.sort_values(['starter_delta', 'draft_score'], ascending=[False, False]).head(7)
        else:
            window = recommendation_window.sort_values(['upside_score', 'draft_score'], ascending=[False, False]).head(7)
        picks[f'Pick {round_number}'] = {
            'primary_targets': window.head(3)['player_name'].tolist(),
            'backup_options': window.iloc[3:7]['player_name'].tolist(),
            'position_priority': _position_priority(window),
            'reasoning': window.iloc[0]['rationale'] if not window.empty else 'No recommendations available',
            'uncertainty_analysis': {
                'risk_tolerance': artifacts.league_settings.risk_tolerance,
                'primary_avg_uncertainty': float(window.head(3)['fragility_score'].mean()) if not window.empty else 0.0,
                'backup_avg_uncertainty': float(window.iloc[3:7]['fragility_score'].mean()) if len(window) > 3 else 0.0,
                'overall_uncertainty': float(window['fragility_score'].mean()) if not window.empty else 0.0,
            },
            'confidence_intervals': {
                row['player_name']: {
                    'floor': float(row['proj_points_floor']),
                    'ceiling': float(row['proj_points_ceiling']),
                }
                for _, row in window.iterrows()
            },
        }

    return {
        'strategy': picks,
        'metadata': {
            **artifacts.metadata,
            'draft_position': artifacts.league_settings.draft_position,
            'league_size': artifacts.league_settings.league_size,
            'scoring_type': artifacts.league_settings.scoring_type,
            'risk_tolerance': artifacts.league_settings.risk_tolerance,
            'generation_timestamp': artifacts.metadata['generated_at'],
            'position_scarcity': artifacts.decision_table.groupby('position')['player_name'].count().to_dict(),
            'decision_table_columns': list(artifacts.decision_table.columns),
        },
    }


def main() -> int:
    """Generate the draft board, dashboard payload, and compatibility JSON."""
    logging.basicConfig(level=logging.INFO)
    logger.info('Generating draft decision artifacts...')

    current_year = datetime.now().year
    try:
        from ffbayes.utils.config_loader import get_config

        config_loader = get_config()
        league_settings = LeagueSettings.from_mapping(
            {
                'league_settings': {
                    'league_size': config_loader.get_league_setting('league_size'),
                    'draft_position': config_loader.get_league_setting('draft_position'),
                    'scoring_type': config_loader.get_league_setting('scoring_type'),
                    'ppr_value': config_loader.get_league_setting('ppr_value') if hasattr(config_loader, 'get_league_setting') else 0.5,
                    'risk_tolerance': config_loader.get_league_setting('risk_tolerance'),
                }
            }
        )
    except Exception:
        league_settings = LeagueSettings()

    predictions = _load_player_frame()
    season_history = _load_season_history()
    artifacts = build_draft_decision_artifacts(
        predictions,
        league_settings=league_settings,
        context=DraftContext(current_pick_number=league_settings.draft_position),
        season_history=season_history,
    )

    output_dir = get_draft_strategy_dir(current_year)
    saved = save_draft_decision_artifacts(artifacts, output_dir, year=current_year)

    compat_path = Path(get_draft_board_path(current_year)).with_suffix('.json')
    compat_payload = _compatibility_payload(artifacts)
    compat_path.write_text(json.dumps(compat_payload, default=str, indent=2), encoding='utf-8')

    legacy_path = Path(get_bayesian_strategy_path(current_year, league_settings.draft_position))
    legacy_path.write_text(json.dumps(compat_payload, default=str, indent=2), encoding='utf-8')

    payload_path = Path(get_dashboard_payload_path(current_year))
    if not payload_path.exists():
        payload_path.write_text(json.dumps(artifacts.dashboard_payload, default=str, indent=2), encoding='utf-8')

    backtest_path = Path(get_draft_decision_backtest_path(current_year))
    if artifacts.backtest and not backtest_path.exists():
        backtest_path.write_text(json.dumps(artifacts.backtest, default=str, indent=2), encoding='utf-8')

    logger.info('Draft decision artifacts created:')
    logger.info('  workbook: %s', saved['workbook_path'])
    logger.info('  dashboard payload: %s', saved['payload_path'])
    logger.info('  dashboard html: %s', saved['html_path'])
    logger.info('  compatibility json: %s', compat_path)
    logger.info('  legacy strategy json: %s', legacy_path)
    if artifacts.backtest:
        logger.info('  backtest: %s', saved['backtest_path'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
