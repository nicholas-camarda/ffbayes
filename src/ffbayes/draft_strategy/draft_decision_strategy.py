#!/usr/bin/env python3
"""
Draft decision strategy entrypoint.

This module is the thin compatibility layer that turns the new draft decision
engine into the public CLI/script interface used by the repo.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    DraftContext,
    DraftDecisionArtifacts,
    LeagueSettings,
    _pick_first_row,
    build_draft_decision_artifacts,
    save_draft_decision_artifacts,
)
from ffbayes.utils.path_constants import (
    SNAKE_DRAFT_DATASETS_DIR,
    get_dashboard_payload_path,
    get_draft_board_path,
    get_draft_decision_backtest_path,
    get_draft_strategy_dir,
    get_pre_draft_dashboard_dir,
    get_unified_dataset_csv_path,
    get_unified_dataset_path,
)
from ffbayes.utils.strategy_path_generator import get_bayesian_strategy_path
from ffbayes.utils.vor_filename_generator import get_vor_csv_filename

logger = logging.getLogger(__name__)


@dataclass
class DraftConfig:
    """Compatibility configuration object for older callers."""

    league_size: int = 10
    draft_position: int = 10
    scoring_type: str = 'PPR'
    roster_positions: dict[str, int] = field(
        default_factory=lambda: {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1,
            'DST': 1,
            'K': 1,
        }
    )
    risk_tolerance: str = 'medium'
    all_slots: bool = False

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
            tiers[tier_name] = group.sort_values('draft_score', ascending=False)[
                'player_name'
            ].tolist()
        return tiers

    def generate_pick_options(
        self, draft_position: int, league_size: int, config: DraftConfig
    ) -> dict[str, Any]:
        settings = config.to_league_settings()
        context = DraftContext(
            current_pick_number=draft_position, drafted_players=set()
        )
        artifacts = build_draft_decision_artifacts(self.predictions, settings, context)
        top = artifacts.recommendations.head(7)
        return {
            'primary_targets': top.head(3)['player_name'].tolist(),
            'backup_options': top.iloc[3:7]['player_name'].tolist(),
            'position_priority': _position_priority(top),
            'reasoning': top.iloc[0]['rationale']
            if not top.empty
            else 'No recommendations available',
            'uncertainty_analysis': {
                'risk_tolerance': settings.risk_tolerance,
                'primary_avg_uncertainty': float(top.head(3)['fragility_score'].mean())
                if not top.empty
                else 0.0,
                'backup_avg_uncertainty': float(top.iloc[3:7]['fragility_score'].mean())
                if len(top) > 3
                else 0.0,
                'overall_uncertainty': float(top['fragility_score'].mean())
                if not top.empty
                else 0.0,
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

    def __init__(
        self,
        predictions: pd.DataFrame | None = None,
        season_history: pd.DataFrame | None = None,
    ):
        self.predictions = predictions
        self.season_history = season_history

    def build(
        self, config: DraftConfig | None = None, current_pick_number: int | None = None
    ) -> DraftDecisionArtifacts:
        config = config or DraftConfig()
        settings = config.to_league_settings()
        context = DraftContext(
            current_pick_number=current_pick_number or settings.draft_position
        )
        if self.predictions is None:
            self.predictions = _load_player_frame()
        return build_draft_decision_artifacts(
            self.predictions, settings, context, season_history=self.season_history
        )

    def save(
        self,
        output_dir: Path | str | None = None,
        year: int | None = None,
        config: DraftConfig | None = None,
    ) -> dict[str, Path]:
        artifacts = self.build(config=config)
        output_dir = (
            Path(output_dir)
            if output_dir is not None
            else get_draft_strategy_dir(year or datetime.now().year)
        )
        return save_draft_decision_artifacts(artifacts, output_dir, year=year)


class TeamConstructionOptimizer:
    """Compatibility optimizer that returns the top roster scenarios."""

    def __init__(self, predictions: pd.DataFrame | None = None):
        self.predictions = predictions

    def optimize_team_construction(
        self, config: DraftConfig | None = None
    ) -> dict[str, Any]:
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
    current_year = datetime.now().year
    vor_path = SNAKE_DRAFT_DATASETS_DIR / get_vor_csv_filename(current_year)
    if vor_path.exists():
        return _build_current_snapshot_from_vor(vor_path)

    csv_path = get_unified_dataset_csv_path()
    json_path = get_unified_dataset_path()
    if csv_path.exists():
        return _collapse_latest_player_snapshot(pd.read_csv(csv_path))
    if json_path.exists():
        return _collapse_latest_player_snapshot(pd.read_json(json_path))
    raise FileNotFoundError(
        f'No unified player dataset found at {csv_path} or {json_path}. '
        'Run the data pipeline first.'
    )


def _build_current_snapshot_from_vor(vor_path: Path) -> pd.DataFrame:
    """Build the canonical one-row-per-player snapshot from the VOR CSV."""
    vor_df = pd.read_csv(vor_path)
    rename_map = {
        'PLAYER': 'player_name',
        'POS': 'position',
        'AVG': 'adp',
        'FPTS': 'proj_points_mean',
        'VOR': 'vor_value',
        'VALUERANK': 'market_rank',
    }
    frame = vor_df.rename(columns=rename_map).copy()
    if 'player_name' not in frame.columns or 'position' not in frame.columns:
        raise ValueError(f'VOR file {vor_path} is missing player or position columns')

    frame['player_name'] = frame['player_name'].astype(str)
    frame['position'] = frame['position'].astype(str)
    frame['source_name'] = vor_path.name
    frame['source_updated_at'] = pd.Timestamp(vor_path.stat().st_mtime, unit='s')
    frame['source_updated_at'] = pd.to_datetime(
        frame['source_updated_at'], errors='coerce'
    )
    frame['site_disagreement'] = (
        1.0
        - pd.to_numeric(frame.get('vor_match_confidence'), errors='coerce').fillna(0.5)
        if 'vor_match_confidence' in frame.columns
        else np.nan
    )
    frame['season_count'] = np.nan
    frame['games_missed'] = np.nan
    frame['team_change'] = np.nan
    frame['role_volatility'] = np.nan
    frame['adp_std'] = np.nan

    history_features = _load_history_features()
    if history_features is not None and not history_features.empty:
        frame = frame.merge(
            history_features,
            on=['player_name', 'position'],
            how='left',
            suffixes=('', '_history'),
        )
        for column in [
            'team',
            'season_count',
            'games_missed',
            'team_change',
            'role_volatility',
            'site_disagreement',
            'adp_std',
            'source_name',
            'source_updated_at',
        ]:
            history_column = f'{column}_history'
            if history_column in frame.columns:
                if column in frame.columns:
                    frame[column] = frame[column].combine_first(frame[history_column])
                else:
                    frame[column] = frame[history_column]
                frame = frame.drop(columns=[history_column])
    return _collapse_latest_player_snapshot(frame)


def _load_history_features() -> pd.DataFrame | None:
    """Load player-level history features from the unified dataset."""
    csv_path = get_unified_dataset_csv_path()
    json_path = get_unified_dataset_path()
    if csv_path.exists():
        history = pd.read_csv(csv_path)
    elif json_path.exists():
        history = pd.read_json(json_path)
    else:
        return None
    return _build_history_features(history)


def _build_history_features(history_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse weekly rows into player-level history features."""
    if history_frame is None or history_frame.empty:
        return pd.DataFrame(columns=['player_name', 'position'])

    df = history_frame.copy()
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = column.strip().lower().replace(' ', '_')
        if normalized in {'name', 'player', 'player_name'}:
            rename_map[column] = 'player_name'
        elif normalized in {'pos', 'position'}:
            rename_map[column] = 'position'
        elif normalized in {'team', 'tm', 'recent_team'}:
            rename_map[column] = 'team'
    df = df.rename(columns=rename_map)
    if 'player_name' not in df.columns or 'position' not in df.columns:
        return pd.DataFrame(columns=['player_name', 'position'])

    for column in ['Season', 'G#']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    rows: list[dict[str, Any]] = []
    for (player_name, position), group in df.groupby(
        ['player_name', 'position'], dropna=False
    ):
        team_series = (
            group['team'] if 'team' in group.columns else pd.Series(dtype=object)
        )
        games_missed = 0.0
        if 'GameInjuryStatus' in group.columns:
            inj = group['GameInjuryStatus'].fillna('').astype(str).str.lower()
            games_missed = float(
                inj.str.contains('out|inactive|ir|doubt|question', regex=True).sum()
            )

        fantasy_points = pd.to_numeric(
            group.get('FantPtPPR', group.get('FantPt')), errors='coerce'
        )
        role_volatility = np.nan
        if fantasy_points is not None and fantasy_points.notna().any():
            mean_points = float(fantasy_points.mean())
            std_points = float(fantasy_points.std(ddof=0))
            role_volatility = std_points / max(1.0, abs(mean_points))

        source_updated_at = pd.NaT
        if 'Date' in group.columns and group['Date'].notna().any():
            source_updated_at = group['Date'].max()
        elif 'Season' in group.columns and group['Season'].notna().any():
            source_updated_at = pd.Timestamp(int(group['Season'].max()), 1, 1)

        rows.append(
            {
                'player_name': player_name,
                'position': position,
                'team': _pick_first_row(team_series),
                'season_count': int(group['Season'].nunique())
                if 'Season' in group.columns
                else np.nan,
                'games_missed': games_missed,
                'team_change': float(team_series.nunique() - 1)
                if 'team' in group.columns
                else np.nan,
                'role_volatility': role_volatility,
                'site_disagreement': np.nan,
                'source_name': 'historical_weekly_snapshot',
                'source_updated_at': source_updated_at,
            }
        )

    return pd.DataFrame(rows)


def _collapse_latest_player_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically collapse a frame to one row per player-position."""
    if frame is None or frame.empty:
        return frame

    df = frame.copy()
    for column in ['source_updated_at', 'Date', 'date']:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')
    for column in ['Season', 'season', 'G#', 'week', 'Week']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    sort_columns = [
        col
        for col in [
            'source_updated_at',
            'Season',
            'season',
            'Date',
            'date',
            'G#',
            'week',
            'Week',
        ]
        if col in df.columns
    ]
    if sort_columns:
        df = df.sort_values(sort_columns)

    if 'player_name' in df.columns and 'position' in df.columns:
        df = df.drop_duplicates(subset=['player_name', 'position'], keep='last')
    return df.reset_index(drop=True)


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
        recommendation_window['availability_to_next_pick'] = recommendation_window[
            'availability_at_pick'
        ]

    round_count = artifacts.league_settings.round_count()
    for round_number in range(1, round_count + 1):
        if round_number <= 5:
            window = recommendation_window.sort_values(
                ['draft_score', 'proj_points_mean'], ascending=[False, False]
            ).head(7)
        elif round_number <= 10:
            window = recommendation_window.sort_values(
                ['starter_delta', 'draft_score'], ascending=[False, False]
            ).head(7)
        else:
            window = recommendation_window.sort_values(
                ['upside_score', 'draft_score'], ascending=[False, False]
            ).head(7)
        picks[f'Pick {round_number}'] = {
            'primary_targets': window.head(3)['player_name'].tolist(),
            'backup_options': window.iloc[3:7]['player_name'].tolist(),
            'position_priority': _position_priority(window),
            'reasoning': window.iloc[0]['rationale']
            if not window.empty
            else 'No recommendations available',
            'uncertainty_analysis': {
                'risk_tolerance': artifacts.league_settings.risk_tolerance,
                'primary_avg_uncertainty': float(
                    window.head(3)['fragility_score'].mean()
                )
                if not window.empty
                else 0.0,
                'backup_avg_uncertainty': float(
                    window.iloc[3:7]['fragility_score'].mean()
                )
                if len(window) > 3
                else 0.0,
                'overall_uncertainty': float(window['fragility_score'].mean())
                if not window.empty
                else 0.0,
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
            'position_scarcity': artifacts.decision_table.groupby('position')[
                'player_name'
            ]
            .count()
            .to_dict(),
            'decision_table_columns': list(artifacts.decision_table.columns),
        },
    }


def _run_single_slot(
    predictions: pd.DataFrame,
    league_settings: LeagueSettings,
    current_year: int,
    filename_prefix: str = '',
    output_dir: Path | None = None,
    dashboard_dir: Path | None = None,
) -> dict[str, Any]:
    """Build and save artifacts for one draft slot."""
    context = DraftContext(current_pick_number=league_settings.draft_position)
    artifacts = build_draft_decision_artifacts(
        predictions,
        league_settings=league_settings,
        context=context,
        season_history=_load_season_history(),
    )

    results_dir = output_dir or get_draft_strategy_dir(current_year)
    dashboard_output_dir = dashboard_dir or get_pre_draft_dashboard_dir(current_year)
    saved = save_draft_decision_artifacts(
        artifacts,
        results_dir,
        year=current_year,
        filename_prefix=filename_prefix,
        dashboard_dir=dashboard_output_dir,
    )

    compat_path = Path(get_draft_board_path(current_year)).with_name(
        f'draft_board_{filename_prefix}{current_year}.json'
    )
    compat_payload = _compatibility_payload(artifacts)
    compat_path.write_text(
        json.dumps(compat_payload, default=str, indent=2), encoding='utf-8'
    )

    legacy_path = Path(
        get_bayesian_strategy_path(current_year, league_settings.draft_position)
    )
    if filename_prefix:
        legacy_path = legacy_path.with_name(
            f'draft_strategy_pos{league_settings.draft_position}_{current_year}.json'
        )
    legacy_path.write_text(
        json.dumps(compat_payload, default=str, indent=2), encoding='utf-8'
    )

    backtest_path = Path(get_draft_decision_backtest_path(current_year))
    if filename_prefix:
        backtest_path = backtest_path.with_name(
            f'draft_decision_backtest_{filename_prefix}{current_year}.json'
        )
    if artifacts.backtest and not backtest_path.exists():
        backtest_path.write_text(
            json.dumps(artifacts.backtest, default=str, indent=2), encoding='utf-8'
        )

    return {
        'artifacts': artifacts,
        'saved': saved,
        'compat_path': compat_path,
        'legacy_path': legacy_path,
        'backtest_path': backtest_path,
    }


def main() -> int:
    """Generate the draft board, dashboard payload, and compatibility JSON."""
    logging.basicConfig(level=logging.INFO)
    logger.info('Generating draft decision artifacts...')

    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        description='Generate the draft board and decision payload'
    )
    parser.add_argument('--draft-position', type=int, default=None)
    parser.add_argument('--league-size', type=int, default=None)
    parser.add_argument(
        '--risk-tolerance', choices=['low', 'medium', 'high'], default=None
    )
    parser.add_argument(
        '--all-slots',
        action='store_true',
        help='Generate artifacts for every draft position',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override the runtime results directory',
    )
    args = parser.parse_args()
    try:
        from ffbayes.utils.config_loader import get_config

        config_loader = get_config()
        league_settings = LeagueSettings.from_mapping(
            {
                'league_settings': {
                    'league_size': args.league_size
                    or config_loader.get_league_setting('league_size'),
                    'draft_position': args.draft_position
                    or config_loader.get_league_setting('draft_position'),
                    'scoring_type': config_loader.get_league_setting('scoring_type'),
                    'ppr_value': config_loader.get_league_setting('ppr_value')
                    if hasattr(config_loader, 'get_league_setting')
                    else 0.5,
                    'risk_tolerance': args.risk_tolerance
                    or config_loader.get_league_setting('risk_tolerance'),
                }
            }
        )
    except Exception:
        league_settings = LeagueSettings()
        if args.draft_position:
            league_settings = LeagueSettings(
                **{**league_settings.to_dict(), 'draft_position': args.draft_position}
            )
        if args.league_size:
            league_settings = LeagueSettings(
                **{**league_settings.to_dict(), 'league_size': args.league_size}
            )
        if args.risk_tolerance:
            league_settings = LeagueSettings(
                **{**league_settings.to_dict(), 'risk_tolerance': args.risk_tolerance}
            )

    predictions = _load_player_frame()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_draft_strategy_dir(current_year)
    )
    dashboard_dir = get_pre_draft_dashboard_dir(current_year)

    if args.all_slots:
        slot_summaries: list[dict[str, Any]] = []
        for slot in range(1, league_settings.league_size + 1):
            slot_settings = LeagueSettings(
                **{**league_settings.to_dict(), 'draft_position': slot}
            )
            result = _run_single_slot(
                predictions,
                slot_settings,
                current_year,
                filename_prefix=f'pos{slot:02d}_',
                output_dir=output_dir,
                dashboard_dir=dashboard_dir,
            )
            artifacts = result['artifacts']
            slot_summaries.append(
                {
                    'draft_position': slot,
                    'top_targets': artifacts.recommendations.head(5)[
                        'player_name'
                    ].tolist(),
                    'workbook_path': str(result['saved']['workbook_path']),
                    'dashboard_payload_path': str(result['saved']['payload_path']),
                    'backtest_path': str(result['backtest_path']),
                }
            )

        summary_path = output_dir / f'draft_slot_sensitivity_{current_year}.json'
        summary_path.write_text(
            json.dumps(slot_summaries, default=str, indent=2), encoding='utf-8'
        )
        logger.info('Slot sensitivity summary: %s', summary_path)
        logger.info('Generated %d slot-specific artifact sets', len(slot_summaries))
        return 0

    result = _run_single_slot(
        predictions,
        league_settings,
        current_year,
        output_dir=output_dir,
        dashboard_dir=dashboard_dir,
    )
    artifacts = result['artifacts']

    payload_path = Path(get_dashboard_payload_path(current_year))
    if not payload_path.exists():
        payload_path.write_text(
            json.dumps(artifacts.dashboard_payload, default=str, indent=2),
            encoding='utf-8',
        )

    logger.info('Draft decision artifacts created:')
    logger.info('  workbook: %s', result['saved']['workbook_path'])
    logger.info('  dashboard payload: %s', result['saved']['payload_path'])
    logger.info('  dashboard html: %s', result['saved']['html_path'])
    logger.info('  compatibility json: %s', result['compat_path'])
    logger.info('  legacy strategy json: %s', result['legacy_path'])
    if artifacts.backtest:
        logger.info('  backtest: %s', result['saved']['backtest_path'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
