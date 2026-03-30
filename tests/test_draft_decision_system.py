from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    DraftContext,
    LeagueSettings,
    availability_probability,
    build_decision_table,
    build_draft_decision_artifacts,
    build_live_recommendation_snapshot,
    build_recommendations,
    export_workbook,
    run_draft_backtest,
    save_draft_decision_artifacts,
)


def _synthetic_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'player_name': [
                'Alpha QB',
                'Beta QB',
                'Alpha RB',
                'Beta RB',
                'Alpha WR',
                'Beta WR',
                'Alpha TE',
            ],
            'position': ['QB', 'QB', 'RB', 'RB', 'WR', 'WR', 'TE'],
            'proj_points_mean': [310, 285, 240, 225, 235, 220, 185],
            'adp': [3, 14, 4, 18, 6, 16, 9],
            'std_projection': [16, 18, 21, 24, 17, 16, 12],
            'uncertainty_score': [0.10, 0.22, 0.15, 0.25, 0.18, 0.16, 0.12],
            'season_count': [4, 3, 2, 1, 5, 2, 1],
            'games_missed': [0, 2, 1, 3, 0, 4, 0],
            'age': [27, 29, 25, 24, 26, 27, 31],
            'years_in_league': [5, 4, 2, 1, 3, 2, 1],
            'team_change': [0, 1, 0, 1, 0, 0, 0],
            'role_volatility': [0.10, 0.30, 0.20, 0.40, 0.15, 0.20, 0.10],
            'site_disagreement': [0.05, 0.12, 0.10, 0.18, 0.08, 0.11, 0.04],
            'adp_std': [1.5, 2.2, 2.0, 3.0, 1.8, 2.4, 1.2],
        }
    )


def test_decision_table_builds_expected_columns():
    settings = LeagueSettings()
    table = build_decision_table(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    required = {
        'player_name',
        'position',
        'proj_points_mean',
        'proj_points_floor',
        'proj_points_ceiling',
        'adp',
        'market_rank',
        'availability_at_pick',
        'replacement_delta',
        'starter_delta',
        'upside_score',
        'fragility_score',
        'draft_score',
        'draft_tier',
        'why_flags',
    }
    assert required.issubset(set(table.columns))
    assert table['availability_at_pick'].between(0, 1).all()
    assert table.iloc[0]['draft_score'] >= table.iloc[-1]['draft_score']


def test_decision_table_collapses_duplicate_player_rows():
    players = pd.concat(
        [_synthetic_players(), _synthetic_players().iloc[[0]]], ignore_index=True
    )
    players.loc[len(players) - 1, 'proj_points_mean'] = 999

    table = build_decision_table(
        players, LeagueSettings(), DraftContext(current_pick_number=10)
    )

    assert len(table) == len(_synthetic_players()['player_name'].unique())
    assert table[table['player_name'] == 'Alpha QB'].iloc[0]['proj_points_mean'] == 999


def test_availability_probability_is_monotonic():
    early = availability_probability(
        adp=10, target_pick=8, adp_std=2.0, uncertainty_score=0.1
    )
    middle = availability_probability(
        adp=10, target_pick=12, adp_std=2.0, uncertainty_score=0.1
    )
    late = availability_probability(
        adp=10, target_pick=16, adp_std=2.0, uncertainty_score=0.1
    )

    assert early > middle > late


def test_recommendations_update_after_drafted_players():
    settings = LeagueSettings()
    table = build_decision_table(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    initial = build_recommendations(
        table, settings, DraftContext(current_pick_number=10)
    )
    after_qb_drafted = build_recommendations(
        table,
        settings,
        DraftContext(current_pick_number=10, drafted_players={'Alpha QB'}),
    )

    assert not initial.empty
    assert not after_qb_drafted.empty
    assert 'Alpha QB' not in after_qb_drafted['player_name'].tolist()
    assert (
        initial.iloc[0]['player_name'] != after_qb_drafted.iloc[0]['player_name']
        or initial.iloc[0]['position'] != after_qb_drafted.iloc[0]['position']
    )


def test_live_recommendation_snapshot_recomputes_with_board_state():
    settings = LeagueSettings()
    base_context = DraftContext(current_pick_number=10)
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, base_context
    )
    table = artifacts.decision_table

    initial_recs = build_recommendations(table, settings, base_context)
    initial_snapshot = build_live_recommendation_snapshot(
        table,
        initial_recs,
        artifacts.roster_scenarios,
        settings,
        base_context,
    )

    after_top_taken_context = DraftContext(
        current_pick_number=10, drafted_players={initial_snapshot['pick_now']['player_name']}
    )
    after_top_taken_recs = build_recommendations(table, settings, after_top_taken_context)
    after_top_taken_snapshot = build_live_recommendation_snapshot(
        table,
        after_top_taken_recs,
        artifacts.roster_scenarios,
        settings,
        after_top_taken_context,
    )

    after_roster_pick_context = DraftContext(
        current_pick_number=10,
        drafted_players={initial_snapshot['pick_now']['player_name']},
        your_players={initial_snapshot['pick_now']['player_name']},
        roster_counts={initial_snapshot['pick_now']['position']: 1},
    )
    after_roster_pick_recs = build_recommendations(
        table, settings, after_roster_pick_context
    )
    after_roster_pick_snapshot = build_live_recommendation_snapshot(
        table,
        after_roster_pick_recs,
        artifacts.roster_scenarios,
        settings,
        after_roster_pick_context,
    )

    later_pick_context = DraftContext(current_pick_number=12)
    later_pick_recs = build_recommendations(table, settings, later_pick_context)
    later_pick_snapshot = build_live_recommendation_snapshot(
        table,
        later_pick_recs,
        artifacts.roster_scenarios,
        settings,
        later_pick_context,
    )

    assert initial_snapshot['pick_now']['player_name'] != after_top_taken_snapshot['pick_now']['player_name']
    assert (
        after_roster_pick_snapshot['roster_need'][
            initial_snapshot['pick_now']['position']
        ]
        == 0
    )
    assert later_pick_snapshot['can_wait'][0]['availability_to_next_pick'] != initial_snapshot['can_wait'][0]['availability_to_next_pick']


def test_workbook_contains_core_sheets():
    settings = LeagueSettings()
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'draft_board.xlsx'
        export_workbook(
            artifacts.decision_table,
            artifacts.recommendations,
            artifacts.tier_cliffs,
            artifacts.roster_scenarios,
            artifacts.source_freshness,
            output_path,
            settings,
            backtest=artifacts.backtest,
        )

        assert output_path.exists()
        import openpyxl

        wb = openpyxl.load_workbook(output_path)
        assert {
            'Big Board',
            'By Position',
            'Live Context',
            'Pick Now',
            'Fallback Ladder',
            'Can Wait',
            'My Picks',
            'Tier Cliffs',
            'Availability',
            'Targets By Round',
            'Roster Construction Scenarios',
            'Player Notes',
            'Model Diagnostics',
            'Source Freshness',
        }.issubset(set(wb.sheetnames))


def test_slot_specific_artifacts_use_prefixed_filenames():
    settings = LeagueSettings()
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / 'artifacts'
        saved = save_draft_decision_artifacts(
            artifacts,
            output_dir,
            year=2026,
            filename_prefix='pos03_',
        )

        assert saved['workbook_path'].name == 'draft_board_pos03_2026.xlsx'
        assert saved['payload_path'].name == 'dashboard_payload_pos03_2026.json'
        assert saved['html_path'].name == 'draft_board_pos03_2026.html'
        assert saved['compat_path'].name == 'draft_board_pos03_2026.json'
        assert saved['comparison_path'].parent.name == 'model_outputs'
        assert saved['backtest_path'].name.startswith('draft_decision_backtest_pos03_')


def test_backtest_payload_has_expected_shape():
    season_history = pd.DataFrame(
        {
            'Season': [2022, 2022, 2023, 2023, 2024, 2024, 2024, 2024],
            'Name': [
                'Alpha QB',
                'Alpha RB',
                'Alpha QB',
                'Alpha RB',
                'Alpha QB',
                'Alpha RB',
                'Alpha WR',
                'Alpha TE',
            ],
            'Position': ['QB', 'RB', 'QB', 'RB', 'QB', 'RB', 'WR', 'TE'],
            'FantPt': [280, 210, 295, 220, 310, 230, 190, 170],
        }
    )

    backtest = run_draft_backtest(season_history, LeagueSettings())
    assert backtest['model_type'] == 'draft_decision_backtest'
    assert 'overall' in backtest
    assert 'by_strategy' in backtest['overall']
    assert len(backtest['overall']['by_strategy']) >= 1


def test_backtest_labels_historical_vor_proxy_explicitly():
    season_history = pd.DataFrame(
        {
            'Season': [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            'Name': [
                'Alpha QB',
                'Alpha RB',
                'Alpha QB',
                'Alpha RB',
                'Alpha QB',
                'Alpha RB',
                'Alpha QB',
                'Alpha RB',
            ],
            'Position': ['QB', 'RB', 'QB', 'RB', 'QB', 'RB', 'QB', 'RB'],
            'FantPt': [250, 180, 260, 190, 270, 200, 280, 210],
        }
    )

    backtest = run_draft_backtest(season_history, LeagueSettings())
    strategies = [row['strategy'] for row in backtest['overall']['by_strategy']]

    assert 'historical_vor_proxy' in strategies
    assert 'vor' not in strategies
