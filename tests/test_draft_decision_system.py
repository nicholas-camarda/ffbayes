from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ffbayes.draft_strategy.draft_decision_system import (
    DraftContext,
    LeagueSettings,
    _starter_points_from_roster,
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
            'FantPt': [300, 275, 230, 215, 225, 210, 175],
            'FantPtPPR': [320, 295, 250, 235, 245, 230, 195],
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
    players.loc[len(players) - 1, 'FantPt'] = 989
    players.loc[len(players) - 1, 'FantPtPPR'] = 1009

    table = build_decision_table(
        players, LeagueSettings(), DraftContext(current_pick_number=10)
    )

    assert len(table) == len(_synthetic_players()['player_name'].unique())
    assert table[table['player_name'] == 'Alpha QB'].iloc[0]['proj_points_mean'] == 999


def test_decision_table_filters_non_fantasy_positions_and_blank_names():
    players = pd.DataFrame(
        {
            'player_name': ['Alpha QB', '  ', 'Mystery DL', 'Unknown Guy', 'Beta RB'],
            'position': ['QB', 'RB', 'DL', 'UNKNOWN', 'RB'],
            'proj_points_mean': [300, 200, 150, 180, 220],
            'adp': [3, 40, 120, 80, 12],
        }
    )

    table = build_decision_table(
        players, LeagueSettings(), DraftContext(current_pick_number=10)
    )

    assert table['player_name'].tolist() == ['Alpha QB', 'Beta RB']
    assert set(table['position']).issubset({'QB', 'RB'})


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


def test_recommendations_use_starter_first_policy_and_wait_gate():
    players = pd.DataFrame(
        {
            'player_name': ['Starter RB', 'Backup QB', 'Safe WR', 'Early DST'],
            'position': ['RB', 'QB', 'WR', 'DST'],
            'proj_points_mean': [225, 240, 185, 170],
            'posterior_prob_beats_replacement': [0.82, 0.68, 0.60, 0.45],
            'posterior_floor': [190, 210, 160, 150],
            'posterior_ceiling': [255, 262, 210, 182],
            'posterior_std': [14, 12, 10, 8],
            'adp': [9, 14, 40, 30],
            'adp_std': [2.0, 2.2, 4.5, 3.0],
        }
    )
    settings = LeagueSettings()
    table = build_decision_table(players, settings, DraftContext(current_pick_number=10))
    recommendations = build_recommendations(
        table,
        settings,
        DraftContext(current_pick_number=10, roster_counts={'QB': 1}),
    )

    pick_now = recommendations[recommendations['recommendation_lane'] == 'pick_now'].iloc[0]
    can_wait = recommendations[recommendations['recommendation_lane'] == 'can_wait']

    assert pick_now['position'] == 'RB'
    assert 'Early DST' not in recommendations[recommendations['recommendation_lane'] != 'can_wait']['player_name'].tolist()
    assert set(can_wait['wait_signal']).issubset({'safe_to_wait', 'late_round_stash_ok'})


def test_starter_points_include_dst_and_k_slots():
    roster = pd.DataFrame(
        {
            'player_name': ['Alpha QB', 'Alpha RB', 'Beta RB', 'Alpha WR', 'Beta WR', 'Alpha TE', 'Alpha DST', 'Alpha K'],
            'position': ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'DST', 'K'],
            'actual_points': [20, 15, 12, 14, 11, 9, 8, 7],
        }
    )

    total = _starter_points_from_roster(roster, LeagueSettings())

    assert total == 96.0


def test_recommendations_fill_specialists_in_late_rounds():
    settings = LeagueSettings()
    late_pick = settings.league_size * (settings.round_count() - 1)
    players = pd.DataFrame(
        {
            'player_name': ['Alpha DST', 'Alpha K', 'Bench WR'],
            'position': ['DST', 'K', 'WR'],
            'proj_points_mean': [10, 9, 8],
            'posterior_prob_beats_replacement': [0.55, 0.52, 0.40],
            'posterior_floor': [8, 7, 6],
            'posterior_ceiling': [12, 11, 9],
            'posterior_std': [1.0, 1.0, 1.0],
            'adp': [140, 145, 110],
            'adp_std': [4.0, 4.0, 6.0],
        }
    )

    table = build_decision_table(
        players,
        settings,
        DraftContext(current_pick_number=late_pick),
    )
    recommendations = build_recommendations(
        table,
        settings,
        DraftContext(current_pick_number=late_pick),
    )

    assert recommendations.iloc[0]['position'] in {'DST', 'K'}


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


def test_decision_table_respects_scoring_preset_projection_inputs():
    players = pd.DataFrame(
        {
            'player_name': ['Alpha RB', 'Beta WR'],
            'position': ['RB', 'WR'],
            'FantPt': [180.0, 160.0],
            'FantPtPPR': [220.0, 210.0],
            'adp': [10, 12],
        }
    )

    standard_table = build_decision_table(
        players,
        LeagueSettings(scoring_type='Standard', ppr_value=0.0),
        DraftContext(current_pick_number=10),
    )
    half_table = build_decision_table(
        players,
        LeagueSettings(scoring_type='Half-PPR', ppr_value=0.5),
        DraftContext(current_pick_number=10),
    )
    ppr_table = build_decision_table(
        players,
        LeagueSettings(scoring_type='PPR', ppr_value=1.0),
        DraftContext(current_pick_number=10),
    )

    alpha_standard = standard_table.loc[
        standard_table['player_name'] == 'Alpha RB', 'proj_points_mean'
    ].iloc[0]
    alpha_half = half_table.loc[
        half_table['player_name'] == 'Alpha RB', 'proj_points_mean'
    ].iloc[0]
    alpha_ppr = ppr_table.loc[
        ppr_table['player_name'] == 'Alpha RB', 'proj_points_mean'
    ].iloc[0]

    assert alpha_standard == 180.0
    assert alpha_half == 200.0
    assert alpha_ppr == 220.0


def test_decision_table_uses_posterior_projection_inputs_when_present():
    players = pd.DataFrame(
        {
            'player_name': ['Alpha RB', 'Beta WR'],
            'position': ['RB', 'WR'],
            'posterior_mean': [212.0, 198.0],
            'posterior_floor': [180.0, 170.0],
            'posterior_ceiling': [245.0, 228.0],
            'posterior_std': [18.0, 16.0],
            'posterior_prob_beats_replacement': [0.82, 0.74],
            'adp': [11, 17],
        }
    )

    table = build_decision_table(
        players, LeagueSettings(), DraftContext(current_pick_number=10)
    )

    alpha = table.set_index('player_name').loc['Alpha RB']
    assert alpha['proj_points_mean'] == 212.0
    assert alpha['proj_points_floor'] == 180.0
    assert alpha['proj_points_ceiling'] == 245.0
    assert alpha['posterior_prob_beats_replacement'] == 0.82


def test_dashboard_payload_includes_preset_bundle_and_model_notes():
    settings = LeagueSettings(scoring_type='Half-PPR', ppr_value=0.5)
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
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(),
        settings,
        DraftContext(current_pick_number=10),
        season_history=season_history,
    )
    payload = artifacts.dashboard_payload

    assert payload['runtime_controls']['active_scoring_preset'] == 'half_ppr'
    assert payload['scoring_presets']['standard']['available'] is True
    assert payload['scoring_presets']['half_ppr']['available'] is True
    assert payload['scoring_presets']['ppr']['available'] is True
    assert 'draft_score' in payload['metric_glossary']
    assert 'headline' in payload['model_overview']
    assert 'top_disagreements' in payload['bayesian_vor_summary']
    assert payload['selected_player']


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
        assert 'repo_dashboard_index' in saved
        assert saved['repo_dashboard_index'].exists()
        if 'runtime_dashboard_index' in saved:
            assert saved['runtime_dashboard_index'].exists()


def test_exported_dashboard_html_contains_live_controls_and_full_board_renderer():
    settings = LeagueSettings()
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'draft_board.html'
        from ffbayes.draft_strategy.draft_decision_system import export_dashboard_html

        export_dashboard_html(
            artifacts.decision_table,
            artifacts.recommendations,
            output_path,
            settings,
            backtest=artifacts.backtest,
            source_freshness=artifacts.source_freshness,
            dashboard_payload=artifacts.dashboard_payload,
        )

        html = output_path.read_text(encoding='utf-8')
        assert 'Full Player Board' in html
        assert 'Draft Controls' in html
        assert 'Bayesian vs simple VOR' in html
        assert 'ffbayes-dashboard-state-v2' in html
        assert 'advance-button' not in html
        assert 'data-status-filter' not in html
        assert 'Marking Taken or Mine advances the draft automatically.' in html
        assert 'slice(0, 30)' not in html


def test_exported_dashboard_html_includes_reset_confirmation_and_state_helpers():
    settings = LeagueSettings()
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'draft_board.html'
        from ffbayes.draft_strategy.draft_decision_system import export_dashboard_html

        export_dashboard_html(
            artifacts.decision_table,
            artifacts.recommendations,
            output_path,
            settings,
            backtest=artifacts.backtest,
            source_freshness=artifacts.source_freshness,
            dashboard_payload=artifacts.dashboard_payload,
        )

        html = output_path.read_text(encoding='utf-8')
        assert 'function clearDraftProgressState()' in html
        assert 'window.confirm(' in html
        assert (
            'Clear taken players, your roster, queue, and undo history while keeping your current league settings and current pick?'
            in html
        )
        assert 'function isInspectableStatus(status)' in html
        assert 'pickNow[0] || availableRows[0] || rows[0] || null' in html


def test_exported_dashboard_html_includes_flex_need_adjustment_logic():
    settings = LeagueSettings()
    artifacts = build_draft_decision_artifacts(
        _synthetic_players(), settings, DraftContext(current_pick_number=10)
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'draft_board.html'
        from ffbayes.draft_strategy.draft_decision_system import export_dashboard_html

        export_dashboard_html(
            artifacts.decision_table,
            artifacts.recommendations,
            output_path,
            settings,
            backtest=artifacts.backtest,
            source_freshness=artifacts.source_freshness,
            dashboard_payload=artifacts.dashboard_payload,
        )

        html = output_path.read_text(encoding='utf-8')
        assert 'const flexEligibleExtras = [\'RB\', \'WR\', \'TE\']' in html
        assert 'function offensiveNeedByPosition(need, position)' in html
        assert 'acc[position] = offensiveNeedByPosition(need, position);' in html


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
    assert backtest['evaluation_scope']['draft_score_label'] == 'posterior_contextual_policy'
    assert backtest['evaluation_scope']['primary_objective'] == 'starter_lineup_points'
    assert backtest['evaluation_scope']['wait_policy'] == 'conservative'
    assert 'overall' in backtest
    assert 'by_strategy' in backtest['overall']
    assert 'draft_score_vs_historical_vor_proxy' in backtest['overall']
    assert 'draft_score_vs_consensus' in backtest['overall']
    assert 'policy_diagnostics' in backtest['overall']
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


def test_backtest_filters_non_fantasy_positions_from_rosters_and_reports_scope():
    season_history = pd.DataFrame(
        {
            'Season': [2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023, 2024, 2024, 2024],
            'Name': [
                'Alpha QB',
                'Alpha RB',
                'Mystery DL',
                'Alpha QB',
                'Alpha RB',
                '',
                'Alpha QB',
                'Alpha RB',
                'Unknown Guy',
                'Alpha QB',
                'Alpha RB',
                'Fullback',
            ],
            'Position': ['QB', 'RB', 'DL', 'QB', 'RB', 'WR', 'QB', 'RB', 'FB', 'QB', 'RB', 'FB'],
            'FantPt': [250, 180, 90, 260, 190, 40, 270, 200, 35, 280, 210, 20],
        }
    )

    backtest = run_draft_backtest(season_history, LeagueSettings())

    assert backtest['evaluation_scope']['eligible_positions'] == ['QB', 'RB', 'WR', 'TE', 'DST', 'K']
    assert backtest['evaluation_scope']['eligible_player_universe']['removed_non_fantasy_position_count'] >= 2
    for season in backtest['by_season']:
        drafted = season['by_strategy']['draft_score']['drafted_players']
        counts = season['by_strategy']['draft_score']['position_counts']
        assert '' not in drafted
        assert set(counts).issubset({'QB', 'RB', 'WR', 'TE', 'DST', 'K'})
