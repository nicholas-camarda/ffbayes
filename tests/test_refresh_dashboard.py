import json

import pytest


def _fresh_decision_evidence():
    return {
        'decision_evidence': {
            'available': True,
            'status': 'available',
            'headline': 'Decision evidence is available for this board.',
            'freshness': {'status': 'fresh', 'override_used': False},
            'strategy_summary': [
                {
                    'strategy': 'draft_score',
                    'mean_lineup_points': 100.0,
                    'season_count': 1,
                }
            ],
            'season_rows': [
                {
                    'holdout_year': 2025,
                    'draft_score_lineup_points': 100.0,
                    'historical_vor_proxy_lineup_points': 90.0,
                    'delta_lineup_points': 10.0,
                }
            ],
            'top_disagreements': [],
        }
    }


def _scoring_preset_bundle_fixture():
    def _entry(key: str, label: str, scoring_type: str, ppr_value: float) -> dict:
        return {
            'key': key,
            'label': label,
            'available': True,
            'scoring_type': scoring_type,
            'ppr_value': ppr_value,
            'league_settings': {
                'league_size': 10,
                'draft_position': 10,
                'scoring_type': scoring_type,
                'ppr_value': ppr_value,
                'risk_tolerance': 'medium',
                'roster_spots': {
                    'QB': 1,
                    'RB': 2,
                    'WR': 2,
                    'TE': 1,
                    'FLEX': 1,
                    'DST': 1,
                    'K': 1,
                },
                'flex_weights': {'RB': 0.45, 'WR': 0.45, 'TE': 0.1},
                'bench_slots': 6,
            },
            'decision_table': [{'player_name': 'Test Player', 'position': 'RB'}],
            'supporting_math': {
                'draft_score_mean': 1.0,
                'draft_score_std': 0.0,
                'availability_mean': 0.5,
                'top_draft_score': 1.0,
                'top_simple_vor_proxy': 1.0,
            },
        }

    return {
        'standard': _entry('standard', 'Standard (0.0 PPR)', 'Standard', 0.0),
        'half_ppr': _entry('half_ppr', 'Half PPR (0.5)', 'Half-PPR', 0.5),
        'ppr': _entry('ppr', 'Full PPR (1.0)', 'PPR', 1.0),
    }


def test_refresh_runtime_dashboard_rebuilds_html_and_stages_pages(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard
    from ffbayes.publish_pages import stage_pages_site as real_stage_pages_site

    refresh_runtime_dashboard = refresh_dashboard.refresh_runtime_dashboard

    project_root = tmp_path / 'project'
    runtime_root = tmp_path / 'runtime'
    project_root.mkdir()
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setattr(
        refresh_dashboard,
        'stage_pages_site',
        lambda **kwargs: real_stage_pages_site(
            **kwargs, output_dir=project_root / 'site'
        ),
    )

    payload_path = (
        runtime_root
        / 'seasons'
        / '2026'
        / 'draft_strategy'
        / 'dashboard_payload_2026.json'
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {
                    'league_size': 10,
                    'draft_position': 10,
                    'scoring_type': 'PPR',
                    'ppr_value': 0.5,
                    'risk_tolerance': 'medium',
                    'roster_spots': {
                        'QB': 1,
                        'RB': 2,
                        'WR': 2,
                        'TE': 1,
                        'FLEX': 1,
                        'DST': 1,
                        'K': 1,
                    },
                    'flex_weights': {'RB': 0.45, 'WR': 0.45, 'TE': 0.1},
                    'bench_slots': 6,
                },
                'decision_table': [
                    {
                        'player_name': 'Test Player',
                        'position': 'RB',
                        'draft_rank': 1,
                        'simple_vor_rank': 1,
                        'draft_score': 1.23,
                        'simple_vor_proxy': 12.0,
                        'availability_to_next_pick': 0.5,
                        'expected_regret': 0.1,
                        'upside_score': 0.7,
                        'fragility_score': 0.2,
                        'status': 'available',
                    }
                ],
                'recommendation_summary': [{'player_name': 'Test Player'}],
                'source_freshness': [
                    {
                        'source_name': 'players',
                        'status': 'fresh',
                        'override_used': False,
                        'latest_expected_year': 2026,
                        'latest_found_year': 2026,
                    }
                ],
                'analysis_provenance': {
                    'overall_freshness': {
                        'status': 'fresh',
                        'override_used': False,
                        'warnings': [],
                    }
                },
                'decision_evidence': {
                    'available': True,
                    'status': 'available',
                    'headline': 'Decision evidence is available for this board.',
                    'freshness': {'status': 'fresh', 'override_used': False},
                    'strategy_summary': [
                        {
                            'strategy': 'draft_score',
                            'mean_lineup_points': 100.0,
                            'season_count': 1,
                        }
                    ],
                    'season_rows': [
                        {
                            'holdout_year': 2025,
                            'draft_score_lineup_points': 100.0,
                            'historical_vor_proxy_lineup_points': 90.0,
                            'delta_lineup_points': 10.0,
                        }
                    ],
                    'top_disagreements': [],
                },
                'war_room_visuals': {
                    'schema_version': 'war_room_visuals_v1',
                    'contextual': {'key': 'contextual_score', 'label': 'Board value score'},
                    'baseline': {'key': 'baseline_score', 'label': 'Simple VOR proxy'},
                    'timing_frontier': {
                        'available': True,
                        'status': 'available',
                        'question': 'Can I safely wait on this value, or do I need to pick now?',
                        'reason': '',
                        'candidates': [
                            {
                                'player_name': 'Test Player',
                                'position': 'RB',
                                'lane': 'pick_now',
                                'timing_survival': 0.5,
                                'wait_regret': 0.1,
                            }
                        ],
                    },
                    'positional_cliffs': {
                        'available': True,
                        'status': 'available',
                        'question': 'Which positions are about to fall off if I wait?',
                        'reason': '',
                        'default_positions': ['RB'],
                        'positions': [],
                    },
                    'comparative_explainer': {
                        'available': True,
                        'status': 'available',
                        'question': 'Why does the contextual board differ from the baseline value view?',
                        'reason': '',
                        'contextual_label': 'Board value score',
                        'baseline_label': 'Simple VOR proxy',
                        'top_disagreements': [],
                    },
                },
                'model_overview': {'headline': 'Model overview'},
                'metric_glossary': {},
                'runtime_controls': {
                    'risk_tolerance_options': ['low', 'medium', 'high'],
                    'supported_scoring_presets': ['standard', 'half_ppr', 'ppr'],
                    'active_scoring_preset': 'half_ppr',
                },
                'scoring_presets': _scoring_preset_bundle_fixture(),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    html_path = payload_path.with_name('draft_board_2026.html')
    result = refresh_runtime_dashboard(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
        stage_pages=True,
    )
    freshness = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
    )

    html_text = html_path.read_text(encoding='utf-8')
    assert 'FFBayes Draft War Room' in html_text
    assert 'Decision evidence' in html_text
    assert 'Freshness and provenance' in html_text
    assert 'Projection breakdown' in html_text
    assert 'Season total mean' in html_text
    assert 'Detailed evidence' in html_text
    assert result['html_path'] == html_path
    assert result['source_payload_path'] == payload_path
    assert freshness['status'] == 'fresh'
    assert freshness['surface_kind'] == 'canonical_runtime'

    runtime_index = runtime_root / 'dashboard' / 'index.html'
    repo_index = project_root / 'dashboard' / 'index.html'
    site_index = project_root / 'site' / 'index.html'
    site_payload = project_root / 'site' / 'dashboard_payload.json'
    site_provenance = project_root / 'site' / 'publish_provenance.json'

    assert runtime_index.exists()
    assert repo_index.exists()
    assert site_index.exists()
    assert site_payload.exists()
    assert site_provenance.exists()
    assert not (runtime_root / 'dashboard' / 'draft_board_2026.html').exists()
    assert not (runtime_root / 'dashboard' / 'dashboard_payload_2026.json').exists()
    assert not (project_root / 'dashboard' / 'draft_board_2026.html').exists()
    assert not (project_root / 'dashboard' / 'dashboard_payload_2026.json').exists()
    assert result['staged_index_path'] == site_index
    assert result['staged_payload_path'] == site_payload
    assert result['staged_provenance_path'] == site_provenance
    staged_payload = json.loads(site_payload.read_text(encoding='utf-8'))
    assert staged_payload['publish_provenance']['schema_version'] == 'publish_provenance_v1'
    assert staged_payload['publish_provenance']['surface_sync']['status'] == 'synchronized'
    assert staged_payload['war_room_visuals']['schema_version'] == 'war_room_visuals_v1'
    assert staged_payload['war_room_visuals']['timing_frontier']['available'] is True


def test_check_dashboard_freshness_reports_fresh_and_stale(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard

    runtime_root = tmp_path / 'runtime'
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = (
        runtime_root
        / 'seasons'
        / '2026'
        / 'draft_strategy'
        / 'dashboard_payload_2026.json'
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {
                    'league_size': 10,
                    'draft_position': 10,
                    'scoring_type': 'PPR',
                    'ppr_value': 0.5,
                    'risk_tolerance': 'medium',
                    'roster_spots': {
                        'QB': 1,
                        'RB': 2,
                        'WR': 2,
                        'TE': 1,
                        'FLEX': 1,
                        'DST': 1,
                        'K': 1,
                    },
                    'flex_weights': {'RB': 0.45, 'WR': 0.45, 'TE': 0.1},
                    'bench_slots': 6,
                },
                'decision_table': [{'player_name': 'Test Player', 'position': 'RB'}],
                'recommendation_summary': [{'player_name': 'Test Player'}],
                'source_freshness': [],
                'analysis_provenance': {
                    'overall_freshness': {
                        'status': 'fresh',
                        'override_used': False,
                        'warnings': [],
                    }
                },
                'decision_evidence': {
                    'available': True,
                    'status': 'available',
                    'headline': 'Decision evidence is available for this board.',
                    'freshness': {'status': 'fresh', 'override_used': False},
                    'strategy_summary': [
                        {
                            'strategy': 'draft_score',
                            'mean_lineup_points': 100.0,
                            'season_count': 1,
                        }
                    ],
                    'season_rows': [
                        {
                            'holdout_year': 2025,
                            'draft_score_lineup_points': 100.0,
                            'historical_vor_proxy_lineup_points': 90.0,
                            'delta_lineup_points': 10.0,
                        }
                    ],
                    'top_disagreements': [],
                },
                'model_overview': {'headline': 'Model overview'},
                'metric_glossary': {},
                'runtime_controls': {
                    'risk_tolerance_options': ['low', 'medium', 'high'],
                    'supported_scoring_presets': ['standard', 'half_ppr', 'ppr'],
                    'active_scoring_preset': 'half_ppr',
                },
                'scoring_presets': _scoring_preset_bundle_fixture(),
            }
        ),
        encoding='utf-8',
    )

    html_path = payload_path.with_name('draft_board_2026.html')
    refresh_dashboard.refresh_runtime_dashboard(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
        stage_pages=False,
    )

    fresh = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
    )
    assert fresh['status'] == 'fresh'
    assert fresh['stale_paths'] == []
    assert fresh['mutated'] is False

    html_path.write_text(html_path.read_text(encoding='utf-8') + '\n<!-- drift -->\n', encoding='utf-8')
    stale = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
    )
    assert stale['status'] == 'stale'
    assert stale['stale_paths'] == [str(html_path)]


def test_check_dashboard_freshness_accepts_staged_site_publish_provenance(
    tmp_path, monkeypatch
):
    import ffbayes.refresh_dashboard as refresh_dashboard
    from ffbayes.publish_pages import stage_pages_site

    project_root = tmp_path / 'project'
    runtime_root = tmp_path / 'runtime'
    project_root.mkdir()
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = (
        runtime_root
        / 'seasons'
        / '2026'
        / 'draft_strategy'
        / 'dashboard_payload_2026.json'
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {'league_size': 10, 'draft_position': 10},
                'decision_table': [{'player_name': 'Test Player', 'position': 'RB'}],
                'recommendation_summary': [{'player_name': 'Test Player'}],
                'analysis_provenance': {
                    'overall_freshness': {
                        'status': 'fresh',
                        'override_used': False,
                        'warnings': [],
                    }
                },
                **_fresh_decision_evidence(),
            }
        ),
        encoding='utf-8',
    )
    html_path = payload_path.with_name('draft_board_2026.html')
    refresh_dashboard.refresh_runtime_dashboard(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
        stage_pages=False,
    )
    stage_pages_site(
        year=2026,
        source_html=html_path,
        source_payload=payload_path,
        output_dir=project_root / 'site',
    )

    result = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=project_root / 'site' / 'index.html',
    )

    assert result['surface_kind'] == 'staged_site'
    assert result['status'] == 'fresh'
    assert result['stale_paths'] == []


def test_check_dashboard_freshness_reports_stale_repo_shortcut_payload(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard

    project_root = tmp_path / 'project'
    runtime_root = tmp_path / 'runtime'
    project_root.mkdir()
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = (
        runtime_root
        / 'seasons'
        / '2026'
        / 'draft_strategy'
        / 'dashboard_payload_2026.json'
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {'league_size': 10, 'draft_position': 10},
                'decision_table': [{'player_name': 'Test Player', 'position': 'RB'}],
                **_fresh_decision_evidence(),
            }
        ),
        encoding='utf-8',
    )
    html_path = payload_path.with_name('draft_board_2026.html')
    refresh_dashboard.refresh_runtime_dashboard(
        year=2026,
        payload_path=payload_path,
        output_html=html_path,
        stage_pages=False,
    )

    repo_payload = project_root / 'dashboard' / 'dashboard_payload.json'
    repo_payload.write_text(
        json.dumps(
            {
                'generated_at': '2020-01-01T00:00:00',
                'league_settings': {'league_size': 99, 'draft_position': 1},
                'decision_table': [],
            }
        ),
        encoding='utf-8',
    )

    result = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=project_root / 'dashboard' / 'index.html',
    )

    assert result['surface_kind'] == 'repo_shortcut'
    assert result['status'] == 'stale'
    assert result['target_payload_path'] == str(repo_payload)
    assert result['stale_paths'] == [str(repo_payload)]


def test_refresh_dashboard_rejects_degraded_decision_evidence(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard

    runtime_root = tmp_path / 'runtime'
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = runtime_root / 'dashboard_payload_2026.json'
    evidence = _fresh_decision_evidence()
    evidence['decision_evidence']['status'] = 'degraded'
    evidence['decision_evidence']['freshness']['status'] = 'degraded'
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {'league_size': 10, 'draft_position': 10},
                'decision_table': [{'player_name': 'Test Player', 'position': 'RB'}],
                **evidence,
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='fresh decision evidence'):
        refresh_dashboard.refresh_runtime_dashboard(
            year=2026,
            payload_path=payload_path,
            output_html=runtime_root / 'draft_board_2026.html',
        )


def test_check_dashboard_freshness_reports_missing_target(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard

    runtime_root = tmp_path / 'runtime'
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = runtime_root / 'dashboard_payload_2026.json'
    payload_path.write_text(
        json.dumps(
            {
                'generated_at': '2026-04-07T14:50:00',
                'league_settings': {'league_size': 10, 'draft_position': 10},
                'decision_table': [],
                **_fresh_decision_evidence(),
            }
        ),
        encoding='utf-8',
    )
    missing_target = runtime_root / 'missing_dashboard.html'

    result = refresh_dashboard.check_dashboard_freshness(
        year=2026,
        payload_path=payload_path,
        output_html=missing_target,
    )

    assert result['status'] == 'missing_target'
    assert result['stale_paths'] == [str(missing_target)]
