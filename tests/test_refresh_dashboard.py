import json


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
        / 'runs'
        / '2026'
        / 'pre_draft'
        / 'artifacts'
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
                    'strategy_summary': [],
                    'season_rows': [],
                    'top_disagreements': [],
                },
                'model_overview': {'headline': 'Model overview'},
                'metric_glossary': {},
                'runtime_controls': {
                    'risk_tolerance_options': ['low', 'medium', 'high'],
                    'supported_scoring_presets': ['ppr'],
                    'active_scoring_preset': 'ppr',
                },
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

    html_text = html_path.read_text(encoding='utf-8')
    assert 'FFBayes Draft War Room' in html_text
    assert 'Decision evidence' in html_text
    assert 'Freshness and provenance' in html_text
    assert result['html_path'] == html_path
    assert result['source_payload_path'] == payload_path

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


def test_check_dashboard_freshness_reports_fresh_and_stale(tmp_path, monkeypatch):
    import ffbayes.refresh_dashboard as refresh_dashboard

    runtime_root = tmp_path / 'runtime'
    runtime_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))

    payload_path = (
        runtime_root
        / 'runs'
        / '2026'
        / 'pre_draft'
        / 'artifacts'
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
                    'strategy_summary': [],
                    'season_rows': [],
                    'top_disagreements': [],
                },
                'model_overview': {'headline': 'Model overview'},
                'metric_glossary': {},
                'runtime_controls': {
                    'risk_tolerance_options': ['low', 'medium', 'high'],
                    'supported_scoring_presets': ['ppr'],
                    'active_scoring_preset': 'ppr',
                },
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
        / 'runs'
        / '2026'
        / 'pre_draft'
        / 'artifacts'
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
