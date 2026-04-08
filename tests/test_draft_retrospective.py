import json
import sys

import pytest


def _write_finalized_payload(path, *, partial_receipts=False, schema_version='finalized_draft_v1'):
    pick_receipts = [
        {
            'pick_number': 1,
            'player_name': 'Alpha RB',
            'position': 'RB',
            'top_recommendation': 'Alpha RB',
            'followed_model': True,
            'top_wait_candidate': {'player_name': 'Wait RB'},
        },
        {
            'pick_number': 2,
            'player_name': 'Beta WR',
            'position': 'WR',
            'top_recommendation': 'Top WR',
            'followed_model': False,
            'top_wait_candidate': {'player_name': 'Wait WR'},
        },
    ]
    if partial_receipts:
        pick_receipts[1].pop('followed_model')

    payload = {
        'schema_version': schema_version,
        'season_year': 2026,
        'exported_at': '2026-09-01T12:00:00',
        'source_payload_generated_at': '2026-08-20T09:00:00',
        'league_settings': {
            'scoring_preset': 'half_ppr',
            'league_size': 10,
            'draft_position': 3,
            'bench_slots': 6,
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1},
        },
        'drafted_players': [
            {
                'player_name': 'Alpha RB',
                'position': 'RB',
                'lineup_slot': 'RB1',
                'proj_points_mean': 200.0,
                'fragility_score': 0.2,
                'value_indicator': 'Value',
            },
            {
                'player_name': 'Beta WR',
                'position': 'WR',
                'lineup_slot': 'WR1',
                'proj_points_mean': 180.0,
                'fragility_score': 0.7,
                'value_indicator': 'Reach',
            },
        ],
        'starters': [
            {
                'player_name': 'Alpha RB',
                'position': 'RB',
                'lineup_slot': 'RB1',
                'proj_points_mean': 200.0,
            },
            {
                'player_name': 'Beta WR',
                'position': 'WR',
                'lineup_slot': 'WR1',
                'proj_points_mean': 180.0,
            },
        ],
        'summary_metrics': {
            'starter_lineup_mean': 380.0,
            'model_follow_count': 1,
            'model_pivot_count': 1,
        },
        'pick_receipts': pick_receipts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def _write_outcomes_csv(path):
    path.write_text(
        '\n'.join(
            [
                'Season,Name,Position,fantasy_points,fantasy_points_ppr',
                '2026,Alpha RB,RB,190,220',
                '2026,Beta WR,WR,170,200',
                '2026,Wait RB,RB,180,205',
                '2026,Top WR,WR,200,230',
                '2026,Wait WR,WR,160,190',
            ]
        ),
        encoding='utf-8',
    )
    return path


def _write_finalized_bundle(tmp_path):
    json_path = _write_finalized_payload(
        tmp_path / 'ffbayes_finalized_draft_2026_test.json'
    )
    locked_html = tmp_path / 'ffbayes_finalized_draft_2026_test.html'
    summary_html = tmp_path / 'ffbayes_finalized_summary_2026_test.html'
    locked_html.write_text('<html>locked</html>', encoding='utf-8')
    summary_html.write_text('<html>summary</html>', encoding='utf-8')
    return json_path, locked_html, summary_html


def test_run_draft_retrospective_builds_json_and_html(tmp_path, monkeypatch):
    from ffbayes.analysis.draft_retrospective import run_draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    finalized_path = _write_finalized_payload(tmp_path / 'ffbayes_finalized_draft_2026_test.json')
    outcomes_path = _write_outcomes_csv(tmp_path / 'unified_dataset.csv')

    result = run_draft_retrospective(
        finalized_json=[finalized_path],
        outcomes_path=outcomes_path,
        year=2026,
    )

    report = result['report']
    season = report['season_reports'][0]
    metrics = season['outcome_metrics']

    assert result['status'] == 'available'
    assert result['json_path'].exists()
    assert result['html_path'].exists()
    assert metrics['actual_starter_points'] == pytest.approx(390.0)
    assert metrics['starter_delta'] == pytest.approx(10.0)
    assert metrics['actual_full_roster_points'] == pytest.approx(390.0)
    assert metrics['drafted_player_hit_rate'] == pytest.approx(1.0)
    assert season['audit_context']['follow_rate'] == pytest.approx(0.5)
    assert (
        season['audit_context']['mean_pivot_actual_delta_vs_recommendation']
        == pytest.approx(-30.0)
    )
    assert (
        season['audit_context']['wait_policy_calibration'][
            'mean_actual_delta_vs_top_wait_candidate'
        ]
        == pytest.approx(11.25)
    )


def test_import_finalized_artifacts_copies_bundle_into_canonical_dir(tmp_path, monkeypatch):
    from ffbayes.analysis.draft_retrospective import import_finalized_artifacts
    from ffbayes.utils.path_constants import get_finalized_drafts_dir

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    json_path, locked_html, summary_html = _write_finalized_bundle(tmp_path)

    result = import_finalized_artifacts(
        [json_path, locked_html, summary_html],
        year=2026,
    )

    canonical_dir = get_finalized_drafts_dir(2026)
    assert result['status'] == 'copied'
    assert result['season_years'] == [2026]
    assert result['imported_json_paths'] == [
        canonical_dir / 'ffbayes_finalized_draft_2026_test.json'
    ]
    assert (canonical_dir / json_path.name).exists()
    assert (canonical_dir / locked_html.name).exists()
    assert (canonical_dir / summary_html.name).exists()
    assert json_path.exists()


def test_run_draft_retrospective_autodiscovers_imported_json(tmp_path, monkeypatch):
    from ffbayes.analysis.draft_retrospective import import_finalized_artifacts
    from ffbayes.analysis.draft_retrospective import run_draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    json_path, locked_html, summary_html = _write_finalized_bundle(tmp_path)
    outcomes_path = _write_outcomes_csv(tmp_path / 'unified_dataset.csv')
    import_finalized_artifacts([json_path, locked_html, summary_html], year=2026)

    result = run_draft_retrospective(
        outcomes_path=outcomes_path,
        year=2026,
    )

    assert result['status'] == 'available'
    assert result['report']['provenance']['finalized_drafts'][0].endswith(
        'finalized_drafts/ffbayes_finalized_draft_2026_test.json'
    )


def test_run_draft_retrospective_requires_available_outcomes(tmp_path, monkeypatch):
    from ffbayes.analysis.draft_retrospective import run_draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    finalized_path = _write_finalized_payload(tmp_path / 'ffbayes_finalized_draft_2026_test.json')

    with pytest.raises(FileNotFoundError):
        run_draft_retrospective(
            finalized_json=[finalized_path],
            outcomes_path=tmp_path / 'missing_outcomes.csv',
            year=2026,
        )


def test_load_finalized_payload_rejects_unsupported_schema(tmp_path):
    from ffbayes.analysis.draft_retrospective import load_finalized_payload

    finalized_path = _write_finalized_payload(
        tmp_path / 'ffbayes_finalized_draft_2026_test.json',
        schema_version='finalized_draft_v0',
    )

    with pytest.raises(ValueError, match='Unsupported finalized draft schema'):
        load_finalized_payload(finalized_path)


def test_build_retrospective_report_degrades_audit_when_follow_metadata_missing(tmp_path):
    from ffbayes.analysis.draft_retrospective import build_retrospective_report

    finalized_path = _write_finalized_payload(
        tmp_path / 'ffbayes_finalized_draft_2026_test.json',
        partial_receipts=True,
    )
    outcomes_path = _write_outcomes_csv(tmp_path / 'unified_dataset.csv')

    report = build_retrospective_report(
        finalized_paths=[finalized_path],
        outcomes_path=outcomes_path,
        fallback_year=2026,
    )

    season = report['season_reports'][0]
    assert season['status'] == 'available'
    assert season['audit_context']['status'] == 'degraded'
    assert season['outcome_metrics']['actual_starter_points'] == pytest.approx(390.0)
    assert 'followed_model metadata' in ' '.join(season['audit_context']['warnings'])


def test_draft_retrospective_main_prints_output_paths(tmp_path, monkeypatch, capsys):
    from ffbayes.analysis import draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    finalized_path = _write_finalized_payload(tmp_path / 'ffbayes_finalized_draft_2026_test.json')
    outcomes_path = _write_outcomes_csv(tmp_path / 'unified_dataset.csv')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'ffbayes.analysis.draft_retrospective',
            '--finalized-json',
            str(finalized_path),
            '--outcomes-path',
            str(outcomes_path),
            '--year',
            '2026',
        ],
    )

    exit_code = draft_retrospective.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'Draft retrospective status: available' in captured.out
    assert 'json:' in captured.out
    assert 'html:' in captured.out


def test_draft_retrospective_main_supports_ingest_only(tmp_path, monkeypatch, capsys):
    from ffbayes.analysis import draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    json_path, locked_html, summary_html = _write_finalized_bundle(tmp_path)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'ffbayes.analysis.draft_retrospective',
            '--import-finalized',
            str(json_path),
            str(locked_html),
            str(summary_html),
            '--ingest-only',
            '--year',
            '2026',
        ],
    )

    exit_code = draft_retrospective.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'Imported 3 finalized artifact(s)' in captured.out
    assert 'Draft retrospective status:' not in captured.out


def test_draft_retrospective_main_autodiscovers_after_ingest(
    tmp_path, monkeypatch, capsys
):
    from ffbayes.analysis import draft_retrospective

    runtime_root = tmp_path / 'runtime'
    project_root = tmp_path / 'project'
    runtime_root.mkdir()
    project_root.mkdir()
    monkeypatch.setenv('FFBAYES_RUNTIME_ROOT', str(runtime_root))
    monkeypatch.setenv('FFBAYES_PROJECT_ROOT', str(project_root))

    json_path, locked_html, summary_html = _write_finalized_bundle(tmp_path)
    outcomes_path = _write_outcomes_csv(tmp_path / 'unified_dataset.csv')

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'ffbayes.analysis.draft_retrospective',
            '--import-finalized',
            str(json_path),
            str(locked_html),
            str(summary_html),
            '--ingest-only',
            '--year',
            '2026',
        ],
    )
    assert draft_retrospective.main() == 0

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'ffbayes.analysis.draft_retrospective',
            '--year',
            '2026',
            '--outcomes-path',
            str(outcomes_path),
        ],
    )

    exit_code = draft_retrospective.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'Draft retrospective status: available' in captured.out
