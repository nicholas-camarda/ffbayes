import json


def test_stage_pages_site_copies_canonical_dashboard_files(tmp_path):
    from ffbayes.publish_pages import stage_pages_site

    source_html = tmp_path / 'draft_board_2026.html'
    source_payload = tmp_path / 'dashboard_payload_2026.json'
    source_html.write_text(
        """
<html><body><script>
window.FFBAYES_DASHBOARD = {"dashboard": true};

    (() => { console.log('dashboard'); })();
</script></body></html>
""".strip(),
        encoding='utf-8',
    )
    source_payload.write_text(
        '{"dashboard": true, "generated_at": "2026-04-04T18:35:49", "analysis_provenance": {"overall_freshness": {"status": "fresh", "override_used": false, "warnings": []}}}',
        encoding='utf-8',
    )

    output_dir = tmp_path / 'site'
    result = stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=source_payload,
        output_dir=output_dir,
    )

    assert result['site_dir'] == output_dir
    assert result['index_path'] == output_dir / 'index.html'
    assert result['payload_path'] == output_dir / 'dashboard_payload.json'
    assert result['provenance_path'] == output_dir / 'publish_provenance.json'
    assert result['nojekyll_path'] == output_dir / '.nojekyll'
    payload = json.loads((output_dir / 'dashboard_payload.json').read_text(encoding='utf-8'))
    provenance = json.loads((output_dir / 'publish_provenance.json').read_text(encoding='utf-8'))
    assert payload['publish_provenance']['schema_version'] == 'publish_provenance_v1'
    assert provenance['schema_version'] == 'publish_provenance_v1'
    assert 'publish_provenance' in (output_dir / 'index.html').read_text(encoding='utf-8')
    assert not (output_dir / 'draft_board_2026.html').exists()


def test_stage_pages_site_removes_stale_payload_when_missing(tmp_path):
    from ffbayes.publish_pages import stage_pages_site

    source_html = tmp_path / 'draft_board_2026.html'
    source_html.write_text('<html><body>dashboard</body></html>', encoding='utf-8')
    output_dir = tmp_path / 'site'
    stale_payload = output_dir / 'dashboard_payload.json'
    stale_provenance = output_dir / 'publish_provenance.json'
    stale_payload.parent.mkdir(parents=True, exist_ok=True)
    stale_payload.write_text('stale', encoding='utf-8')
    stale_provenance.write_text('stale', encoding='utf-8')

    stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=tmp_path / 'missing_payload.json',
        output_dir=output_dir,
    )

    assert not stale_payload.exists()
    assert not stale_provenance.exists()


def test_stage_pages_site_surfaces_stale_paths_without_timestamp_false_positive(tmp_path):
    from ffbayes.publish_pages import stage_pages_site

    source_html = tmp_path / 'draft_board_2026.html'
    source_payload = tmp_path / 'dashboard_payload_2026.json'
    source_html.write_text(
        """
<html><body><script>
window.FFBAYES_DASHBOARD = {"dashboard": true};

    (() => { console.log('dashboard'); })();
</script></body></html>
""".strip(),
        encoding='utf-8',
    )
    source_payload.write_text(
        '{"dashboard": true, "generated_at": "2026-04-04T18:35:49", "analysis_provenance": {"overall_freshness": {"status": "fresh", "override_used": false, "warnings": []}}}',
        encoding='utf-8',
    )

    output_dir = tmp_path / 'site'
    first = stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=source_payload,
        output_dir=output_dir,
    )
    assert first['stale_paths'] == []

    second = stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=source_payload,
        output_dir=output_dir,
    )
    assert second['stale_paths'] == []

    source_html.write_text(
        """
<html><body><script>
window.FFBAYES_DASHBOARD = {"dashboard": "updated"};

    (() => { console.log('dashboard template updated'); })();
</script></body></html>
""".strip(),
        encoding='utf-8',
    )
    third = stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=source_payload,
        output_dir=output_dir,
    )
    assert output_dir / 'index.html' in third['stale_paths']
