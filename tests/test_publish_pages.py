def test_stage_pages_site_copies_canonical_dashboard_files(tmp_path):
    from ffbayes.publish_pages import stage_pages_site

    source_html = tmp_path / 'draft_board_2026.html'
    source_payload = tmp_path / 'dashboard_payload_2026.json'
    source_html.write_text('<html><body>dashboard</body></html>', encoding='utf-8')
    source_payload.write_text('{"dashboard": true}', encoding='utf-8')

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
    assert result['nojekyll_path'] == output_dir / '.nojekyll'
    assert (output_dir / 'index.html').read_text(encoding='utf-8') == source_html.read_text(
        encoding='utf-8'
    )
    assert (output_dir / 'dashboard_payload.json').read_text(
        encoding='utf-8'
    ) == source_payload.read_text(encoding='utf-8')
    assert not (output_dir / 'draft_board_2026.html').exists()


def test_stage_pages_site_removes_stale_payload_when_missing(tmp_path):
    from ffbayes.publish_pages import stage_pages_site

    source_html = tmp_path / 'draft_board_2026.html'
    source_html.write_text('<html><body>dashboard</body></html>', encoding='utf-8')
    output_dir = tmp_path / 'site'
    stale_payload = output_dir / 'dashboard_payload.json'
    stale_payload.parent.mkdir(parents=True, exist_ok=True)
    stale_payload.write_text('stale', encoding='utf-8')

    stage_pages_site(
        year=2026,
        source_html=source_html,
        source_payload=tmp_path / 'missing_payload.json',
        output_dir=output_dir,
    )

    assert not stale_payload.exists()
