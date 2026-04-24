from __future__ import annotations

import sys
from pathlib import Path

import ffbayes.publish_artifacts as publish_artifacts


def test_publish_artifacts_stages_pages_before_cloud_publish(monkeypatch, capsys):
    calls: list[tuple[str, int]] = []

    def fake_stage_dashboard(year):
        calls.append(('stage_pages', year))
        return {'staged_index_path': Path('/tmp/site/index.html')}

    def fake_manage_visualizations(current_year, phase):
        calls.append((f'cloud_{phase}', current_year))
        return {
            'synced_data_files': ['data.csv'],
            'published_snapshot_files': ['snapshot.json'],
            'snapshot_dir': '/tmp/cloud/Analysis/2026-04-24',
        }

    monkeypatch.setattr(publish_artifacts, 'stage_dashboard', fake_stage_dashboard)
    monkeypatch.setattr(
        publish_artifacts, 'manage_visualizations', fake_manage_visualizations
    )
    monkeypatch.setattr(
        sys, 'argv', ['ffbayes.publish_artifacts', '--year', '2026']
    )

    assert publish_artifacts.main() == 0
    assert calls == [('stage_pages', 2026), ('cloud_pre_draft', 2026)]
    output = capsys.readouterr().out
    assert 'GitHub Pages: /tmp/site/index.html' in output
    assert 'Cloud snapshot: /tmp/cloud/Analysis/2026-04-24' in output
