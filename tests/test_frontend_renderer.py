"""Tests for the frontend template renderer and renderer selection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ffbayes.dashboard.frontend_renderer import (
    RENDERER_ENV_VAR,
    active_renderer,
    render_dashboard_html,
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestFrontendRenderer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_default_renderer_is_frontend(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RENDERER_ENV_VAR, None)
            self.assertEqual(active_renderer(), 'frontend')

    def test_legacy_renderer_when_configured(self):
        with mock.patch.dict(os.environ, {RENDERER_ENV_VAR: 'legacy'}):
            self.assertEqual(active_renderer(), 'legacy')

    def test_invalid_renderer_value_raises(self):
        with mock.patch.dict(os.environ, {RENDERER_ENV_VAR: 'bogus'}):
            with self.assertRaises(ValueError):
                active_renderer()

    def test_render_injects_payload_and_label(self):
        payload = json.loads(
            (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(
                encoding='utf-8'
            )
        )
        out = Path(self.tmp.name) / 'board.html'
        render_dashboard_html(payload, out, generated_label='2026-06-12 10:00')
        html = out.read_text(encoding='utf-8')
        self.assertNotIn('__PAYLOAD_JSON__', html)
        self.assertIn('id="ffbayes-dashboard-payload"', html)
        self.assertIn(payload['generated_at'], html)

    def test_render_escapes_script_breaking_strings(self):
        payload = json.loads(
            (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(
                encoding='utf-8'
            )
        )
        payload['decision_table'][0]['player_name'] = 'Bad </script><script>alert(1)'
        out = Path(self.tmp.name) / 'board.html'
        render_dashboard_html(payload, out, generated_label='x')
        html = out.read_text(encoding='utf-8')
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn('<\\/script>', html)
