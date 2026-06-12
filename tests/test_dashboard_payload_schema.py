"""Tests for the versioned dashboard payload contract."""

import json
import unittest
from pathlib import Path

from ffbayes.dashboard.payload_contract import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardPayloadError,
    load_dashboard_schema,
    stamp_schema_version,
    validate_dashboard_payload,
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _minimal_payload() -> dict:
    return json.loads(
        (FIXTURES_DIR / 'dashboard_payload_minimal.json').read_text(encoding='utf-8')
    )


class TestPayloadContract(unittest.TestCase):
    def test_schema_loads_and_declares_draft_2020_12(self):
        schema = load_dashboard_schema()
        self.assertEqual(
            schema['$schema'], 'https://json-schema.org/draft/2020-12/schema'
        )

    def test_minimal_fixture_validates(self):
        payload = _minimal_payload()
        self.assertIs(validate_dashboard_payload(payload, source='fixture'), payload)

    def test_stamp_sets_current_version(self):
        payload = {}
        stamp_schema_version(payload)
        self.assertEqual(
            payload['dashboard_schema_version'], DASHBOARD_SCHEMA_VERSION
        )

    def test_missing_critical_key_fails_closed(self):
        payload = _minimal_payload()
        del payload['decision_table']
        with self.assertRaises(DashboardPayloadError) as ctx:
            validate_dashboard_payload(payload, source='fixture')
        self.assertIn('decision_table', str(ctx.exception))

    def test_wrong_type_fails_closed(self):
        payload = _minimal_payload()
        payload['decision_table'] = 'not-a-list'
        with self.assertRaises(DashboardPayloadError):
            validate_dashboard_payload(payload, source='fixture')

    def test_unknown_extra_keys_are_allowed(self):
        payload = _minimal_payload()
        payload['some_future_section'] = {'anything': True}
        validate_dashboard_payload(payload, source='fixture')

    def test_legacy_payload_without_version_uses_legacy_contract(self):
        payload = _minimal_payload()
        del payload['dashboard_schema_version']
        # Legacy contract only requires the four pre-existing keys.
        validate_dashboard_payload(payload, source='fixture')
        del payload['decision_evidence']
        with self.assertRaises(DashboardPayloadError):
            validate_dashboard_payload(payload, source='fixture')

    def test_future_version_is_rejected(self):
        payload = _minimal_payload()
        payload['dashboard_schema_version'] = DASHBOARD_SCHEMA_VERSION + 1
        with self.assertRaises(DashboardPayloadError) as ctx:
            validate_dashboard_payload(payload, source='fixture')
        self.assertIn('version', str(ctx.exception).lower())


if __name__ == '__main__':
    unittest.main()
