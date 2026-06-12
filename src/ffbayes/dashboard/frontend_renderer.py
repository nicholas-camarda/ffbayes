"""Render the dashboard HTML from the prebuilt frontend template."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

from ffbayes.utils.json_serialization import dumps_strict_json

PAYLOAD_PLACEHOLDER = '__PAYLOAD_JSON__'
GENERATED_LABEL_PLACEHOLDER = '__GENERATED_LABEL__'

RENDERER_ENV_VAR = 'FFBAYES_DASHBOARD_RENDERER'
RENDERER_LEGACY = 'legacy'
RENDERER_FRONTEND = 'frontend'
_VALID_RENDERERS = (RENDERER_LEGACY, RENDERER_FRONTEND)


def dumps_html_safe_json(payload: dict[str, Any]) -> str:
    """Serialize payload JSON safely for embedding inside a <script> tag.

    A literal ``</script>`` inside a JSON string would terminate the script
    element early (HTML parses tags before JS). ``<\\/`` is a valid JSON/JS
    string escape for ``</``, so this is lossless.
    """
    return dumps_strict_json(payload).replace('</', '<\\/')


def active_renderer() -> str:
    """Return the configured dashboard renderer (default: legacy)."""
    value = os.environ.get(RENDERER_ENV_VAR, RENDERER_LEGACY).strip().lower()
    if value not in _VALID_RENDERERS:
        raise ValueError(
            f'{RENDERER_ENV_VAR} must be one of {_VALID_RENDERERS}, got {value!r}.'
        )
    return value


def load_dashboard_template() -> str:
    """Load the packaged single-file frontend template."""
    template = resources.files('ffbayes.dashboard').joinpath(
        'assets/dashboard_template.html'
    )
    if not template.is_file():
        raise FileNotFoundError(
            'Frontend dashboard template is missing from the ffbayes package. '
            'Build it with `cd dashboard_frontend && npm ci && npm run build:template`, '
            f'or unset {RENDERER_ENV_VAR} to use the legacy renderer.'
        )
    return template.read_text(encoding='utf-8')


def render_dashboard_html(
    payload: dict[str, Any], output_path: Path | str, generated_label: str
) -> Path:
    """Inject the payload into the frontend template and write the HTML."""
    output_path = Path(output_path)
    html = load_dashboard_template()
    html = html.replace(PAYLOAD_PLACEHOLDER, dumps_html_safe_json(payload))
    html = html.replace(GENERATED_LABEL_PLACEHOLDER, generated_label)
    output_path.write_text(html, encoding='utf-8')
    return output_path
