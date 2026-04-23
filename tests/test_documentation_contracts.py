from __future__ import annotations

import json
import shlex
from pathlib import Path

import ffbayes.cli as cli
from ffbayes.analysis.draft_retrospective import (
    build_parser as build_retrospective_parser,
)
from ffbayes.publish_pages import build_parser as build_publish_pages_parser
from ffbayes.refresh_dashboard import build_parser as build_refresh_dashboard_parser
from ffbayes.stage_dashboard import build_parser as build_stage_dashboard_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / 'README.md'
DOCS_DIR = REPO_ROOT / 'docs'
SITE_DIR = REPO_ROOT / 'site'

GUIDE_SUITE_PATHS = [
    DOCS_DIR / 'DASHBOARD_OPERATOR_GUIDE.md',
    DOCS_DIR / 'LAYPERSON_GUIDE.md',
    DOCS_DIR / 'TECHNICAL_DEEP_DIVE.md',
    DOCS_DIR / 'METRIC_REFERENCE.md',
    DOCS_DIR / 'DATA_LINEAGE_AND_PATHS.md',
]

DOC_PATHS = [
    README_PATH,
    DOCS_DIR / 'README.md',
    DOCS_DIR / 'OUTPUT_EXAMPLES.md',
    *GUIDE_SUITE_PATHS,
]


def _reject_json_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def _loads_strict_json(text):
    return json.loads(text, parse_constant=_reject_json_constant)


REQUIRED_GUIDE_MARKERS = [
    'Audience:',
    'Scope:',
    'Trust boundary:',
    '## What This Is',
    '## When To Use It',
    '## What To Inspect',
    '## What Not To Infer',
    '## Commands And Paths',
]

REQUIRED_CANONICAL_TERMS = [
    'Board value score',
    'Simple VOR proxy',
    'Availability to next pick',
    'Expected regret',
    'Fragility score',
    'Upside score',
    'Decision evidence',
    'Freshness and provenance',
]

COMMAND_SOURCE_ALLOWLIST = {
    'pre-draft': {'--year'},
    'draft-strategy': {
        '--draft-position',
        '--league-size',
        '--risk-tolerance',
        '--all-slots',
        '--output-dir',
    },
    'publish': {'--year'},
    'refresh-dashboard': {
        option
        for action in build_refresh_dashboard_parser()._actions
        for option in action.option_strings
        if option.startswith('--')
    },
    'stage-dashboard': {
        option
        for action in build_stage_dashboard_parser()._actions
        for option in action.option_strings
        if option.startswith('--')
    },
    'publish-pages': {
        option
        for action in build_publish_pages_parser()._actions
        for option in action.option_strings
        if option.startswith('--')
    },
    'draft-retrospective': {
        option
        for action in build_retrospective_parser()._actions
        for option in action.option_strings
        if option.startswith('--')
    },
    'collect': set(),
    'validate': set(),
    'preprocess': set(),
    'mc': set(),
    'agg': set(),
    'compare': set(),
    'bayesian-vor': set(),
    'draft-backtest': set(),
}


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _all_doc_text() -> str:
    return '\n'.join(_read(path) for path in DOC_PATHS)


def _extract_bash_commands(path: Path) -> list[str]:
    commands: list[str] = []
    inside_bash = False
    current: list[str] = []

    for raw_line in _read(path).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == '```bash':
            inside_bash = True
            current = []
            continue
        if inside_bash and stripped == '```':
            if current:
                commands.extend(_normalize_bash_lines(current))
            inside_bash = False
            current = []
            continue
        if inside_bash:
            current.append(line)
    return commands


def _normalize_bash_lines(lines: list[str]) -> list[str]:
    commands: list[str] = []
    active = ''
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if active:
            active = f'{active} {line}'
        else:
            active = line
        if active.endswith('\\'):
            active = active[:-1].rstrip()
            continue
        commands.append(active)
        active = ''
    if active:
        commands.append(active)
    return [command for command in commands if command.startswith('ffbayes ')]


def _extract_flags(command: str) -> tuple[str, list[str]]:
    tokens = shlex.split(command)
    assert tokens[0] == 'ffbayes'
    subcommand = tokens[1]
    flags = []
    for token in tokens[2:]:
        if token.startswith('--'):
            flags.append(token.split('=', 1)[0])
    return subcommand, flags


def test_docs_index_links_the_full_guide_suite():
    docs_index = _read(DOCS_DIR / 'README.md')
    for relative_name in [
        'DASHBOARD_OPERATOR_GUIDE.md',
        'LAYPERSON_GUIDE.md',
        'TECHNICAL_DEEP_DIVE.md',
        'METRIC_REFERENCE.md',
        'DATA_LINEAGE_AND_PATHS.md',
        'OUTPUT_EXAMPLES.md',
    ]:
        assert relative_name in docs_index


def test_each_guide_has_shared_documentation_conventions():
    for path in GUIDE_SUITE_PATHS:
        text = _read(path)
        for marker in REQUIRED_GUIDE_MARKERS:
            assert marker in text, f'{path.name} is missing {marker!r}'


def test_documented_commands_use_supported_subcommands_and_flags():
    valid_subcommands = {spec.name for spec in cli.COMMANDS}

    for path in DOC_PATHS:
        for command in _extract_bash_commands(path):
            subcommand, flags = _extract_flags(command)
            assert subcommand in valid_subcommands, f'{path.name}: unknown {subcommand}'
            allowed_flags = COMMAND_SOURCE_ALLOWLIST[subcommand]
            for flag in flags:
                assert flag in allowed_flags, (
                    f'{path.name}: unsupported flag {flag} for {subcommand}'
                )


def test_docs_do_not_use_known_stale_command_examples():
    combined = _all_doc_text()
    assert '--payload /path/to/dashboard_payload.json' not in combined
    assert '--payload ' not in combined
    assert 'Just get VOR rankings' not in combined


def test_docs_path_contract_and_authority_language_remain_explicit():
    combined = _all_doc_text()

    required_paths = [
        'runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.xlsx',
        'runs/<year>/pre_draft/artifacts/draft_strategy/dashboard_payload_<year>.json',
        'runs/<year>/pre_draft/artifacts/draft_strategy/draft_board_<year>.html',
        'runs/<year>/pre_draft/artifacts/draft_strategy/draft_decision_backtest_<year_range>.json',
        'dashboard/index.html',
        'dashboard/dashboard_payload.json',
        'site/index.html',
        'site/dashboard_payload.json',
        'site/publish_provenance.json',
    ]
    for required_path in required_paths:
        assert required_path in combined

    assert 'authoritative runtime' in combined.lower()
    assert 'derived local shortcut' in combined.lower()
    assert 'derived publish surface' in combined.lower()


def test_docs_use_canonical_terms_for_core_metrics_and_trust_surfaces():
    combined = _all_doc_text()
    for term in REQUIRED_CANONICAL_TERMS:
        assert term in combined

    conflicting_terms = [
        'overall model score',
        'validated universally',
        'proves the board',
    ]
    lowered = combined.lower()
    for term in conflicting_terms:
        assert term not in lowered


def test_optional_outputs_are_clearly_marked_optional():
    output_examples = _read(DOCS_DIR / 'OUTPUT_EXAMPLES.md')
    assert '## Optional Analyses' in output_examples
    assert 'not be presented as default `ffbayes pre-draft` outputs' in output_examples


def test_committed_site_payload_contains_required_guide_fields_and_consistent_provenance():
    payload = _loads_strict_json(
        (SITE_DIR / 'dashboard_payload.json').read_text(encoding='utf-8')
    )
    provenance = _loads_strict_json(
        (SITE_DIR / 'publish_provenance.json').read_text(encoding='utf-8')
    )

    for key in [
        'runtime_controls',
        'analysis_provenance',
        'decision_evidence',
        'metric_glossary',
        'model_overview',
        'publish_provenance',
    ]:
        assert key in payload

    if isinstance(payload.get('war_room_visuals'), dict):
        assert payload['war_room_visuals']['schema_version'] == 'war_room_visuals_v1'
        for key in ['timing_frontier', 'positional_cliffs', 'comparative_explainer']:
            assert key in payload['war_room_visuals']

    assert provenance['schema_version'] == 'publish_provenance_v1'
    assert payload['publish_provenance']['schema_version'] == 'publish_provenance_v1'
    assert provenance['surface_sync']['status'] == 'synchronized'
    assert payload['publish_provenance']['surface_sync']['status'] == 'synchronized'
    assert payload['publish_provenance']['season_year'] == provenance['season_year']
    assert payload['publish_provenance']['source_html'] == provenance['source_html']
    assert (
        payload['publish_provenance']['source_payload'] == provenance['source_payload']
    )


def test_metric_reference_terms_align_with_committed_payload_labels():
    metric_reference = _read(DOCS_DIR / 'METRIC_REFERENCE.md')
    payload = _loads_strict_json(
        (SITE_DIR / 'dashboard_payload.json').read_text(encoding='utf-8')
    )
    glossary = payload.get('metric_glossary') or {}

    for key in [
        'draft_score',
        'replacement_delta',
        'availability_to_next_pick',
        'expected_regret',
        'fragility_score',
        'upside_score',
    ]:
        label = glossary[key]['label']
        assert label in metric_reference


def test_docs_pair_paths_and_commands_with_contextual_language():
    operator_guide = _read(DOCS_DIR / 'DASHBOARD_OPERATOR_GUIDE.md')
    path_guide = _read(DOCS_DIR / 'DATA_LINEAGE_AND_PATHS.md')

    assert 'Purpose:' in operator_guide
    assert 'Authority' in operator_guide
    assert 'Purpose:' in path_guide
    assert 'Authority' in path_guide
