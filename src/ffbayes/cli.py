"""Unified command-line entry point for FFBayes.

This module provides a single `ffbayes` executable that forwards to the
existing module entry points while keeping the current standalone scripts
available for direct use.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CommandSpec:
    """Description of a CLI subcommand."""

    name: str
    module: str
    help_text: str
    aliases: tuple[str, ...] = ()
    argv_prefix: tuple[str, ...] = ()
    public: bool = False


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name='dashboard',
        module='ffbayes.draft_2026.dashboard_app',
        help_text='Start the local 2026 draft dashboard.',
        public=True,
    ),
    CommandSpec(
        name='pre-draft',
        module='ffbayes.run_pipeline_split',
        help_text='Full workflow: collect data → build board → write dashboard.',
        aliases=('pre_draft', 'pipeline', 'split'),
    ),
    CommandSpec(
        name='collect',
        module='ffbayes.data_pipeline.collect_data',
        help_text='Collect raw fantasy football data.',
    ),
    CommandSpec(
        name='validate',
        module='ffbayes.data_pipeline.validate_data',
        help_text='Validate collected and derived data.',
    ),
    CommandSpec(
        name='preprocess',
        module='ffbayes.data_pipeline.preprocess_analysis_data',
        help_text='Build the analysis-ready dataset.',
    ),
    CommandSpec(
        name='mc',
        module='ffbayes.analysis.montecarlo_historical_ff',
        help_text='Run the Monte Carlo historical analysis.',
    ),
    CommandSpec(
        name='draft-strategy',
        module='ffbayes.draft_2026.pipeline',
        help_text='Build validated current-season league boards from fresh inputs.',
    ),
    CommandSpec(
        name='draft-backtest',
        module='ffbayes.analysis.draft_decision_backtest',
        help_text='Backtest draft decision strategies.',
    ),
    CommandSpec(
        name='draft-retrospective',
        module='ffbayes.analysis.draft_retrospective',
        help_text='Evaluate finalized drafts against realized season outcomes.',
    ),
    CommandSpec(
        name='bayesian-vor',
        module='ffbayes.analysis.bayesian_vor_comparison',
        help_text='Compare Bayesian and VOR approaches.',
    ),
    CommandSpec(
        name='publish',
        module='ffbayes.publish_artifacts',
        help_text='Stage GitHub Pages and mirror selected runtime artifacts to cloud.',
    ),
    CommandSpec(
        name='stage-dashboard',
        module='ffbayes.stage_dashboard',
        help_text='Refresh dashboard HTML from payload and stage GitHub Pages.',
        aliases=('stage_dashboard',),
    ),
    CommandSpec(
        name='refresh-dashboard',
        module='ffbayes.refresh_dashboard',
        help_text='Developer helper: rebuild dashboard HTML from payload only.',
    ),
)

_COMMAND_BY_NAME = {spec.name: spec for spec in COMMANDS}
_ALIAS_TO_NAME = {
    alias: spec.name for spec in COMMANDS for alias in spec.aliases
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version('ffbayes')
    except Exception:
        return '0.1.0'


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level FFBayes CLI parser."""
    parser = argparse.ArgumentParser(
        prog='ffbayes',
        description='FFBayes draft-day operator command.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Operator workflow:\n'
            '  ffbayes dashboard --year 2026\n\n'
            'Lower-level collection, validation, modelling, publishing, and retrospective\n'
            'commands remain developer/maintenance surfaces and are intentionally hidden.'
        ),
    )
    parser.add_argument('--version', action='version', version=f'ffbayes {_version()}')

    subparsers = parser.add_subparsers(dest='command', metavar='command')
    for spec in COMMANDS:
        if not spec.public:
            continue
        subparsers.add_parser(
            spec.name,
            add_help=False,
            help=spec.help_text if spec.public else argparse.SUPPRESS,
            aliases=list(spec.aliases),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f'Forwards to `{spec.module}`.',
        )

    return parser


def _normalize_exit_code(result: object) -> int:
    """Convert a module return value into an integer exit code."""
    if result is None:
        return 0
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    return 0


def _run_module(module_name: str, argv: Sequence[str]) -> int:
    """Import a module, run its `main`, and restore `sys.argv` afterwards."""
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, 'main')

    original_argv = sys.argv[:]
    sys.argv = [module_name, *argv]
    try:
        try:
            result = entrypoint()
        except SystemExit as exc:
            result = exc.code
    finally:
        sys.argv = original_argv

    return _normalize_exit_code(result)


def dispatch(command: str, argv: Sequence[str]) -> int:
    """Dispatch a parsed command to the matching module entry point."""
    canonical_name = _ALIAS_TO_NAME.get(command, command)
    spec = _COMMAND_BY_NAME.get(canonical_name)
    if spec is None:
        print(f'Unknown command: {command}', file=sys.stderr)
        return 2

    return _run_module(spec.module, [*spec.argv_prefix, *argv])


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the consolidated `ffbayes` executable."""
    parser = build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)

    if not args_list:
        parser.print_help()
        return 0

    if args_list[0] in {'--help', '-h'}:
        parser.print_help()
        return 0
    if args_list[0] == '--version':
        parser.parse_args(args_list)
        return 0

    command = args_list[0]
    canonical_name = _ALIAS_TO_NAME.get(command, command)
    if canonical_name not in _COMMAND_BY_NAME:
        parser.error(f'unknown command: {command}')
    remaining_args = args_list[1:]
    if '--help' in remaining_args and canonical_name != 'dashboard':
        print(
            f'{command}: developer/maintenance command; help is side-effect free and no work is run.'
        )
        return 0
    return dispatch(command, remaining_args)


if __name__ == '__main__':
    raise SystemExit(main())
