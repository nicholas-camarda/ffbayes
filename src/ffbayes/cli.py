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


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name='pipeline',
        module='ffbayes.run_pipeline',
        help_text='Run the full end-to-end pipeline.',
    ),
    CommandSpec(
        name='split',
        module='ffbayes.run_pipeline_split',
        help_text='Run the pre-draft split pipeline.',
    ),
    CommandSpec(
        name='pre-draft',
        module='ffbayes.run_pipeline_split',
        help_text='Run only the pre-draft pipeline.',
        aliases=('pre_draft',),
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
        name='bayes',
        module='ffbayes.analysis.hybrid_mc_bayesian',
        help_text='Run the hybrid Monte Carlo + Bayesian analysis.',
    ),
    CommandSpec(
        name='agg',
        module='ffbayes.analysis.bayesian_team_aggregation',
        help_text='Build team aggregation outputs.',
    ),
    CommandSpec(
        name='compare',
        module='ffbayes.analysis.model_comparison_framework',
        help_text='Compare candidate models.',
    ),
    CommandSpec(
        name='viz',
        module='ffbayes.visualization.create_team_aggregation_visualizations',
        help_text='Generate team aggregation visualizations.',
    ),
    CommandSpec(
        name='draft-strategy',
        module='ffbayes.draft_strategy.draft_decision_strategy',
        help_text='Generate the live draft command center and workbook.',
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
        name='compare-strategies',
        module='ffbayes.analysis.draft_strategy_comparison',
        help_text='Compare draft strategy variants.',
    ),
    CommandSpec(
        name='bayesian-vor',
        module='ffbayes.analysis.bayesian_vor_comparison',
        help_text='Compare Bayesian and VOR approaches.',
    ),
    CommandSpec(
        name='publish',
        module='ffbayes.publish_artifacts',
        help_text='Mirror selected runtime artifacts into cloud storage.',
    ),
    CommandSpec(
        name='publish-pages',
        module='ffbayes.publish_pages',
        help_text='Stage the live dashboard for GitHub Pages.',
    ),
    CommandSpec(
        name='refresh-dashboard',
        module='ffbayes.refresh_dashboard',
        help_text='Regenerate dashboard HTML from the current runtime payload.',
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
        description='Unified command-line entry point for the FFBayes project.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  ffbayes collect --years 2021,2022,2023\n'
            '  ffbayes preprocess\n'
            '  ffbayes split\n'
            '  ffbayes draft-strategy --draft-position 10\n'
            '  ffbayes draft-retrospective --import-finalized ~/Downloads/ffbayes_finalized_*_2026_*\n'
            '  ffbayes draft-retrospective --year 2026\n'
            '  ffbayes refresh-dashboard --year 2025 --stage-pages\n'
            '  ffbayes publish --year 2025\n\n'
            '  ffbayes publish-pages --year 2025\n\n'
            'Any extra arguments after the command are forwarded to the existing '
            'module-level CLI.'
        ),
    )
    parser.add_argument('--version', action='version', version=f'ffbayes {_version()}')

    subparsers = parser.add_subparsers(dest='command', metavar='command')
    for spec in COMMANDS:
        subparsers.add_parser(
            spec.name,
            add_help=False,
            help=spec.help_text,
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
        print(f"Unknown command: {command}", file=sys.stderr)
        return 2

    return _run_module(spec.module, [*spec.argv_prefix, *argv])


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the consolidated `ffbayes` executable."""
    parser = build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)

    if not args_list:
        parser.print_help()
        return 0

    parsed_args, remaining_args = parser.parse_known_args(args_list)
    if parsed_args.command is None:
        parser.print_help()
        return 0

    return dispatch(parsed_args.command, remaining_args)


if __name__ == '__main__':
    raise SystemExit(main())
