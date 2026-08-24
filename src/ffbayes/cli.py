"""The public command-line entry point for FFBayes."""

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


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name='dashboard',
        module='ffbayes.draft_2026.dashboard_app',
        help_text='Start the local 2026 draft dashboard.',
    ),
)

_COMMAND_BY_NAME = {spec.name: spec for spec in COMMANDS}


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
            'Run `ffbayes dashboard --help` for dashboard options.'
        ),
    )
    parser.add_argument('--version', action='version', version=f'ffbayes {_version()}')

    subparsers = parser.add_subparsers(dest='command', metavar='command')
    for spec in COMMANDS:
        subparsers.add_parser(
            spec.name,
            add_help=False,
            help=spec.help_text,
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
    spec = _COMMAND_BY_NAME.get(command)
    if spec is None:
        print(f'Unknown command: {command}', file=sys.stderr)
        return 2

    return _run_module(spec.module, argv)


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
    if command not in _COMMAND_BY_NAME:
        parser.error(f'unknown command: {command}')
    remaining_args = args_list[1:]
    return dispatch(command, remaining_args)


if __name__ == '__main__':
    raise SystemExit(main())
