"""Draft strategy package.

The package intentionally avoids eager submodule imports so `python -m`
entrypoints are not preloaded into `sys.modules` during orchestration.
Import concrete modules directly when you need them.
"""

__all__: tuple[str, ...] = ()
