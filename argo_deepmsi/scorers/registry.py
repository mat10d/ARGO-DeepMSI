"""Lazy registry for MSI scorer implementations."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import Callable

from .base import Scorer

_FACTORIES: dict[str, Callable[[], Scorer]] = {}

# Built-in scorer names match module names. Discover filenames without importing
# their ML stacks, so adding a scorer requires one file and one registration.
_BUILTIN_SCORERS = frozenset(
    module.name
    for module in iter_modules([str(Path(__file__).parent)])
    if not module.name.startswith("_") and module.name not in {"base", "registry"}
)


def register(name: str, factory: Callable[[], Scorer]) -> None:
    if name in _FACTORIES:
        raise KeyError(f"Scorer already registered: {name}")
    _FACTORIES[name] = factory


def get_scorer(name: str) -> Scorer:
    if name not in _FACTORIES and name in _BUILTIN_SCORERS:
        import_module(f".{name}", __package__)
    if name not in _FACTORIES:
        raise KeyError(f"Unknown scorer: {name!r}. Available: {list_scorers()}")
    return _FACTORIES[name]()


def list_scorers() -> list[str]:
    return sorted(_BUILTIN_SCORERS | _FACTORIES.keys())
