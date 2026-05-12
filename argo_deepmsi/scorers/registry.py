"""Scorer registry — get_scorer(name) returns an instance.

The registry is populated lazily so loading one scorer doesn't drag in
heavy dependencies (TITAN, NuLite, torch) for the others.
"""

from __future__ import annotations

from typing import Callable

from .base import Scorer

_FACTORIES: dict[str, Callable[[], Scorer]] = {}


def register(name: str, factory: Callable[[], Scorer]) -> None:
    if name in _FACTORIES:
        raise KeyError(f"Scorer already registered: {name}")
    _FACTORIES[name] = factory


def get_scorer(name: str) -> Scorer:
    if name not in _FACTORIES:
        raise KeyError(f"Unknown scorer: {name!r}. Registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def list_scorers() -> list[str]:
    return sorted(_FACTORIES)
