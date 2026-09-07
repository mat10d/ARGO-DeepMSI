"""Uniform interfaces and lazy lookup for MSI scorers."""

from .base import Scorer, ScoreColumn
from .registry import get_scorer, list_scorers, register

__all__ = ["Scorer", "ScoreColumn", "get_scorer", "list_scorers", "register"]
