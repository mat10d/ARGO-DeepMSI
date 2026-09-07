"""Project-local models layered on top of LazySlide's registry.

Importing this package registers the Waiv patch encoders. Trainable CTransPath
and Wagner helpers remain lazy so catalog inspection does not load their model
stacks.
"""

from __future__ import annotations

from importlib import import_module

from . import waiv  # noqa: F401  (import triggers @register side effects)

__all__ = ["load_trainable_ctranspath", "load_wagner", "waiv"]

_LOADERS = {
    "load_trainable_ctranspath": ("ctranspath", "load_trainable_ctranspath"),
    "load_wagner": ("wagner", "load_wagner"),
}


def __getattr__(name: str):
    if name not in _LOADERS:
        raise AttributeError(name)
    module_name, attribute = _LOADERS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
