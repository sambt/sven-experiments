"""Experiment runner package.

``scan`` and ``set_seed`` are exported **lazily** (PEP 562 module ``__getattr__``):
importing them eagerly pulled in ``generic_scan`` -> torch + hydra + sven, so
``import experiments.experiment_code.grid`` -- which is deliberately torch-free so
a launcher or ``tools/reconcile.py`` can enumerate a grid in milliseconds -- paid
for a CUDA import via this file.  ``from experiments.experiment_code import scan``
and ``getattr(experiment_code, cfg.name)`` (``run.py``) keep working unchanged.
"""

#: attribute name -> the submodule that defines it, imported on first access.
_LAZY = {
    "scan": ".generic_scan",
    "run_grid": ".generic_scan",
    "execute": ".generic_scan",
    "set_seed": ".experiment_utils",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(_LAZY[name], __name__), name)
    globals()[name] = value          # subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
