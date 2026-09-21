"""``analysis/paper_assets`` -- the ONE re-runnable command that produces every figure,
table and number macro the ICLR revision quotes.

    cd analysis && ../.venv/bin/python -m paper_assets                 # everything
    cd analysis && ../.venv/bin/python -m paper_assets --only main     # one module

Shared conventions live in :mod:`paper_assets.common` (output directories, the
manuscript's visual language, the macro writer, the booktabs emitter, provenance).  Each
module exposes ``build(root=None) -> dict`` and is runnable on its own
(``python -m paper_assets.main``).  Importing this package has no side effects.

Module ownership (``campaign/PAPER_PLAN.md`` section 5.4):

==========  ====================================================================
``main``    F1 F3 F12 F13 - T1 T2 T3 T4 T5 T10 T18 T20 - macro group G1
``reviewer``F4 F5 F6 F8 F10 F14 - T6 T11 T12 T13 T14 T21 - G2
``large``   F7 F11 F15 - T7 T15 T16 T17 T19 - G3
``spectra`` F2 F9 - T8 T9 - G4
==========  ====================================================================
"""
from . import common  # noqa: F401  (output dirs / conventions; no side effects)

__all__ = ['common', 'build', 'MODULES']

MODULES = common.MODULES


def build(modules=None, root=None, **kw):
    """Run the asset modules and write ``numbers_v2.tex``.  Returns ``{module: report}``."""
    import importlib
    out = {}
    for name in (modules or MODULES):
        try:
            mod = importlib.import_module(f'{__name__}.{name}')
        except ModuleNotFoundError:
            out[name] = {'status': 'not implemented yet'}
            continue
        out[name] = mod.build(root=root, **kw)
    path, written = common.write_numbers_index()
    out['numbers_v2'] = {'path': str(path), 'inputs': written}
    return out
