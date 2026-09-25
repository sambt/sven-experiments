"""Alias for :mod:`paper_assets.common`.

``campaign/PAPER_PLAN.md`` section 5.4 names the shared-conventions module ``_common.py``
and the work packages name it ``common.py``.  The module itself is ``common.py``; this
re-exports it so ``from paper_assets import _common`` and
``from paper_assets._common import save_fig`` both work and there is only one
implementation.
"""
from .common import *      # noqa: F401,F403
from .common import __all__  # noqa: F401
from . import common as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith('__')})
