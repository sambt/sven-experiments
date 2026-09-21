"""``paper_assets.figspec`` -- the figure registry and the override layer.

Why this exists: the paper's figures used to be buildable only by ``python -m
paper_assets``, which draws and writes in one breath, so the only way to change a legend
position or an aspect ratio was to edit the builder and rebuild the whole group.  This
module splits the two halves and puts a named options dict between them, so the same
builder can be driven from a notebook (see ``analysis/notebooks/paper/``) and the CLI
keeps producing exactly what the notebook last agreed to.

**The contract every ``paper_assets`` figure module implements** (binding; see
``campaign/FIGURE_API_CONTRACT.md``):

* a module-level ``FIGURE_SPECS``: ``{name: FigureSpec(draw=..., doc=..., defaults=...)}``
  where ``name`` is the output stem (``figures_iclr/<group>/<name>.pdf``);
* each ``draw(ctx, opts)`` returns ``(fig, meta)`` and **never writes anything**.  ``meta``
  carries ``provenance`` (a :func:`common.provenance` record) plus whatever the module's
  ``build()`` needs afterwards (the methods drawn, a frame that feeds macros, ...);
* ``context(root=None)`` returns the data object those ``draw`` functions take, and is
  cheap to call twice (the module caches it);
* ``build()`` iterates ``FIGURE_SPECS`` and does the writing, through
  :func:`common.save_fig`, so there is ONE save path.

Options resolve in this order, last one wins::

    spec.defaults  <  figure_overrides.yaml (_global, then the figure's entry)  <  call

The middle layer is the file the notebooks write, which is why a tweak made in a notebook
survives the next ``python -m paper_assets``: the CLI passes no call overrides, so it
replays the file.
"""
from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from . import common as C

#: the modules that declare figures (``campaign`` has none)
FIGURE_MODULES = ('main', 'reviewer', 'large', 'spectra')

#: where the notebooks' persisted tweaks live.  Committed on purpose: it is part of how
#: the paper's figures look, exactly like the builders are.
OVERRIDES_PATH = C.ANALYSIS / 'paper_assets' / 'figure_overrides.yaml'

#: the key inside that file whose options apply to EVERY figure
GLOBAL_KEY = '_global'


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------
@dataclass
class FigureSpec:
    """One paper figure: how to draw it, and which options it understands.

    ``draw(ctx, opts) -> (fig, meta)``.  ``defaults`` documents the figure's own knobs
    (a reader of the notebook should be able to see what can be changed without reading
    the builder); the generic cosmetics of :func:`apply_opts` work on every figure and do
    not need declaring.  ``provenance`` is an escape hatch for a module that builds its
    provenance record outside the draw function: ``provenance(ctx, meta) -> record``.
    """
    draw: Callable[..., tuple]
    doc: str = ''
    defaults: dict = field(default_factory=dict)
    provenance: Callable[..., Any] | None = None
    #: filled in by :func:`registry`
    name: str = ''
    module: str = ''
    group: str = ''

    def options(self):
        """The figure's own knobs, with the persisted overrides folded in."""
        return figure_opts(self.name, self.defaults)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, FigureSpec] = {}


def registry(reload=False):
    """``{name: FigureSpec}`` over every figure module, with ``name``/``module``/``group``
    filled in.  A name declared twice is a bug (two modules would write one PDF)."""
    global _REGISTRY
    if _REGISTRY and not reload:
        return _REGISTRY
    out: dict[str, FigureSpec] = {}
    for mod_name in FIGURE_MODULES:
        mod = importlib.import_module(f'paper_assets.{mod_name}')
        if reload:
            mod = importlib.reload(mod)
        specs = getattr(mod, 'FIGURE_SPECS', None)
        if not specs:
            continue
        for name, spec in specs.items():
            if name in out:
                raise RuntimeError(f'figure {name!r} is declared by both '
                                   f'{out[name].module} and {mod_name}')
            spec = copy.copy(spec)
            spec.name = name
            spec.module = mod_name
            spec.group = spec.group or getattr(mod, 'GROUP', mod_name)
            out[name] = spec
    _REGISTRY = out
    return out


def spec_for(name, reload=False):
    reg = registry(reload=reload)
    if name not in reg:
        near = [n for n in reg if name in n or n in name]
        raise KeyError(f'no figure {name!r}; {len(reg)} known'
                       + (f', did you mean {near}?' if near else ''))
    return reg[name]


def context(module, root=None, reload=False):
    """The data object ``module``'s draw functions take, cached by the module itself."""
    mod = importlib.import_module(f'paper_assets.{module}')
    if not hasattr(mod, 'context'):
        raise RuntimeError(f'paper_assets.{module} has no context(); see '
                           f'campaign/FIGURE_API_CONTRACT.md')
    return mod.context(root=root, reload=reload)


# ---------------------------------------------------------------------------
# The override store
# ---------------------------------------------------------------------------
_CACHE: dict = {'mtime': None, 'data': {}}


def _yaml():
    import yaml                                     # a hydra dependency; always present
    return yaml


def load_overrides(path=None):
    """The persisted overrides, ``{figure_name: {...}}`` plus ``_global``.

    Re-read whenever the file changes, so editing the YAML by hand takes effect in a live
    notebook kernel without a restart.
    """
    p = path or OVERRIDES_PATH
    if not p.exists():
        return {}
    # (mtime, size), not mtime alone: two writes inside one filesystem clock tick --
    # pf.pin() followed by a hand edit, or the other way round -- carry the same
    # timestamp, and a stale read would silently ignore the newer one
    st = p.stat()
    mtime = (st.st_mtime_ns, st.st_size)
    if _CACHE['mtime'] != mtime or path is not None:
        data = _yaml().safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f'{p} must be a mapping of figure name -> options')
        if path is not None:
            return data
        _CACHE.update(mtime=mtime, data=data)
    return _CACHE['data']


def save_overrides(data, path=None):
    p = path or OVERRIDES_PATH
    head = ('# Per-figure option overrides for the paper figures.\n'
            '#\n'
            '# WRITTEN BY THE NOTEBOOKS in analysis/notebooks/paper/ (pf.set_overrides)\n'
            '# and read by BOTH the notebooks and `python -m paper_assets`, which is what\n'
            '# makes a tweak made in a notebook survive the next CLI rebuild.  Hand-editing\n'
            '# is fine; a live kernel picks the change up on the next pf.make().\n'
            '#\n'
            '# Keys are figure names (the PDF stem under figures_iclr/<group>/), plus\n'
            '# `_global` for options that apply to every figure.  Each figure\'s own knobs\n'
            '# are in pf.info(<name>).defaults; the generic cosmetics (figsize, aspect,\n'
            '# xlabel/ylabel/title, xlim/ylim, xscale/yscale, legend, suptitle, grid,\n'
            '# tick_labelsize, rc) work on every figure -- see paper_assets/figspec.py.\n')
    body = _yaml().safe_dump(data, sort_keys=True, default_flow_style=False, width=100)
    p.write_text(head + body)
    st = p.stat()
    _CACHE.update(mtime=(st.st_mtime_ns, st.st_size), data=data)
    return p


def overrides_for(name, path=None):
    """The persisted options that apply to ``name``: ``_global`` then its own entry."""
    data = load_overrides(path)
    out = dict(data.get(GLOBAL_KEY) or {})
    out.update(data.get(name) or {})
    return out


def set_overrides(name, path=None, **opts):
    """Persist ``opts`` for ``name`` (merged into what is already there).

    A value of ``None`` deletes that key, which is how a notebook goes back to the
    builder's default without editing the file by hand.
    """
    if name != GLOBAL_KEY:
        spec_for(name)                                  # fail fast on a typo'd name
    data = dict(load_overrides(path))
    entry = dict(data.get(name) or {})
    for key, value in opts.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    if entry:
        data[name] = entry
    else:
        data.pop(name, None)
    return save_overrides(data, path)


def clear_overrides(name=None, path=None):
    """Drop one figure's overrides, or (``name=None``) every override there is."""
    data = dict(load_overrides(path))
    if name is None:
        data = {}
    else:
        data.pop(name, None)
    return save_overrides(data, path)


def figure_opts(name, defaults=None, **call):
    """``defaults`` < persisted overrides < ``call``, with dict-valued keys merged.

    A builder calls this once at the top: ``opts = figspec.figure_opts(name, DEFAULTS,
    **call_opts)``.  ``None`` in ``call`` means "not passed", so a notebook can forward a
    keyword it did not set.
    """
    out = copy.deepcopy(dict(defaults or {}))
    for layer in (overrides_for(name), {k: v for k, v in call.items() if v is not None}):
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                merged = dict(out[key])
                merged.update(value)
                out[key] = merged
            else:
                out[key] = value
    return out


# ---------------------------------------------------------------------------
# The generic cosmetics
# ---------------------------------------------------------------------------
#: option keys :func:`apply_opts` consumes.  A figure's own knobs must not collide with
#: these; :func:`check_defaults` enforces that at registration time.
COMMON_KEYS = ('figsize', 'xlabel', 'ylabel', 'title', 'xlim', 'ylim', 'xscale', 'yscale',
               'legend', 'suptitle', 'grid', 'tick_labelsize', 'rc')


def merge_kw(kw, **dedicated):
    """``kw`` on top of the knob-level defaults, so neither call raises.

    A builder often has both a dedicated knob and a free-form kwargs dict for the same
    matplotlib call (``figure_legend_ncol`` beside ``figure_legend_kw``).  Splatting the
    dict next to the keyword -- ``fig.legend(..., ncol=n, **opts['figure_legend_kw'])`` --
    raises ``TypeError: got multiple values for keyword argument 'ncol'`` the moment
    somebody pins the same key inside the dict, which is exactly what a person tuning a
    legend from a notebook does first.  The more specific value (the one in ``kw``) wins.
    """
    out = dict(dedicated)
    out.update(kw or {})
    return {k: v for k, v in out.items() if v is not None}


def check_defaults(specs):
    """Refuse a figure whose own knob shadows a generic one (it would be applied twice)."""
    bad = {n: sorted(set(s.defaults) & set(COMMON_KEYS)) for n, s in specs.items()
           if set(s.defaults) & set(COMMON_KEYS)}
    if bad:
        raise RuntimeError(f'figure defaults shadow the generic option names {bad}; '
                           f'rename them or drop them (figspec.COMMON_KEYS)')
    return specs


def _axes_list(axes):
    import numpy as np
    if axes is None:
        return []
    if hasattr(axes, 'ravel'):
        return list(np.asarray(axes).ravel())
    if isinstance(axes, (list, tuple)):
        out = []
        for a in axes:
            out.extend(_axes_list(a))
        return out
    return [axes]


def _per_axes(value, axes):
    """A cosmetic given as a scalar (all panels), a list (by position) or a dict keyed by
    panel index -> ``[(ax, value), ...]``."""
    flat = _axes_list(axes)
    if isinstance(value, dict):
        return [(flat[int(k)], v) for k, v in value.items() if int(k) < len(flat)]
    if isinstance(value, (list, tuple)) and not (len(value) == 2
                                                 and all(isinstance(v, (int, float))
                                                         for v in value)):
        return list(zip(flat, value))
    return [(ax, value) for ax in flat]


def apply_opts(fig, axes, opts):
    """Apply the generic cosmetics in ``opts`` to an already-drawn figure.

    A no-op when none of :data:`COMMON_KEYS` is present, which is the case on every CLI
    build with an empty override file -- that is what keeps the committed figures
    byte-identical until somebody actually asks for a change.
    """
    if not opts or not any(k in opts for k in COMMON_KEYS):
        return fig
    flat = _axes_list(axes)
    if opts.get('figsize') is not None:
        fig.set_size_inches(*opts['figsize'])
    for key, setter in (('xlabel', 'set_xlabel'), ('ylabel', 'set_ylabel'),
                        ('title', 'set_title'), ('xlim', 'set_xlim'),
                        ('ylim', 'set_ylim'), ('xscale', 'set_xscale'),
                        ('yscale', 'set_yscale')):
        if opts.get(key) is None:
            continue
        for ax, value in _per_axes(opts[key], axes):
            if value is None:
                continue
            if key in ('xlim', 'ylim') and isinstance(value, (list, tuple)):
                getattr(ax, setter)(*value)             # (lo, hi)
            else:
                getattr(ax, setter)(value)
    if opts.get('tick_labelsize') is not None:
        for ax in flat:
            ax.tick_params(labelsize=opts['tick_labelsize'])
    grid = opts.get('grid')
    if grid is not None:
        for ax in flat:
            ax.grid(**grid) if isinstance(grid, dict) else ax.grid(bool(grid))
    if opts.get('suptitle') is not None:
        fig.suptitle(opts['suptitle'])
    legend = opts.get('legend')
    if legend is not None:
        _apply_legend(fig, flat, dict(legend) if isinstance(legend, dict) else {})
    return fig


def _apply_legend(fig, flat, kw):
    """Restyle or move the legends that the builder already made.

    ``remove: True`` drops every legend; ``axes: <index>`` picks which panel's legend to
    restyle (default: every panel that has one, plus the figure legend); the remaining
    keys go straight to ``legend()`` (``loc``, ``ncol``, ``fontsize``,
    ``bbox_to_anchor``, ``frameon``, ...).  Re-creating the legend from the existing
    handles is the only way to change ``loc`` -- matplotlib legends are immutable in that
    respect.
    """
    remove = bool(kw.pop('remove', False))
    which = kw.pop('axes', None)
    targets = flat if which is None else [flat[int(which)]]
    # ``axes=<i>`` means "that panel's legend and nothing else"; without it, every panel
    # legend AND the figure legend are restyled, since a builder may have made either
    figure_legends = list(getattr(fig, 'legends', [])) if which is None else []
    for ax in targets:
        leg = ax.get_legend()
        if leg is None:
            continue
        if remove:
            leg.remove()
            continue
        handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
        labels = [t.get_text() for t in leg.get_texts()]
        leg.remove()
        if kw:
            ax.legend(handles, labels, **kw)
        else:
            ax.legend(handles, labels)
    for leg in figure_legends:
        handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
        labels = [t.get_text() for t in leg.get_texts()]
        loc = kw.get('loc', leg._loc if isinstance(leg._loc, str) else None)
        leg.remove()
        if remove:
            continue
        # a figure legend placed with 'outside lower center' keeps that placement unless
        # the caller asked for another one: re-creating it with matplotlib's default
        # would drop it on top of the panels
        fig.legend(handles, labels, **({**kw, 'loc': loc} if loc else kw))
    return fig


# ---------------------------------------------------------------------------
# Draw / save, the two halves the CLI and the notebooks share
# ---------------------------------------------------------------------------
def paper_style(fraction=0.32, keep_backend=None):
    """:func:`common.set_paper_style`, without stealing an interactive backend.

    ``set_paper_style`` selects Agg so a headless build cannot fail; inside a notebook
    that would silently kill inline display, so the notebook API keeps its backend.
    """
    if keep_backend is None:
        keep_backend = os.environ.get('PAPER_ASSETS_KEEP_BACKEND') == '1'
    if not keep_backend:
        return C.set_paper_style(fraction)
    import matplotlib
    before = matplotlib.get_backend()
    base = C.set_paper_style(fraction)
    if matplotlib.get_backend() != before:
        matplotlib.use(before, force=False)
    return base


def draw_figure(name, root=None, ctx=None, reload=False, **call_opts):
    """Build one figure and write NOTHING.  Returns ``(fig, meta, opts)``.

    ``meta`` is the builder's own return value (provenance record, methods drawn, ...)
    with ``opts`` added under ``'options'`` so a notebook can see what it drew with.
    """
    spec = spec_for(name, reload=reload)
    opts = figure_opts(name, spec.defaults, **call_opts)
    if ctx is None:
        ctx = context(spec.module, root=root)
    rc = opts.get('rc') or {}
    import matplotlib.pyplot as plt
    with plt.rc_context(rc):
        fig, meta = spec.draw(ctx, opts)
        meta = dict(meta or {})
        apply_opts(fig, meta.get('axes'), opts)
        # The builder set the paper rcParams (``common.set_paper_style``) INSIDE this
        # context, and leaving it restores whatever the caller had -- but `savefig` reads
        # `pdf.fonttype`, `savefig.bbox` and the font list at SAVE time, not at draw time.
        # So carry the params the figure was drawn under, and save under them
        # (:func:`save_figure`): otherwise a notebook's `f.save()` would write a different
        # PDF from the one `python -m paper_assets` writes.
        meta['rcparams'] = dict(plt.rcParams)
    meta['options'] = opts
    return fig, meta, opts


def save_figure(name, fig, meta=None, ctx=None, root=None, png=True, close=False):
    """Write ``fig`` exactly as ``build()`` would: the manuscript PDF, the PNG twin and
    the provenance sidecar.  Refuses a figure with no provenance record."""
    spec = spec_for(name)
    meta = dict(meta or {})
    record = meta.get('provenance')
    if record is None and spec.provenance is not None:
        record = spec.provenance(ctx if ctx is not None else context(spec.module, root),
                                 meta)
    if record is None:
        raise RuntimeError(f'{name}: no provenance record; a paper figure is never '
                           f'written without one (see FIGURE_API_CONTRACT.md)')
    with rc_of(meta):
        return C.save_fig(fig, name, spec.group, provenance_record=record, png=png,
                          close=close)


def rc_of(meta):
    """The rcParams the figure was drawn under, as a context manager.

    ``savefig`` reads ``pdf.fonttype``, ``savefig.bbox``/``pad_inches`` and the font list
    when it writes, so a figure drawn under the paper style must be SAVED under it too --
    even if the notebook has gone back to its own style in between.
    """
    import contextlib

    import matplotlib
    rc = (meta or {}).get('rcparams')
    return matplotlib.rc_context(rc) if rc else contextlib.nullcontext()
