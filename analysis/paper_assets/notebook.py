"""``paper_assets.notebook`` -- make, edit and save the paper's figures from a notebook.

    import paper_assets.notebook as pf

    pf.figures()                          # every paper figure, its knobs, its overrides
    pf.info('headline_curves')            # what this one understands, and what it is now

    f = pf.make('headline_curves', methods='all')     # draws, writes NOTHING
    f.show(zoom=2)                                    # readable on screen, unchanged on disk
    f.ax(1).set_xlabel('Epoch')                       # ... edit anything you like
    f.save()                                          # PDF + PNG + provenance, as the CLI does

    pf.pin('headline_curves', methods='all', legend={'loc': 'lower left', 'ncol': 6})
    pf.rebuild('headline_curves')                     # draw with the pinned options and save

Pinned options go to ``analysis/paper_assets/figure_overrides.yaml``, which
``python -m paper_assets`` reads as well -- so the next full rebuild reproduces what the
notebook agreed to instead of reverting it.  ``pf.unpin(name)`` goes back to the builder's
default.

The figures are drawn at their true size on the page (a 0.32-linewidth panel is 1.76 in
wide at 7 pt type), which is why :meth:`Drawn.show` exists: it magnifies the *screen*
copy only.  Judge the type size on the PNG twin under ``agent_lab/paper_assets/``, which
is written at the physical size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# keep an inline/widget backend alive: set_paper_style() would otherwise switch to Agg
# and every plt.show() in the notebook would go silently nowhere
os.environ.setdefault('PAPER_ASSETS_KEEP_BACKEND', '1')

from . import common as C          # noqa: E402
from . import figspec as F         # noqa: E402

__all__ = ['figures', 'info', 'make', 'save', 'rebuild', 'pin', 'unpin', 'pinned',
           'overrides_path', 'context', 'reload', 'Drawn', 'groups']


@dataclass
class Drawn:
    """One drawn, unsaved figure: ``.fig``, ``.axes``, ``.meta``, ``.opts``."""
    name: str
    fig: object
    meta: dict = field(default_factory=dict)
    opts: dict = field(default_factory=dict)
    ctx: object = None

    @property
    def axes(self):
        """Whatever the builder laid out -- usually the 2-D array from ``plt.subplots``."""
        return self.meta.get('axes')

    @property
    def flat(self):
        """Every panel as a flat list, in row-major order.

        The builders lay their panels out as they need them (a 2x3 grid, a 1-D row, a
        single Axes), so ``drawn.axes[0][1]`` works on one figure and raises on the next;
        ``drawn.flat[1]`` and :meth:`ax` work on all of them.
        """
        from . import figspec as _F
        return _F._axes_list(self.axes)

    def ax(self, index=0):
        """One panel by flat index -- ``f.ax(0)`` is the top-left one."""
        return self.flat[int(index)]

    @property
    def group(self):
        return F.spec_for(self.name).group

    @property
    def path(self):
        """Where :meth:`save` will write."""
        return C.fig_path(self.name, self.group)

    def save(self, png=True, close=False):
        """Write the manuscript PDF, the PNG twin and the provenance sidecar."""
        pdf, png_path = F.save_figure(self.name, self.fig, self.meta, ctx=self.ctx,
                                      png=png, close=close)
        print(f'{pdf.relative_to(C.REPO)}'
              + (f'   (png {png_path.relative_to(C.REPO)})' if png_path else ''))
        return pdf, png_path

    def pin(self, **opts):
        """Persist the options this was drawn with (or the ones given) for the CLI too."""
        return pin(self.name, **(opts or {k: v for k, v in self.opts.items()
                                 if k != 'rc'}))

    def show(self, zoom=2.0, dpi=None):
        """Display a magnified screen copy; the saved figure is unaffected."""
        import io
        from IPython.display import Image, display

        from . import figspec as _F
        buf = io.BytesIO()
        with _F.rc_of(self.meta):        # the fonts it will be saved with, not the kernel's
            self.fig.savefig(buf, format='png', dpi=(dpi or 150) * float(zoom),
                             bbox_inches=None)
        w, h = self.fig.get_size_inches()
        display(Image(data=buf.getvalue(), width=int(w * 96 * float(zoom))))
        print(f'{self.name}: {w:.2f} x {h:.2f} in on the page '
              f'(shown at {zoom:g}x)')
        return None

    def _repr_html_(self):
        w, h = self.fig.get_size_inches()
        return (f'<b>{self.name}</b> &mdash; {w:.2f} x {h:.2f} in, '
                f'group <code>{self.group}</code>; '
                f'<code>.show(zoom=2)</code>, <code>.save()</code>, '
                f'<code>.axes</code>, <code>.opts</code>')


def figures(group=None):
    """A table of every paper figure: where it goes, what it understands, what is pinned."""
    import pandas as pd
    rows = []
    for name, spec in sorted(F.registry().items(),
                             key=lambda kv: (kv[1].group, kv[0])):
        if group and spec.group != group:
            continue
        ov = F.overrides_for(name)
        rows.append({'figure': name, 'group': spec.group, 'module': spec.module,
                     'knobs': ', '.join(sorted(spec.defaults)) or '-',
                     'pinned': ', '.join(sorted(ov)) or '-',
                     'what': spec.doc.strip().split('\n')[0]})
    return pd.DataFrame(rows)


def groups():
    return sorted({s.group for s in F.registry().values()})


def info(name):
    """Print what this figure is, what it understands and what it would draw with now."""
    spec = F.spec_for(name)
    opts = F.figure_opts(name, spec.defaults)
    pinned = F.overrides_for(name)
    print(f'{name}  ->  {C.fig_path(name, spec.group).relative_to(C.REPO)}')
    print(f'module: paper_assets.{spec.module}   group: {spec.group}')
    if spec.doc:
        print('\n' + spec.doc.strip() + '\n')
    print('its own knobs (default -> effective):')
    for key in sorted(spec.defaults):
        mark = '  <- pinned' if key in pinned else ''
        print(f'  {key:22s} {spec.defaults[key]!r:>28}  ->  {opts[key]!r}{mark}')
    extra = sorted(set(pinned) - set(spec.defaults))
    if extra:
        print('generic cosmetics pinned: ' + ', '.join(f'{k}={pinned[k]!r}'
                                                       for k in extra))
    print('\ngeneric cosmetics available on every figure:\n  '
          + ', '.join(F.COMMON_KEYS))
    return spec


def make(name, root=None, ctx=None, **opts):
    """Draw one figure and write nothing.  Returns a :class:`Drawn`."""
    fig, meta, resolved = F.draw_figure(name, root=root, ctx=ctx, **opts)
    return Drawn(name=name, fig=fig, meta=meta, opts=resolved,
                 ctx=ctx if ctx is not None else
                 F.context(F.spec_for(name).module, root=root))


def save(drawn_or_name, fig=None, meta=None, **kw):
    """Save a :class:`Drawn` (or a name plus your own figure)."""
    if isinstance(drawn_or_name, Drawn):
        return drawn_or_name.save(**kw)
    return F.save_figure(drawn_or_name, fig, meta, **kw)


def rebuild(name, root=None, **opts):
    """Draw with the pinned options and save -- one figure's worth of ``-m paper_assets``."""
    drawn = make(name, root=root, **opts)
    drawn.save()
    return drawn


def pin(name, **opts):
    """Persist options for ``name`` so the CLI draws it that way too.  ``key=None`` unpins."""
    path = F.set_overrides(name, **opts)
    print(f'{name}: pinned {sorted(opts)} in {path.relative_to(C.REPO)}')
    return F.overrides_for(name)


def unpin(name=None):
    """Drop one figure's pinned options (or every one), back to the builders' defaults."""
    path = F.clear_overrides(name)
    print(f'cleared {name or "ALL"} in {path.relative_to(C.REPO)}')
    return F.load_overrides()


def pinned(name=None):
    return F.overrides_for(name) if name else F.load_overrides()


def overrides_path():
    return F.OVERRIDES_PATH


def context(module_or_figure, root=None, reload=False):
    """The data object a module's builders take -- for plotting something of your own.

    Takes a module name (``'main'``, ``'reviewer'``, ``'large'``, ``'spectra'``) or a
    figure name, and returns that module's cached context.
    """
    name = module_or_figure
    if name not in F.FIGURE_MODULES:
        name = F.spec_for(name).module
    return F.context(name, root=root, reload=reload)


def reload():
    """Re-import the figure modules, after editing a builder on disk."""
    return F.registry(reload=True)
