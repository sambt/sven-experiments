"""CPU tests for the figure registry + override layer (`analysis/paper_assets/figspec.py`)
and the notebook API (`notebook.py`) -- the machinery that lets the paper's figures be
drawn, edited and saved from `analysis/notebooks/paper/`.

The point of the layer is that it changes NOTHING until somebody asks it to: with an empty
override file a builder must draw exactly what it drew before.  These tests pin that, the
precedence order (defaults < file < call), and the two halves of the split (draw writes
nothing; save writes the PDF, the PNG twin and the provenance sidecar).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import pytest                     # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))
sys.path.insert(0, str(REPO / 'analysis'))

from paper_assets import common as C     # noqa: E402
from paper_assets import figspec as F    # noqa: E402

#: every figure the paper carries, by group -- the list the registry must cover once all
#: four modules declare their specs.  Taken from figures_iclr/ as committed.
PAPER_FIGURES = {
    'main': ('headline_curves', 'headline_curves_all', 'cost_memory', 'k_sweeps',
             'hparam_landscape', 'allseed_curves', 'allseed_curves_ce_lm'),
    'reviewer': ('overparam', 'overparam_outcomes', 'batchsize', 'kappa', 'knobs',
                 'budget', 'divergence', 'divergence_grids'),
    'large': ('cifar_curves_epoch', 'cifar_curves_time', 'cifar_cost', 'cifar_gap',
              'cifar_landscape', 'cifar_rank_used', 'fig5_quality', 'fig5_cost',
              'nanogpt', 'gpt2', 'gpt2_lr', 'profile_methods', 'profile_scaling',
              'profile_batchsize', 'profile_k_sweep', 'profile_pareto',
              'profile_phases'),
    'spectra': ('spectrum_truncation', 'online_spectra', 'online_utr',
                'online_mechanism', 'online_norms', 'mnist_lr_vs_ce', 'used_rank_grid',
                'probe_spectra', 'probe_spectra_all', 'probe_metrics',
                'probe_metrics_all', 'probe_norms', 'probe_energy'),
}


# ---------------------------------------------------------------------------
# a synthetic figure, so the layer is testable without loading a scan
# ---------------------------------------------------------------------------
def _draw(ctx, opts):
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.2), squeeze=False)
    for ax in axes.ravel():
        ax.plot([1, 2, 3], [1, 4, 9], lw=opts['lw'], color='k')
        ax.set_ylabel('loss')
        ax.legend([f"n={opts['n_methods']}"], loc='upper left')
    return fig, {'axes': axes, 'provenance': C.provenance(functions=['fake'],
                                                          note='test'),
                 'n_drawn': opts['n_methods'], 'ctx_seen': ctx}


@pytest.fixture
def spec(monkeypatch):
    s = F.FigureSpec(draw=_draw, doc='synthetic', defaults={'lw': 1.0, 'n_methods': 5})
    s.name, s.module, s.group = 'fake_fig', 'main', 'testgroup'
    monkeypatch.setattr(F, '_REGISTRY', {'fake_fig': s})
    monkeypatch.setattr(F, 'registry', lambda reload=False: {'fake_fig': s})
    monkeypatch.setattr(F, 'context', lambda module, root=None, reload=False: 'CTX')
    return s


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'figure_overrides.yaml'
    path.write_text('{}\n')
    monkeypatch.setattr(F, 'OVERRIDES_PATH', path)
    monkeypatch.setattr(F, '_CACHE', {'mtime': None, 'data': {}})
    return path


# ---------------------------------------------------------------------------
# precedence and persistence
# ---------------------------------------------------------------------------
def test_defaults_win_when_nothing_is_pinned(spec, store):
    assert F.figure_opts('fake_fig', spec.defaults) == {'lw': 1.0, 'n_methods': 5}


def test_precedence_is_defaults_then_file_then_call(spec, store):
    F.set_overrides('fake_fig', lw=2.0, n_methods=9)
    assert F.figure_opts('fake_fig', spec.defaults) == {'lw': 2.0, 'n_methods': 9}
    # a call argument beats the file; an explicit None means "not passed"
    assert F.figure_opts('fake_fig', spec.defaults, lw=3.0)['lw'] == 3.0
    assert F.figure_opts('fake_fig', spec.defaults, lw=None)['lw'] == 2.0


def test_global_entry_applies_to_every_figure_but_yields_to_its_own(spec, store):
    F.set_overrides(F.GLOBAL_KEY, lw=5.0, n_methods=1)
    F.set_overrides('fake_fig', n_methods=7)
    opts = F.figure_opts('fake_fig', spec.defaults)
    assert (opts['lw'], opts['n_methods']) == (5.0, 7)


def test_dict_valued_options_merge_rather_than_replace(spec, store):
    F.set_overrides('fake_fig', legend={'loc': 'lower left', 'ncol': 6})
    opts = F.figure_opts('fake_fig', spec.defaults, legend={'ncol': 3})
    assert opts['legend'] == {'loc': 'lower left', 'ncol': 3}


def test_pinning_none_unpins_and_an_empty_entry_disappears(spec, store):
    F.set_overrides('fake_fig', lw=2.0)
    F.set_overrides('fake_fig', lw=None)
    assert F.overrides_for('fake_fig') == {}
    assert 'fake_fig' not in F.load_overrides()


def test_a_hand_edit_is_picked_up_without_a_kernel_restart(spec, store):
    assert F.overrides_for('fake_fig') == {}
    store.write_text('fake_fig:\n  lw: 4.0\n')
    assert F.overrides_for('fake_fig') == {'lw': 4.0}


def test_pinning_an_unknown_figure_fails_fast(spec, store):
    with pytest.raises(KeyError):
        F.set_overrides('no_such_figure', lw=1.0)


def test_the_committed_override_file_is_loadable_and_only_names_real_figures():
    data = F.load_overrides()
    assert isinstance(data, dict)
    known = set(F.registry()) | {F.GLOBAL_KEY}
    assert set(data) <= known, set(data) - known


# ---------------------------------------------------------------------------
# draw / save
# ---------------------------------------------------------------------------
def test_draw_writes_nothing_and_returns_axes_and_provenance(spec, store, monkeypatch):
    calls = []
    monkeypatch.setattr(C, 'save_fig', lambda *a, **k: calls.append(a))
    fig, meta, opts = F.draw_figure('fake_fig')
    assert calls == []                              # nothing written
    assert meta['axes'].shape == (1, 2)
    assert meta['provenance']['functions'] == ['fake']
    assert meta['options'] == opts == {'lw': 1.0, 'n_methods': 5}
    assert meta['ctx_seen'] == 'CTX'                # the module's context was passed in
    plt.close(fig)


def test_draw_applies_a_pinned_knob_and_the_generic_cosmetics(spec, store):
    F.set_overrides('fake_fig', n_methods=11, ylabel='Validation loss',
                    ylim=[0.1, 10.0], legend={'loc': 'lower right'})
    fig, meta, opts = F.draw_figure('fake_fig')
    ax = meta['axes'][0][0]
    assert meta['n_drawn'] == 11                    # the builder saw the knob
    assert ax.get_ylabel() == 'Validation loss'     # applied after the builder drew
    assert ax.get_ylim() == (0.1, 10.0)
    assert ax.get_legend()._loc == 4                # 'lower right'
    plt.close(fig)


def test_generic_cosmetics_can_target_one_panel(spec, store):
    fig, meta, _ = F.draw_figure('fake_fig', title={1: 'only the second'})
    assert [ax.get_title() for ax in meta['axes'].ravel()] == ['', 'only the second']
    plt.close(fig)


def test_apply_opts_is_a_no_op_without_cosmetic_keys(spec, store):
    fig, axes = plt.subplots(1, 1)
    axes.set_xlabel('kept')
    F.apply_opts(fig, axes, {'lw': 3.0})
    assert axes.get_xlabel() == 'kept'
    plt.close(fig)


def test_save_writes_pdf_png_and_the_provenance_sidecar(spec, store, tmp_path,
                                                        monkeypatch):
    monkeypatch.setattr(C, 'FIG_DIR', tmp_path / 'figures_iclr')
    monkeypatch.setattr(C, 'MANUSCRIPT', tmp_path / 'nope')
    monkeypatch.setattr(C, 'LAB', tmp_path / 'lab')
    fig, meta, _ = F.draw_figure('fake_fig')
    pdf, png = F.save_figure('fake_fig', fig, meta)
    assert pdf == tmp_path / 'figures_iclr' / 'testgroup' / 'fake_fig.pdf'
    assert pdf.exists() and png.exists()
    side = pdf.with_name('fake_fig.pdf.provenance.json')
    assert json.loads(side.read_text())['functions'] == ['fake']
    plt.close(fig)


def test_save_refuses_a_figure_with_no_provenance(spec, store):
    fig, axes = plt.subplots()
    with pytest.raises(RuntimeError, match='provenance'):
        F.save_figure('fake_fig', fig, {'axes': axes})
    plt.close(fig)


# ---------------------------------------------------------------------------
# the registry over the real modules
# ---------------------------------------------------------------------------
def test_registry_covers_every_paper_figure_exactly_once():
    reg = F.registry(reload=True)
    expected = {n for names in PAPER_FIGURES.values() for n in names}
    assert set(reg) == expected, (expected - set(reg), set(reg) - expected)
    for group, names in PAPER_FIGURES.items():
        for name in names:
            assert reg[name].group == group, (name, reg[name].group)


def test_every_spec_declares_its_knobs_without_shadowing_the_generic_ones():
    reg = F.registry()
    F.check_defaults(reg)                            # raises on a collision
    for name, spec in reg.items():
        assert callable(spec.draw), name
        assert spec.doc.strip(), f'{name} has no doc'
        for key, value in spec.defaults.items():
            assert isinstance(value, (str, int, float, bool, list, tuple, dict,
                                      type(None))), (name, key, type(value))


def test_every_figure_module_exposes_a_context():
    import importlib
    for module in F.FIGURE_MODULES:
        mod = importlib.import_module(f'paper_assets.{module}')
        assert hasattr(mod, 'context'), module
        assert hasattr(mod, 'FIGURE_SPECS'), module


def test_the_notebook_api_keeps_an_interactive_backend():
    """``set_paper_style`` selects Agg; importing the notebook API must stop it doing
    that, or every inline figure in analysis/notebooks/paper/ would go nowhere."""
    import os
    from paper_assets import notebook as pf          # noqa: F401  (import sets the flag)
    assert os.environ['PAPER_ASSETS_KEEP_BACKEND'] == '1'
    before = matplotlib.get_backend()
    F.paper_style(0.32)
    assert matplotlib.get_backend() == before


def test_flat_axes_work_whatever_layout_the_builder_chose(spec, store, monkeypatch):
    """A builder lays its panels out as it needs them -- a 2-D grid, a 1-D row, one bare
    Axes -- so ``drawn.axes[0][1]`` works on one figure and raises on the next.
    ``flat``/``ax(i)`` are the accessors a notebook can use on any of them."""
    from paper_assets import notebook as pf
    fig, meta, opts = F.draw_figure('fake_fig')
    drawn = pf.Drawn(name='fake_fig', fig=fig, meta=meta, opts=opts)
    assert len(drawn.flat) == 2
    assert drawn.ax(1) is meta['axes'][0][1]
    plt.close(fig)

    one = plt.figure().add_subplot()
    assert pf.Drawn(name='fake_fig', fig=one.figure,
                    meta={'axes': one}).flat == [one]
    plt.close(one.figure)


def test_a_figure_is_saved_under_the_rcparams_it_was_drawn_with(spec, store,
                                                                monkeypatch, tmp_path):
    """`savefig` reads pdf.fonttype / savefig.bbox / the font list when it WRITES.  The
    builders set those with `common.set_paper_style` while drawing, and `draw_figure`
    restores the caller's params afterwards -- so without carrying the draw-time params
    to the save, a notebook's `f.save()` would write a different PDF from the CLI's."""
    import matplotlib

    def _paper_draw(ctx, opts):
        matplotlib.rcParams.update({'pdf.fonttype': 42, 'savefig.bbox': None,
                                    'font.size': 7.0})
        fig, ax = plt.subplots()
        return fig, {'axes': ax, 'provenance': C.provenance(functions=['fake'])}

    monkeypatch.setattr(spec, 'draw', _paper_draw)
    matplotlib.rcParams.update({'pdf.fonttype': 3, 'savefig.bbox': 'tight',
                                'font.size': 16.0})

    fig, meta, _ = F.draw_figure('fake_fig')
    # the kernel's own style is back, so the user's other plots are unaffected ...
    assert matplotlib.rcParams['pdf.fonttype'] == 3
    assert matplotlib.rcParams['savefig.bbox'] == 'tight'
    # ... and the paper params were carried along with the figure
    assert meta['rcparams']['pdf.fonttype'] == 42
    assert meta['rcparams']['savefig.bbox'] is None

    seen = {}
    monkeypatch.setattr(C, 'save_fig',
                        lambda *a, **k: seen.update(
                            fonttype=matplotlib.rcParams['pdf.fonttype'],
                            bbox=matplotlib.rcParams['savefig.bbox']) or ('pdf', 'png'))
    F.save_figure('fake_fig', fig, meta)
    assert seen == {'fonttype': 42, 'bbox': None}      # saved under the paper style
    assert matplotlib.rcParams['pdf.fonttype'] == 3    # and restored again afterwards
    plt.close(fig)
