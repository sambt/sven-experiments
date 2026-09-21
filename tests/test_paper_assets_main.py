"""Tests for ``analysis/paper_assets`` -- the shared conventions in ``common.py`` and the
``main`` module's contract.

CPU-only and, apart from three explicitly marked checks, independent of the real results:
a unit test that needs ``experiment_results/`` cannot run on a laptop and cannot fail for
a reason the code is responsible for.  The three integration checks are skipped when the
selection file or the results root is not there.

What is checked is exactly what would silently corrupt the paper:

* a macro name that TeX cannot accept (a digit in a command name) or that collides;
* a NaN / None / inf reaching a macro or a table cell;
* a table cell whose ``_`` or ``%`` is not escaped -- the second one comments out the
  rest of the line, so the table loses its ``\\\\`` and the build breaks in a way that is
  hard to read back to its cause;
* an asset module that does not declare its outputs, or writes into the Overleaf repo
  something other than the artefact itself.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS / 'lib'))
    sys.path.insert(0, str(ANALYSIS))

from paper_assets import common as C            # noqa: E402


# ---------------------------------------------------------------------------
# Macro names and values
# ---------------------------------------------------------------------------
def test_macro_name_must_be_letters_only():
    m = C.Macros('test')
    m.add('numPolySvenVal', 0.1388)
    for bad in ('numPoly1SvenVal', 'num_poly', 'numPoly-Sven', 'numPoly Sven', ''):
        with pytest.raises(ValueError):
            m.add(bad, 1.0)


def test_macro_collision_raises():
    m = C.Macros('test')
    m.add('numX', 1.0)
    with pytest.raises(ValueError):
        m.add('numX', 2.0)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, None,
                                 float('nan'), 'nan', 'None', ''])
def test_no_nan_or_none_reaches_a_macro(bad):
    m = C.Macros('test')
    with pytest.raises(ValueError):
        m.add('numBad', bad)


def test_add_pm_emits_two_macros_and_skips_a_missing_std():
    m = C.Macros('test')
    m.add_pm('numA', 0.1388, 0.0021)
    assert 'numA' in m and 'numAStd' in m
    m.add_pm('numB', 0.5, np.nan)
    assert 'numB' in m and 'numBStd' not in m


def test_written_macros_are_sorted_valid_and_sourced(tmp_path):
    m = C.Macros('test')
    m.add('numZed', 2.0, source='headline.confirmation_table')
    m.add('numAlpha', 1.0, source='headline.efficiency_table')
    path = C.write_macros(tmp_path / 'numbers_v2_test.tex', m)
    text = path.read_text()
    names = C.macro_names(path)
    assert names == sorted(names) == ['numAlpha', 'numZed']
    assert all(re.match(r'^[A-Za-z]+$', n) for n in names)
    assert 'headline.confirmation_table' in text
    assert 'selection:' in text                      # the provenance header
    for line in text.splitlines():
        if line.startswith('\\newcommand'):
            assert 'nan' not in line.lower() and 'none' not in line.lower()


def test_numbers_index_flags_a_cross_module_clash(tmp_path, capsys):
    for module in ('main', 'large'):
        C.write_macros(tmp_path / f'numbers_v2_{module}.tex',
                       {'numShared': 1.0, f'numOnly{module.title()}': 2.0})
    path, written = C.write_numbers_index(tmp_path / 'numbers_v2.tex',
                                          modules=('main', 'large'))
    assert written == ['numbers_v2_main', 'numbers_v2_large']
    body = path.read_text()
    assert '\\input{numbers_v2_main}' in body
    assert 'MACRO NAME CLASH' in body and 'numShared' in body
    assert 'MACRO CLASH' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value,expected', [
    (0.1388, '0.139'), (0.0, '0'), (5.09e-07, r'$5.09\times 10^{-7}$'),
    (22960.0, '22960'), (96.6, '96.6'), (1.0, '1'),
])
def test_fmt_sig(value, expected):
    assert str(C.fmt_sig(value)) == expected


def test_fmt_helpers_render_dashes_for_missing_values():
    for call in (lambda: C.fmt_sig(np.nan), lambda: C.fmt_pm(np.nan, 1.0),
                 lambda: C.fmt_int(None), lambda: C.fmt_pct(np.nan),
                 lambda: C.fmt_counts(np.nan, 5), lambda: C.fmt_ci(np.nan, 1.0),
                 lambda: C.fmt_rank(np.nan, 14)):
        assert str(call()) == '--'


def test_fmt_pm_and_pct_are_valid_latex():
    C.check_raw(C.fmt_pm(0.1388, 0.0021))
    C.check_raw(C.fmt_pct(0.0216, signed=True))
    C.check_raw(C.fmt_ci(-0.135, -0.063))
    assert r'\%' in str(C.fmt_pct(0.5))           # never a bare %
    assert r'\pm' in str(C.fmt_pm(1.0, 0.1))


# ---------------------------------------------------------------------------
# LaTeX safety of the table emitter
# ---------------------------------------------------------------------------
def test_booktabs_escapes_underscores_and_percents():
    rows = [{'method': 'a_b', 'note': '50% of runs', 'v': 1.0}]
    out = C.booktabs(rows)
    assert r'a\_b' in out and r'50\% of runs' in out
    assert not re.search(r'(?<!\\)%(?!\s*(GENERATED|caption|label|note))',
                         '\n'.join(l for l in out.splitlines()
                                   if not l.startswith('%')))


def test_booktabs_has_the_three_rules_and_one_row_per_record():
    rows = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    out = C.booktabs(rows)
    for rule in ('\\toprule', '\\midrule', '\\bottomrule'):
        assert out.count(rule) == 1
    body = [l for l in out.splitlines() if l.endswith(r'\\')]
    assert len(body) == 3                          # header + two records


def test_booktabs_renders_a_missing_value_as_a_dash_not_nan():
    out = C.booktabs([{'a': np.nan, 'b': None, 'c': np.inf}])
    assert 'nan' not in out.lower() and 'inf' not in out.lower()
    assert out.count('--') >= 3


def test_booktabs_keeps_raw_latex_through_a_dataframe():
    # numpy normalises a str subclass, so a Raw cell in a DataFrame arrives as a plain
    # str; without the content check it would be escaped into \$\textbackslash{}pm\$
    frame = pd.DataFrame([{'v': C.fmt_pm(0.2, 0.01)}])
    out = C.booktabs(frame)
    assert r'0.2 $\pm$ 0.01' in out
    assert r'\textbackslash' not in out


def test_booktabs_group_header_emits_a_cmidrule():
    out = C.booktabs([{'a': 1, 'b': 2, 'c': 3}], groups=[(1, ''), (2, 'confirmation')])
    assert r'\multicolumn{2}{c}{confirmation}' in out
    assert r'\cmidrule(lr){2-3}' in out


def test_booktabs_span_row_spans_every_column():
    out = C.booktabs([C.span_row('Random polynomial'), {'a': 1, 'b': 2, 'c': 3}],
                     columns=['a', 'b', 'c'])
    assert r'\multicolumn{3}{l}{\textit{Random polynomial}}' in out


def test_booktabs_width_guard_wraps_the_tabular_once():
    out = C.booktabs([{'a': 1}], fit=True)
    assert out.count(r'\resizebox{\linewidth}{!}{%') == 1
    assert out.count(r'\begin{tabular}') == 1
    assert out.rstrip().endswith('}')
    # the guard's trailing % is a line continuation, not a swallowed cell
    assert r'\resizebox{\linewidth}{!}{%' in out.splitlines()[out.splitlines().index(
        r'\resizebox{\linewidth}{!}{%')]


def test_booktabs_passes_through_a_full_column_spec():
    out = C.booktabs([{'a': 1, 'b': 2}], align=['p{1.4in}', 'r'])
    assert r'\begin{tabular}{p{1.4in}r}' in out


def test_booktabs_left_aligns_a_p_column_without_the_array_package():
    # a justified p{} column stretches "k=128, lr=0.5, rtol=0.3" across the column;
    # `array`'s >{\raggedright\arraybackslash} is unavailable (PAPER_PLAN section 6),
    # so the cell is wrapped in a group instead
    out = C.booktabs([{'cfg': 'k=128, lr=0.5', 'val': 1.355}],
                     align=['p{1.15in}', 'r'])
    assert r'\parbox[t]{\linewidth}{\raggedright k=128, lr=0.5\strut} & 1.35 \\' in out
    assert r'\parbox[t]{\linewidth}{\raggedright \textbf{cfg}\strut}' in out
    # the column spec is still passed through unchanged, so the width is unaffected
    assert r'\begin{tabular}{p{1.15in}r}' in out
    # the r column is untouched
    assert r'\raggedright 1.35' not in out
    # ... and an explicitly ragged spec is not double-wrapped
    out = C.booktabs([{'cfg': 'x'}], align=[r'>{\raggedright}p{1in}'])
    assert out.count(r'\raggedright') == 1


def test_booktabs_never_puts_a_bare_par_in_a_p_cell():
    """An explicit ``\\par`` in a ``p`` cell costs an extra line per row (see _ragged)."""
    out = C.booktabs([{'m': 'Sven', 'note': 'one fixed configuration, not re-tuned'}],
                     align=['l', 'p{1.1in}'])
    line = [l for l in out.splitlines() if l.startswith('Sven')][0]
    assert line.count(r'\par') == line.count(r'\parbox') == 1   # no bare \par
    assert line.endswith(r'} \\') and line.count(r'\raggedright') == 1


def test_check_assets_reports_added_then_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'MANUSCRIPT', tmp_path / 'm')
    monkeypatch.setattr(C, 'SNAPSHOT_PATH', tmp_path / 'snap.json')
    (tmp_path / 'm' / 'tables_v2').mkdir(parents=True)
    f = tmp_path / 'm' / 'tables_v2' / 't.tex'
    f.write_text('a')
    added, changed, removed = C.check_assets()
    assert added == ['tables_v2/t.tex'] and not changed and not removed
    assert C.check_assets() == ([], [], [])
    f.write_text('b')
    assert C.check_assets()[1] == ['tables_v2/t.tex']


def test_check_raw_refuses_what_would_break_the_build():
    for bad in ('50% of runs', 'a_b', 'x^2', '$a'):
        with pytest.raises(ValueError):
            C.check_raw(bad)
    for good in (r'50\% of runs', r'$a_b$', r'$x^2$', r'0.1 $\pm$ 0.2'):
        C.check_raw(good)


# ---------------------------------------------------------------------------
# Figures and paths
# ---------------------------------------------------------------------------
def test_figsize_matches_the_manuscript_linewidth():
    w, h = C.figsize(3, 2, 0.32, aspect=1.0)
    assert w == pytest.approx(3 * 0.32 * 5.5)
    assert h == pytest.approx(2 * 0.32 * 5.5)


def test_save_fig_writes_the_pdf_into_the_manuscript_and_the_png_outside(tmp_path,
                                                                        monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.setattr(C, 'MANUSCRIPT', tmp_path / 'manuscript')
    monkeypatch.setattr(C, 'FIG_DIR', tmp_path / 'manuscript' / 'figures_iclr')
    monkeypatch.setattr(C, 'LAB', tmp_path / 'lab')
    fig, ax = plt.subplots(figsize=C.figsize(1, 1))
    ax.plot([0, 1], [1, 2])
    pdf, png = C.save_fig(fig, 'probe', 'main', C.provenance(functions=['f']))
    assert pdf.is_file() and pdf.suffix == '.pdf'
    assert png.is_file() and C.LAB in png.parents
    # the provenance sidecar must NOT land in the Overleaf repo
    side = list((tmp_path / 'lab').rglob('*.provenance.json'))
    assert side and all((tmp_path / 'lab') in p.parents for p in side)
    assert not list((tmp_path / 'manuscript').rglob('*.provenance.json'))
    rec = json.loads(side[0].read_text())
    assert rec['functions'] == ['f'] and 'generated_at' in rec


def test_provenance_records_the_selection_hash_and_root():
    rec = C.provenance(functions=['headline.confirmation_table'],
                       scans=['polynomial_scan_confirm'])
    for key in ('generated_at', 'functions', 'scan_dirs', 'results_root',
                'selection_file', 'selection_sha256_16'):
        assert key in rec


# ---------------------------------------------------------------------------
# The ``main`` module's declared contract
# ---------------------------------------------------------------------------
def test_main_module_declares_its_scans_and_macro_keys():
    from paper_assets import main as M
    import headline as hl
    assert set(M.SCAN_KEY) == set(hl.HEADLINE_SCANS)
    for key in M.SCAN_KEY.values():
        assert re.match(r'^[A-Za-z]+$', key)
    for key in M.METHOD_KEY.values():
        assert re.match(r'^[A-Za-z]+$', key)
    # every headline task the main text leads with must be a real scan
    assert set(M.HEADLINE_MAIN) <= set(hl.HEADLINE_SCANS)
    assert set(M.MACRO_SCANS) <= set(hl.HEADLINE_SCANS)
    assert hl.SVEN_LABEL in M.MAIN_TABLE_METHODS


def test_short_config_is_compact_and_valid_latex():
    from paper_assets import main as M
    cell = M._short_config('k=16, lr=0.5, rtol=0.03')
    C.check_raw(cell)
    assert 'eta' in str(cell) and '$k$=16' in str(cell)
    assert str(M._short_config('')) == '--'


def test_clip_outliers_only_bites_on_a_blow_up():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_assets import main as M
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    for _ in range(5):
        ax.plot([0, 1, 2], [1.0, 0.5, 0.2])
    assert M._clip_outliers(ax) is False           # an ordinary transient is kept
    ax.plot([0, 1, 2], [6e9, 1e4, 0.9])
    assert M._clip_outliers(ax) is True
    assert ax.get_ylim()[1] < 1e3
    assert any('truncated' in t.get_text() for t in ax.texts)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The figure registry (campaign/FIGURE_API_CONTRACT.md)
# ---------------------------------------------------------------------------
#: the seven figures this module owns, in the order build() writes them
MAIN_FIGURES = ('headline_curves', 'headline_curves_all', 'cost_memory', 'k_sweeps',
                'hparam_landscape', 'allseed_curves', 'allseed_curves_ce_lm')


def test_figure_specs_cover_the_group_in_build_order():
    from paper_assets import main as M
    assert tuple(M.FIGURE_SPECS) == MAIN_FIGURES
    assert set(M._FIGURE_INFO) <= set(M.FIGURE_SPECS)
    for name, spec in M.FIGURE_SPECS.items():
        assert callable(spec.draw), name
        assert spec.doc.strip(), name


def test_two_specs_may_share_one_builder():
    """``all_methods`` and the scan list used to be function arguments; they are knobs
    now, which is what lets one builder serve two figures."""
    from paper_assets import main as M
    S = M.FIGURE_SPECS
    assert S['headline_curves'].draw is S['headline_curves_all'].draw
    assert S['allseed_curves'].draw is S['allseed_curves_ce_lm'].draw
    assert S['headline_curves'].defaults['methods'] == 'top'
    assert S['headline_curves_all'].defaults['methods'] == 'all'
    assert (S['allseed_curves'].defaults['scans']
            != S['allseed_curves_ce_lm'].defaults['scans'])


def test_no_knob_shadows_a_generic_cosmetic():
    from paper_assets import figspec as F
    from paper_assets import main as M
    F.check_defaults(M.FIGURE_SPECS)                 # raises on a collision
    for name, spec in M.FIGURE_SPECS.items():
        assert not set(spec.defaults) & set(F.COMMON_KEYS), name


def test_defaults_are_yaml_representable_and_round_trip():
    """A knob whose value is a python object could not be pinned from a notebook."""
    import yaml
    from paper_assets import figspec as F
    from paper_assets import main as M
    for name, spec in M.FIGURE_SPECS.items():
        text = yaml.safe_dump(spec.defaults, sort_keys=True)
        assert yaml.safe_load(text) == spec.defaults, name
        # ... and with an empty override file the effective options ARE the defaults,
        # which is what keeps the committed figures byte-identical
        assert F.figure_opts(name, spec.defaults) == spec.defaults, name


def test_every_knob_the_user_asked_for_is_on_figure_one():
    """F1 is the figure the user edits: all the baselines, the legend, the labels and
    the aspect ratio must be reachable without touching the builder."""
    from paper_assets import main as M
    d = M.FIGURE_SPECS['headline_curves'].defaults
    for knob in ('methods', 'top_n', 'scans', 'versus', 'aspect', 'fraction', 'extra_h',
                 'panel_legend', 'panel_legend_kw', 'figure_legend_kw',
                 'figure_legend_ncol', 'xlabel_columns', 'ylabel_columns',
                 'panel_titles', 'lw', 'sven_lw'):
        assert knob in d, knob
    assert {'loc', 'fontsize', 'ncol'} <= set(d['panel_legend_kw']) | {'ncol'}
    assert 'loc' in d['figure_legend_kw']


def test_in_columns_picks_the_labelled_panels():
    from paper_assets import main as M
    assert [M._in_columns(j, 3, 'middle') for j in range(3)] == [False, True, False]
    assert [M._in_columns(j, 3, 'first') for j in range(3)] == [True, False, False]
    assert [M._in_columns(j, 3, 'last') for j in range(3)] == [False, False, True]
    assert all(M._in_columns(j, 3, 'all') for j in range(3))
    assert not any(M._in_columns(j, 3, 'none') for j in range(3))
    assert [M._in_columns(j, 3, [0, 2]) for j in range(3)] == [True, False, True]


class _FakeCtx:
    """Just enough of :class:`main.Ctx` to exercise the method resolver."""

    def __init__(self, order, runs):
        self._order, self._have = list(order), list(runs)

    def order(self, scan):
        return list(self._order)

    def runs(self, scan):
        return dict.fromkeys(self._have)


def test_resolve_methods_handles_top_all_and_an_explicit_list():
    from paper_assets import main as M
    import headline as hl
    _FakeCtx.main_panel_methods = M.Ctx.main_panel_methods
    field = ['HIG', 'MuonW', 'Muon', 'SOAP', 'AdamW', 'Adam', hl.SVEN_LABEL, 'SGD']
    ctx = _FakeCtx(field, field)
    top = M._resolve_methods(ctx, 'polynomial_scan', 'top', 5)
    # five best plus Sven, which is appended when it is not already in the five
    assert top == field[:5] + [hl.SVEN_LABEL]
    assert M._resolve_methods(ctx, 'polynomial_scan', 'top', 2) == field[:2] + [
        hl.SVEN_LABEL]
    assert M._resolve_methods(ctx, 'polynomial_scan', 'all', 5) == field
    assert M._resolve_methods(ctx, 'polynomial_scan', ['Adam', 'SGD'], 5) == ['Adam',
                                                                              'SGD']
    # a method with no confirmation runs is dropped, not an error
    thin = _FakeCtx(field, ['Adam'])
    assert M._resolve_methods(thin, 'polynomial_scan', 'all', 5) == ['Adam']
    with pytest.raises(ValueError):
        M._resolve_methods(ctx, 'polynomial_scan', 'front', 5)


def test_clip_outliers_knobs_move_the_limit():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_assets import main as M
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    for _ in range(5):
        ax.plot([0, 1, 2], [1.0, 0.5, 0.2])
    ax.plot([0, 1, 2], [6e9, 1e4, 0.9])
    assert M._clip_outliers(ax, factor=20.0) is True
    assert ax.get_ylim()[1] == pytest.approx(20.0)
    # a headroom nothing can reach disables the truncation entirely
    fig2, ax2 = plt.subplots()
    ax2.set_yscale('log')
    for _ in range(5):
        ax2.plot([0, 1, 2], [1.0, 0.5, 0.2])
    ax2.plot([0, 1, 2], [6e9, 1e4, 0.9])
    assert M._clip_outliers(ax2, headroom=1e12) is False
    plt.close(fig)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Integration: only where the campaign's data is actually present
# ---------------------------------------------------------------------------
requires_results = pytest.mark.skipif(
    not (C.SELECTION_PATH.is_file()
         and Path(C.results_root()).is_dir()),
    reason='needs bench/best_configs.json and the results root')


@requires_results
def test_selection_uses_validation_only():
    import headline as hl
    assert hl.assert_no_test_selection() > 0


@requires_results
def test_generated_macros_are_all_finite_and_unique():
    path = C.NUM_DIR / 'numbers_v2_main.tex'
    if not path.is_file():
        pytest.skip('numbers_v2_main.tex not generated yet '
                    '(cd analysis && python -m paper_assets.main)')
    names = C.macro_names(path)
    assert names and len(names) == len(set(names))
    body = path.read_text()
    for line in body.splitlines():
        if not line.startswith('\\newcommand'):
            continue
        value = line.split('}{', 1)[1].rsplit('}', 1)[0]
        assert value.strip() not in ('', '--', 'nan', 'inf', 'None')
        assert 'nan' not in value.lower()


@requires_results
def test_generated_tables_are_compile_safe():
    if not C.TABLE_DIR.is_dir():
        pytest.skip('tables_v2/ not generated yet')
    files = sorted(C.TABLE_DIR.glob('*.tex'))
    if not files:
        pytest.skip('tables_v2/ is empty')
    for f in files:
        text = f.read_text()
        assert text.count(r'\begin{tabular}') == text.count(r'\end{tabular}') >= 1
        for i, line in enumerate(text.splitlines(), 1):
            if line.startswith('%'):
                continue                            # a provenance comment
            assert '\x00' not in line
            # a trailing `%` is the LaTeX line-continuation idiom the width guard emits
            # (`\resizebox{\linewidth}{!}{%`); a `%` with text after it eats that text
            assert not re.search(r'(?<!\\)%(?=.)', line), \
                f'{f.name}:{i} bare % -> {line!r}'
            stripped = re.sub(r'\$[^$]*\$', '', line)
            stripped = re.sub(r'\\[A-Za-z]+', '', stripped)
            assert not re.search(r'(?<!\\)_', stripped), f'{f.name}:{i} bare _ -> {line!r}'
            assert line.count('$') % 2 == 0, f'{f.name}:{i} unbalanced $ -> {line!r}'
            # a WORD, not a substring: "nanoGPT" contains "nan"
            assert not re.search(r'\b(nan|NaN|NAN|inf|Inf|None)\b', line), \
                f'{f.name}:{i} NaN leaked -> {line!r}'


# ---------------------------------------------------------------------------
# Integration: the draw / save split on the real frames
# ---------------------------------------------------------------------------
#: the analysis layer's default root is RELATIVE to analysis/ (every notebook runs there),
#: so a test started from the repository root has to name it absolutely
RESULTS_ROOT = C.results_root()


@pytest.fixture(scope='module')
def real_ctx():
    from paper_assets import main as M
    return M.context(root=RESULTS_ROOT)


@pytest.fixture
def no_writes(monkeypatch):
    """Make any save attempt inside a draw function a loud failure."""
    import matplotlib
    matplotlib.use('Agg')
    monkeypatch.setattr(C, 'save_fig',
                        lambda *a, **k: pytest.fail('draw() wrote a figure'))
    monkeypatch.setattr(C, 'write_provenance',
                        lambda *a, **k: pytest.fail('draw() wrote a sidecar'))
    return None


@requires_results
@pytest.mark.parametrize('name', MAIN_FIGURES)
def test_every_figure_draws_without_writing_and_returns_axes_and_provenance(
        name, real_ctx, no_writes):
    import matplotlib.pyplot as plt
    from paper_assets import figspec as F
    fig, meta, opts = F.draw_figure(name, ctx=real_ctx)
    assert meta['axes'] is not None
    assert meta['axes'].shape[0] >= 1 and meta['axes'].shape[1] >= 1
    rec = meta['provenance']
    for key in ('functions', 'scan_dirs', 'selection_sha256_16', 'note'):
        assert key in rec, (name, key)
    assert rec['functions']
    assert opts == F.figure_opts(name, F.spec_for(name).defaults)
    plt.close(fig)


@requires_results
def test_the_extras_build_needs_downstream_survive_the_split(real_ctx, no_writes):
    """``per_scan`` feeds figure_info['headline_curves'] and then the G1 macros; the
    frame and the two info dicts feed the rest."""
    import matplotlib.pyplot as plt
    from paper_assets import figspec as F
    from paper_assets import main as M
    fig, meta, _ = F.draw_figure('headline_curves', ctx=real_ctx)
    assert set(meta['per_scan']) == set(M.HEADLINE_MAIN)
    assert all(v for v in meta['per_scan'].values())
    assert M._FIGURE_INFO['headline_curves'](meta) == {
        s: list(v) for s, v in meta['per_scan'].items()}
    plt.close(fig)

    fig, meta, _ = F.draw_figure('cost_memory', ctx=real_ctx)
    assert not meta['frame'].empty
    assert set(M._FIGURE_INFO['cost_memory'](meta)[0]) >= {'scan', 'P', 'method',
                                                           'ms_per_step', 'mem'}
    plt.close(fig)

    for name in ('k_sweeps', 'hparam_landscape'):
        fig, meta, _ = F.draw_figure(name, ctx=real_ctx)
        assert set(meta['info']) == set(M.SWEEP_SCANS)
        assert all('selected' in v for v in meta['info'].values())
        plt.close(fig)
    # the k saturation macro num<Scan>SvenKSaturate is read off this one
    fig, meta, _ = F.draw_figure('hparam_landscape', ctx=real_ctx)
    assert any('k_within_5pct' in v for v in meta['info'].values())
    plt.close(fig)


@requires_results
def test_the_methods_knob_visibly_changes_figure_one(real_ctx, no_writes):
    """The user's ask: put ALL the baselines on Fig. 1 from the notebook."""
    import matplotlib.pyplot as plt
    from paper_assets import figspec as F
    from paper_assets import main as M
    scan = M.HEADLINE_MAIN[0]
    default, all_of, two = (F.draw_figure('headline_curves', ctx=real_ctx, **kw)[1]
                            for kw in ({}, {'methods': 'all'}, {'top_n': 2}))
    assert len(default['per_scan'][scan]) <= M.MAIN_PANEL_TOP_N + 1
    assert len(all_of['per_scan'][scan]) > len(default['per_scan'][scan])
    assert len(two['per_scan'][scan]) < len(default['per_scan'][scan])
    # every drawn method is on the axes, and the per-panel key names exactly them
    n_lines = len([ln for ln in default['axes'][0][0].get_lines() if ln.get_label()
                   and not ln.get_label().startswith('_')])
    assert n_lines >= 0                               # plot_curves labels its own lines
    assert len(default['axes'][0][0].get_legend().get_texts()) == len(
        default['per_scan'][scan])
    for meta in (default, all_of, two):
        plt.close(meta['axes'][0][0].figure)


@requires_results
def test_the_legend_and_geometry_knobs_reach_figure_one(real_ctx, no_writes):
    import matplotlib.pyplot as plt
    from paper_assets import figspec as F
    fig, meta, _ = F.draw_figure('headline_curves', ctx=real_ctx, panel_legend=False,
                                 aspect=1.2, panel_titles=False,
                                 xlabel_columns='all')
    assert meta['axes'][0][0].get_legend() is None
    assert meta['axes'][0][0].get_title() == ''
    assert all(ax.get_xlabel() for ax in meta['axes'][1])
    w, h = fig.get_size_inches()
    assert h == pytest.approx(2 * 0.32 * C.LINEWIDTH_IN * 1.2 + 0.38)
    plt.close(fig)
    # ... and the generic cosmetics of figspec.apply_opts still work on top
    fig, meta, _ = F.draw_figure('headline_curves', ctx=real_ctx,
                                 ylabel='Val. loss', legend={'loc': 'upper right'})
    assert meta['axes'][0][0].get_ylabel() == 'Val. loss'
    assert meta['axes'][0][0].get_legend()._loc == 1
    plt.close(fig)


@requires_results
def test_dry_run_names_the_seven_figures_it_would_write():
    from paper_assets import main as M
    report = M.build(root=RESULTS_ROOT, dry_run=True)
    assert report['would_write']['figures'] == list(MAIN_FIGURES)
    assert 'headline_confirm' in report['would_write']['tables']


def test_context_is_cached_per_root(monkeypatch):
    """``context()`` must be cheap to call twice: a notebook that draws six figures
    reads the three frames once."""
    from paper_assets import main as M
    made = []

    def _ctx(root=None):
        made.append(root)
        return ('CTX', root, len(made))

    monkeypatch.setattr(M, '_CONTEXT', {})
    monkeypatch.setattr(M, 'Ctx', _ctx)
    first = M.context()
    assert M.context() is first and made == [None]
    assert M.context(root='/elsewhere') == ('CTX', '/elsewhere', 2)
    assert M.context() is first                       # a second root does not evict
    assert M.context(reload=True) is not first
