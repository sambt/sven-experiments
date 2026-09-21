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
