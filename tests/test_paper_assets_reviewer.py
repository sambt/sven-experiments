"""Tests for ``analysis/paper_assets/reviewer.py`` -- the App. F/G/H/M/N/P assets.

Two layers, for the two ways this module can corrupt the paper.

**The helpers, on synthetic data** (always run, no results root needed).  Every number the
reviewer appendices quote is indexed by a swept value -- a batch size, a $\\kappa$, a
``rtol``, a micro-batch size -- and TeX forbids a digit in a command name, so the module
spells those values as words.  A speller that collides (two different values giving one
macro name) would silently overwrite a number with another number, which no later review
could catch; a speller that emits a digit makes the paper not compile.  The arithmetic
helpers are checked on frames small enough to write the answer down by hand.

**The generated artefacts, if they are there** (skipped otherwise).  The macro file and
every generated table are parsed the way LaTeX would read them: a NaN that reached a macro,
an un-escaped ``%`` that comments out the rest of a row, a ``_`` in text mode, an unbalanced
``$``, a row without its ``\\\\``.  These are the failures that produce a broken build or --
worse -- a table that compiles and is wrong.
"""
from __future__ import annotations

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

from paper_assets import common as C          # noqa: E402
from paper_assets import reviewer as R        # noqa: E402


# ---------------------------------------------------------------------------
# (a) macro-name spelling
# ---------------------------------------------------------------------------
#: every value this module indexes a macro by, taken from the grids it reads
#: (EXPERIMENTS.md sections 3.1-3.5)
INDEXED_VALUES = [
    8, 16, 32, 64, 128, 256,                      # batch sizes
    1, 2, 3,                                      # kappa
    0.25, 0.5, 1.0,                               # matched effective steps
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3,  # rtol
    0.05, 0.75, 4, 48,                            # learning rates, k
    150, 170, 300, 340, 600, 675, 1200, 1350, 2500, 50000,   # n_data arms
]


@pytest.mark.parametrize('value', INDEXED_VALUES)
def test_word_is_letters_only(value):
    """A macro name may contain letters only -- the whole reason the speller exists."""
    word = R._word(value)
    assert word, f'{value!r} spelled as the empty string'
    assert re.fullmatch(r'[A-Za-z]+', word), f'{value!r} -> {word!r}'
    # and it is accepted where it will be used
    C.Macros('test').add(f'numTest{word}', 1.0)


def test_word_is_injective_over_the_values_used():
    seen = {}
    for v in INDEXED_VALUES:
        w = R._word(v)
        assert w not in seen or seen[w] == float(v), \
            f'{v!r} and {seen[w]!r} both spell {w!r}'
        seen[w] = float(v)


def test_word_survives_an_unexpected_value():
    """A value the short table does not know is spelled digit by digit rather than
    dropped: a new grid point must not be able to produce an invalid macro name."""
    for v in (3.7e-7, -12.5, 1234.5, 7e5):
        assert re.fullmatch(r'[A-Za-z]+', R._word(v)), v


# ---------------------------------------------------------------------------
# (b) LaTeX fragments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value,expect', [(1e-5, '10^{-5}'), (1e-6, '10^{-6}'),
                                          (0.001, '0.001'), (0.01, '0.01'),
                                          (0.3, '0.3'), (1.0, '1')])
def test_tex_num_writes_a_power_of_ten_as_one(value, expect):
    """``f'{1e-5:g}'`` is ``1e-05``, which is what a table must never print."""
    assert R._tex_num(value) == expect


def test_tex_num_is_not_nan():
    assert R._tex_num(float('nan')) == '--'
    assert R._tex_num(None) == '--'


def test_tex_list_is_balanced_math():
    body = R._tex_list([1e-5, 1e-4, 0.001, 0.01])
    assert body.count('$') % 2 == 0
    C.check_raw(C.Raw(body), 'tex_list')
    assert '1e-05' not in body


def test_join_switches_separator_when_an_item_has_a_comma():
    """Scan labels contain commas ("MNIST CE, micro-batch"), so a comma-joined list of
    them is unreadable -- and the paper would quote it verbatim."""
    assert R._join(['a', 'b', 'c']) == 'a, b and c'
    joined = R._join(['MNIST CE, micro-batch', 'nanoGPT', 'GPT-2 small'])
    assert '; ' in joined and joined.endswith(' and GPT-2 small')
    assert R._join([]) == ''
    assert R._join(['only']) == 'only'


def test_gap_cell_stars_a_resolved_gap_and_never_prints_nan():
    sig = pd.Series({'t': -2.84, 'significant': True})
    ties = pd.Series({'t': -0.0672, 'significant': False})
    assert '^{*}' in str(R._gap_cell(sig))
    assert '^{*}' not in str(R._gap_cell(ties))
    assert R._gap_cell(None) == '--'
    for cell in (R._gap_cell(sig), R._gap_cell(ties)):
        assert 'nan' not in str(cell)
        C.check_raw(C.Raw(str(cell)), 'gap cell')


# ---------------------------------------------------------------------------
# (c) the small frame helpers
# ---------------------------------------------------------------------------
def test_num_and_row_and_sven():
    frame = pd.DataFrame({'method': ['Sven', 'Adam', 'SGD'],
                          'batch_size': [8.0, 8.0, 16.0],
                          'value': [0.1, 0.2, np.nan]})
    assert R._num('x') != R._num('x')                      # NaN
    assert R._num(None) != R._num(None)
    assert R._num('2.5') == 2.5
    assert R._sven(frame)['value'] == 0.1
    assert R._row(frame, method='SGD', batch_size=16)['method'] == 'SGD'
    assert R._row(frame, method='Muon') is None
    assert R._row(frame, no_such_column=1) is None


def test_maybe_skips_a_missing_number_but_adds_a_real_one():
    m = C.Macros('test')
    assert R._maybe(m, 'numA', float('nan')) is None
    assert R._maybe(m, 'numB', None) is None
    assert R._maybe(m, 'numC', 0.5) == 'numC'
    assert 'numA' not in m and 'numB' not in m and 'numC' in m


def test_reach_ratio_divides_by_the_arm_median_and_flags_partial_seeds():
    """The panel is scale-free on purpose: 200 full-batch epochs on the synthetics and 20
    on MNIST cannot share a linear axis."""
    reach = pd.DataFrame({
        'n_data': [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
        'method': ['Sven', 'Adam', 'SGD'] * 2,
        'epochs': [5.0, 10.0, 20.0, 8.0, 4.0, 2.0],
        'all_reached': [True, True, True, False, True, True],
        'n_reached': [5, 5, 5, 3, 5, 5],
        'n_runs': [5, 5, 5, 5, 5, 5],
    })
    out = R._reach_ratio({'P': 400, 'reach': reach})
    first = out[out['n_data'] == 100.0].iloc[0]
    assert first['median_epochs'] == 10.0
    assert first['ratio'] == pytest.approx(0.5)
    assert first['P_over_N'] == pytest.approx(4.0)
    assert bool(first['all_reached'])
    second = out[out['n_data'] == 200.0].iloc[0]
    assert second['ratio'] == pytest.approx(2.0)
    assert not bool(second['all_reached'])           # hollow marker in the figure


def test_reach_ratio_survives_an_arm_nothing_reached():
    reach = pd.DataFrame({'n_data': [100.0, 100.0], 'method': ['Sven', 'Adam'],
                          'epochs': [np.nan, np.nan], 'all_reached': [False, False],
                          'n_reached': [0, 0], 'n_runs': [5, 5]})
    out = R._reach_ratio({'P': 100, 'reach': reach})
    assert len(out) == 1 and not np.isfinite(out['ratio'].iloc[0])


def test_baseline_mean_rank_ignores_sven_and_takes_the_strongest():
    ranks = pd.DataFrame({'n_data': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                          'method': ['Sven', 'Adam', 'SGD'] * 2,
                          'rank': [1, 2, 3, 1, 4, 2]})
    over = {'toy': {'ranks': {'final_val_loss': ranks, 'final_test_loss': ranks}}}
    value, who = R._baseline_mean_rank(over, 'rank_val')
    assert value == pytest.approx(2.5)               # SGD (3, 2), Adam is (2, 4) = 3.0
    assert who == 'SGD'


def test_baseline_mean_rank_with_no_baseline():
    ranks = pd.DataFrame({'n_data': [1.0], 'method': ['Sven'], 'rank': [1]})
    over = {'toy': {'ranks': {'final_val_loss': ranks, 'final_test_loss': ranks}}}
    value, who = R._baseline_mean_rank(over, 'rank_val')
    assert not np.isfinite(value) and who == '--'


def test_knob_levels_is_the_union_of_the_family():
    per = {'a': {'values': [1.0, 2.0, 4.0]}, 'b': {'values': [1.0, 64.0]}}
    assert R._knob_levels(per, 'microbatch_size') == [1.0, 2.0, 4.0, 64.0]


# ---------------------------------------------------------------------------
# (d) the module's contract
# ---------------------------------------------------------------------------
def test_dry_run_declares_every_planned_asset_and_writes_nothing(tmp_path):
    before = {p: p.stat().st_mtime for p in C.TABLE_DIR.glob('*.tex')} \
        if C.TABLE_DIR.is_dir() else {}
    rep = R.build(dry_run=True)
    assert rep['status'] == 'dry run'
    figs = {Path(p).stem for p in rep['figures']}
    tabs = {Path(p).stem for p in rep['tables']}
    # PAPER_PLAN section 5: F4 F5 F6 F8 F10 F14 and T6 T11 T12 T13 T14 T21
    assert {'budget', 'overparam', 'batchsize', 'kappa', 'knobs',
            'divergence'} <= figs
    assert {'budget', 'equal_budget', 'overparam', 'batchsize', 'kappa',
            'knobs', 'divergence'} <= tabs
    after = {p: p.stat().st_mtime for p in C.TABLE_DIR.glob('*.tex')} \
        if C.TABLE_DIR.is_dir() else {}
    assert before == after, 'a dry run wrote into the manuscript'


def test_every_section_is_reachable():
    assert set(R.SECTIONS) == {'budget', 'overparam', 'batchsize', 'kappa', 'knobs',
                               'divergence'}
    for name in R.SECTIONS:
        assert callable(getattr(R, f'build_{name}'))


def test_scan_labels_are_macro_safe():
    """Scan labels end up inside macro bodies and table cells, so they may not carry a
    bare ``_`` or ``%``; several scan directory names do."""
    for scan, label in R.SCAN_LABELS.items():
        C.check_raw(C.Raw(label), f'label for {scan}')
        assert '_' not in label.replace('\\_', '') or '$' in label
    for scans in (R.OVERPARAM_SCANS, R.MICROBATCH_SCANS, R.PARAMFRAC_SCANS):
        for _key, name in scans:
            assert name in R.SCAN_LABELS, f'{name} has no reader-facing label'
    for name in (R.BATCHSIZE_SCAN, R.KAPPA_SCAN, *R.BUDGET_SCANS):
        assert name in R.SCAN_LABELS


def test_task_keys_cover_every_scan_family():
    for scans in (R.OVERPARAM_SCANS, R.MICROBATCH_SCANS, R.PARAMFRAC_SCANS):
        for key, _name in scans:
            assert key in R.TASK_KEY and key in R.SCAN_COLORS
    for scan in R.BUDGET_SCANS:
        assert R.SCAN_TASK[scan] in R.TASK_KEY


# ---------------------------------------------------------------------------
# (e) the generated artefacts, read the way LaTeX reads them
# ---------------------------------------------------------------------------
MACROS = C.NUM_DIR / 'numbers_v2_reviewer.tex'
TABLES = ('budget', 'equal_budget', 'overparam', 'overparam_loss',
          'batchsize', 'batchsize_loss', 'kappa', 'knobs', 'divergence')

needs_build = pytest.mark.skipif(not MACROS.is_file(),
                                 reason='run `python -m paper_assets.reviewer` first')

_NEWCOMMAND = re.compile(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*(?:%.*)?$')
_MATH = re.compile(r'\$[^$]*\$')


def _body_lines(path):
    """The lines LaTeX will typeset: comments (and the ``%`` that ends a
    ``\\resizebox{...}{%`` line) are not content."""
    for line in path.read_text().splitlines():
        if line.lstrip().startswith('%') or not line.strip():
            continue
        yield line


@needs_build
def test_macro_file_parses_and_carries_no_missing_number():
    names = []
    for line in _body_lines(MACROS):
        m = _NEWCOMMAND.match(line.strip())
        assert m, f'not a \\newcommand line: {line!r}'
        name, body = m.group(1), m.group(2)
        names.append(name)
        assert re.fullmatch(r'[A-Za-z]+', name)
        assert body.strip() not in ('', '--', 'nan', 'NaN', 'None', 'inf', '-inf')
        assert not re.search(r'\b(nan|None|inf)\b', body), f'{name}: {body!r}'
        assert body.count('$') % 2 == 0, f'{name}: unbalanced math in {body!r}'
        C.check_raw(C.Raw(body), f'macro {name}')
    assert len(names) == len(set(names)), 'a macro is defined twice'
    assert len(names) > 50, f'only {len(names)} macros -- a section did not run'


@needs_build
def test_macro_names_do_not_collide_with_the_other_modules():
    mine = {m.group(1) for line in _body_lines(MACROS)
            if (m := _NEWCOMMAND.match(line.strip()))}
    for other in C.MODULES:
        if other == R.MODULE:
            continue
        path = C.NUM_DIR / f'numbers_v2_{other}.tex'
        if not path.is_file():
            continue
        theirs = {m.group(1) for line in _body_lines(path)
                  if (m := _NEWCOMMAND.match(line.strip()))}
        clash = sorted(mine & theirs)
        assert not clash, f'{other} also defines {clash}'


@needs_build
@pytest.mark.parametrize('name', TABLES)
def test_generated_table_is_compile_safe(name):
    path = C.TABLE_DIR / f'{name}.tex'
    assert path.is_file(), f'{path} not generated'
    text = path.read_text()
    assert text.count(r'\begin{tabular}') == text.count(r'\end{tabular}') > 0
    assert text.count(r'\toprule') == text.count(r'\bottomrule')
    for line in _body_lines(path):
        # an un-escaped % mid-line comments out the rest of the row, including its \\
        stripped = line.rstrip()
        assert not re.search(r'(?<!\\)%.', stripped), f'{name}: mid-line % in {line!r}'
        assert stripped.count('$') % 2 == 0, f'{name}: unbalanced $ in {line!r}'
        outside = _MATH.sub('', stripped)
        assert not re.search(r'(?<!\\)_', outside), f'{name}: text-mode _ in {line!r}'
        assert not re.search(r'(?<!\\)\^', outside), f'{name}: text-mode ^ in {line!r}'
        # the tokens `str()` produces for a missing number, as WORDS and
        # case-sensitively: "nanoGPT" is a method and a lower-case "none" is this
        # module's English for "no value of this knob diverged"
        assert not re.search(r'\b(nan|NaN|None|inf|<NA>)\b', outside), \
            f'{name}: a missing number reached a cell: {line!r}'
        if ' & ' in stripped:
            assert stripped.endswith(r'\\'), f'{name}: row without \\\\: {line!r}'


@needs_build
@pytest.mark.parametrize('name', TABLES)
def test_generated_table_has_a_draft_caption_and_a_label(name):
    """The integrator owns the float; the emitter leaves the intended wording and label as
    comments so a table cannot arrive without either."""
    text = (C.TABLE_DIR / f'{name}.tex').read_text()
    assert '% caption (draft):' in text
    assert '% label:' in text


@needs_build
def test_every_declared_figure_exists_as_pdf_and_png():
    rep = R.build(dry_run=True)
    for p in rep['figures']:
        pdf = Path(p)
        assert pdf.is_file(), f'{pdf} not generated'
        assert pdf.stat().st_size > 2000, f'{pdf} is empty'
        png = C.LAB / R.GROUP / f'{pdf.stem}.png'
        assert png.is_file(), f'no PNG twin for {pdf.name} to eyeball'


def test_rtol_is_spelled_as_code_in_both_output_languages():
    """``rtol`` is an identifier, and the two constants are the only two spellings.

    The module used to write ``\\`rtol'`` -- LaTeX's quoting convention -- into matplotlib
    labels as well as into tables, and matplotlib has no LaTeX, so the figures printed a
    literal backtick and apostrophe.  The manuscript sets the name as ``\\texttt{rtol}``
    (14 times), so a table matches it and a figure uses the mathtext equivalent.
    """
    assert R.RTOL_TEX == r'\texttt{rtol}'
    assert R.RTOL_FIG.count('$') == 2 and R.RTOL_FIG.startswith('$')
    assert 'mathtt' in R.RTOL_FIG              # typewriter, and renderable by mathtext
    # and the LaTeX spelling survives the emitter's escaping / checking gate
    assert C.is_raw(R.RTOL_TEX)
    C.check_raw(C.Raw(R.RTOL_TEX), 'RTOL_TEX')


@needs_build
@pytest.mark.parametrize('name', TABLES)
def test_generated_table_never_quotes_rtol(name):
    text = (C.TABLE_DIR / f'{name}.tex').read_text()
    assert "`rtol'" not in text, f'{name}: quoted rtol -- use R.RTOL_TEX'


@needs_build
def test_macro_provenance_names_the_scans_the_numbers_came_from():
    """A campaign-wide macro (``\\numDivSvenNRuns``) is read off every on-grid scan, not
    off the 17 this module loads itself, so the sidecar has to list them."""
    import json
    side = C.LAB / 'provenance' / f'{MACROS.name}.provenance.json'
    assert side.is_file(), 'the macro file has no provenance record'
    rec = json.loads(side.read_text())
    assert rec['selection_sha256_16'], 'no selection hash recorded'
    assert rec['generated_at'] and rec['results_root']
    assert isinstance(rec.get('provisional'), list)
    scans = set(rec['scan_dirs'])
    assert set(R.BUDGET_SCANS) <= scans
    # the divergence section reads the CIFAR grids, which no other section touches
    assert any(s.startswith('cifar') for s in scans), \
        'the campaign-wide divergence scans are missing from the provenance'


@needs_build
def test_every_figure_has_a_provenance_record():
    rep = R.build(dry_run=True)
    for p in rep['figures']:
        pdf = Path(p)
        side = (C.LAB / 'provenance' / 'figures_iclr' / R.GROUP
                / f'{pdf.name}.provenance.json')
        assert side.is_file(), f'{pdf.name} has no provenance record'


@needs_build
def test_nothing_but_the_artefact_lands_in_the_manuscript_repo():
    """Provenance sidecars belong in ``agent_lab/``: the manuscript repo is Overleaf-synced."""
    assert not list(C.MANUSCRIPT.rglob('*.provenance.json'))
    for name in TABLES:
        side = (C.LAB / 'provenance' / 'tables_v2' / f'{name}.tex.provenance.json')
        assert side.is_file(), f'{name}.tex has no provenance record'
