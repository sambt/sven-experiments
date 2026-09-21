"""Tests for ``analysis/paper_assets/spectra.py`` -- F2, F9, T8, T9 and macro group G4.

CPU-only.  Everything except the three integration checks at the bottom runs against a
SYNTHETIC :class:`paper_assets.spectra.Data` built here, so no unit test needs
``experiment_results/``, ``analysis/ckpt_spectra/`` or a GPU, and none of them can fail
for a reason the module is not responsible for.

What is checked is what would corrupt the paper silently:

* a macro name TeX cannot accept (a digit in a command name), a duplicate, or a
  non-finite value reaching the prose -- the module's own numbers AND the ones another
  module may already have written (:func:`spectra.macro_collisions`);
* a NaN / None leaking into a table cell instead of a dash, and a cell whose ``_`` or
  ``%`` is not escaped (a bare ``%`` comments out the rest of the line, which silently
  eats a row's ``\\\\``);
* the two measurements staying apart: the online per-batch columns and the offline
  probe-set columns are different objects, so a scan with no probe cache must print
  ``--`` in the probe column rather than borrow the online number;
* the ``finished/attempted`` denominator of the diag pass coming from the PASS (the
  CIFAR-CE re-selection leaves Sven with 2-3 of 5 seeds while the other methods have 5);
* a provisional scan being stamped rather than quietly published;
* the formatters that decide whether a number is readable on the page at all: the
  round-off floor as ``$3.5\\times 10^{-4}$`` and not ``0.00035``, rtol as a power of ten,
  a seed-mean rank at one decimal.
"""
from __future__ import annotations

import copy
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS / 'lib'))
    sys.path.insert(0, str(ANALYSIS))

import matplotlib                                  # noqa: E402
matplotlib.use('Agg')

from paper_assets import common as C               # noqa: E402
from paper_assets import figspec as F              # noqa: E402
from paper_assets import spectra as S              # noqa: E402
import spectra_figs as sf                          # noqa: E402


# ---------------------------------------------------------------------------
# A synthetic Data: two scans, one of them with a probe cache and provisional
# ---------------------------------------------------------------------------
SCAN_A = 'polynomial_scan'          # a real key, so SCAN_KEY / SCAN_SHORT resolve
SCAN_B = 'cifar10_resnet_ce_scan'   # the provisional one, no probe cache
SEEDS = (2000, 2001, 2002)
N_LOG, N_STEP = 12, 240


def _diag(scan, k, B, rtol, lr, title, seeds=SEEDS):
    rows = pd.DataFrame({'model_seed': list(seeds), 'k': k, 'rtol': rtol, 'lr': lr,
                         'batch_size': B, 'optimizer': sf.SVEN_OPTIMIZER})
    return sf.SvenDiag(scan=scan, dir_name=f'{scan}_diag', title=title, rows=rows,
                       k=k, rtol=rtol, lr=lr, B=B, n_epochs=2, n_steps=N_STEP,
                       seeds=list(seeds))


def _arrays(d, n_seeds, used_first, used_final, energy_first, energy_final):
    """One :func:`spectra_figs.diag_arrays`-shaped dict per seed, with known endpoints."""
    step = np.unique(np.round(np.linspace(1, N_STEP, N_LOG)).astype(int))
    out = []
    for s in range(n_seeds):
        used = np.linspace(used_first, used_final, len(step)) + 0.1 * s
        frac = np.linspace(energy_first, energy_final, len(step))
        svs = np.geomspace(1.0, 1e-5, d.B)[None, :] * np.ones((len(step), 1))
        utr = np.sqrt(np.diff(np.concatenate([[0.0], np.linspace(0, 1, d.B)])))
        out.append({
            'step': step, 'svs': svs, 'utr': np.tile(utr, (len(step), 1)),
            'k': d.k, 'rtol': d.rtol, 'width': d.B,
            'nnz_logged': np.round(used),
            'svs_rel': svs, 'noise_rel': np.full(len(step), 3.45e-4),
            'cum_energy': np.cumsum(np.tile(utr ** 2, (len(step), 1)), axis=1),
            'rtol_rank': np.round(used), 'used_rank': np.round(used),
            'frac_top_k': np.clip(frac + 0.005, 0, 1), 'frac_used': frac,
            'update_norm': np.geomspace(1.0, 0.1, len(step)),
            'resid_norm': np.geomspace(10.0, 1.0, len(step)),
            'sv_max': np.ones(len(step)), 'sv_min_kept': np.full(len(step), 1e-3),
            'sv_noise_floor': np.full(len(step), 3.45e-4),
        })
    return out


def _probe_table(scan, methods=('Sven', 'Adam', 'MuonW', 'HIG'), seeds=SEEDS):
    steps = [0, 1, 4, 16, 64, 240]
    rows = []
    for m in methods:
        for seed in seeds:
            for i, st in enumerate(steps):
                rows.append({
                    'scan': scan, 'method': m, 'model_seed': seed, 'step': st,
                    'epoch': i, 'n_rows': 10000, 'n_params': 673,
                    'sigma_max': 1.0, 'sigma_min': 1e-8,
                    'cond': 10.0 ** (6 + 0.1 * i), 'rank': 673 - i,
                    'eff_rank': 20.0 + 10 * i, 'sigma_B_over_1': 10.0 ** (-2 + 0.1 * i),
                    'sigma_b': 32, 'sigma_b_resolved': True, 'floor': 1e-12,
                    'probe_loss': 1.0 / (1 + i), 'param_norm': 8.0 + 0.1 * i,
                    'dist_init': 0.5 * i + (0.3 if m == 'Sven' else 0.0),
                    'resid_norm': 1.0, 'proj_frac': 0.994,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def data():
    a = _diag(SCAN_A, k=16, B=32, rtol=0.03, lr=0.5, title='Random Polynomial')
    b = _diag(SCAN_B, k=128, B=128, rtol=0.3, lr=0.5, title='CIFAR-10 (CE)',
              seeds=SEEDS[:2])
    d = S.Data(
        diags={SCAN_A: a, SCAN_B: b},
        arrays={SCAN_A: _arrays(a, 3, 7.0, 13.4, 0.9225, 0.9959),
                SCAN_B: _arrays(b, 2, 107.0, 40.3, 0.9075, 0.7890)},
        seed_target={SCAN_A: 3, SCAN_B: 3},
        probe_methods={SCAN_A: ['Sven', 'Adam', 'MuonW', 'HIG']},
        probe={SCAN_A: _probe_table(SCAN_A)},
        rank_law_ok=True, rank_law_steps=46438,
        provisional=(SCAN_B,))
    d.mech = sf.mechanism_table([a, b], results_root=None) if False else pd.DataFrame([
        {'scan': a.title, 'B': a.B, 'k': a.k, 'rtol': a.rtol, 'used_first': 7.0,
         'used_final': 13.4, 'binds': 'rtol', 'energy_top_k_first': 0.9955,
         'energy_top_k_final': 0.9984, 'energy_used_first': 0.9225,
         'energy_used_final': 0.9959, 'noise_rel': 3.45e-4, 'n_seeds': 3},
        {'scan': b.title, 'B': b.B, 'k': b.k, 'rtol': b.rtol, 'used_first': 107.0,
         'used_final': 40.3, 'binds': 'rtol', 'energy_top_k_first': 1.0,
         'energy_top_k_final': 1.0, 'energy_used_first': 0.9075,
         'energy_used_final': 0.7890, 'noise_rel': 3.45e-4, 'n_seeds': 2}])
    d.widths = pd.DataFrame([{'scan': SCAN_A, 'n_rows': 10000, 'n_params': 673,
                              'width': 673, 'n_ckpts': 6, 'methods': 4, 'seeds': 3,
                              'n_used': 12, 'n_cached_any_tag': 12}])
    d.low4 = sf.low4_table({SCAN_A: d.probe[SCAN_A]})
    d.probe_energy = pd.DataFrame([
        {'scan': SCAN_A, 'k_selected': 16, 'B': 32, 'when': w, 'step': st, 'k': k,
         'top_k': v, 'top_k_std': 0.1, 'in_span': 0.9949, 'n_seeds': 3}
        for w, st, off in (('first', 0, 0.0), ('last', 240, 0.03))
        for k, v in ((8, 0.4359 + off), (16, 0.5882 + off), (32, 0.7370 + off))])
    return d


@pytest.fixture
def paper_dirs(tmp_path, monkeypatch):
    """Point every output directory at a tmp tree: a test never writes to Overleaf."""
    monkeypatch.setattr(C, 'MANUSCRIPT', tmp_path / 'manuscript')
    monkeypatch.setattr(C, 'FIG_DIR', tmp_path / 'manuscript' / 'figures_iclr')
    monkeypatch.setattr(C, 'TABLE_DIR', tmp_path / 'manuscript' / 'tables_v2')
    monkeypatch.setattr(C, 'NUM_DIR', tmp_path / 'manuscript')
    monkeypatch.setattr(C, 'LAB', tmp_path / 'lab')
    return tmp_path


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value,expected', [
    (1e-6, r'$10^{-6}$'), (1e-4, r'$10^{-4}$'), (0.001, r'$10^{-3}$'),
    (0.03, '$0.03$'), (0.3, '$0.3$'), (0.1, '$0.1$')])
def test_rtol_is_a_power_of_ten_only_when_that_is_shorter(value, expected):
    assert S._rtol_tex(value) == expected


def test_the_round_off_floor_is_printed_in_scientific_notation():
    # common.fmt_sig keeps a decimal spelling down to 1e-4, which reads as a rounding
    # artefact for a sqrt(eps) floor
    assert str(S._sci(3.4526e-4)) == r'$3.5\times 10^{-4}$'
    assert str(S._sci(1.2e-5)) == r'$1.2\times 10^{-5}$'
    assert str(C.fmt_sig(3.4526e-4)) == '0.000345'


def test_formatters_render_a_missing_value_as_a_dash():
    for fn in (S._sci, S._thousands, S._rank_cell):
        assert str(fn(float('nan'))) == '--'
        assert str(fn(None)) == '--'


def test_thousands_uses_a_tex_safe_separator():
    assert str(S._thousands(10000)) == '10{,}000'
    assert str(S._rank_cell(64)) == '64.0'


def test_probe_method_order_draws_sven_last():
    assert S._probe_method_order(['Sven', 'Adam', 'HIG']) == ['Adam', 'HIG', 'Sven']
    assert S._probe_method_order(['Adam']) == ['Adam']


def test_the_step_label_goes_where_the_data_is_not():
    """``_emptiest_corner`` picks the corner the drawn curves leave free.

    Every spectrum panel has to put two step labels and a ``k`` label somewhere, and no
    single corner is free on all seven scans (a decaying toy spectrum empties the right
    half; a nearly flat CIFAR-CE one fills it).  A wrong answer here is a label on top of
    the data in the appendix figure.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # a decaying curve: fills the upper LEFT and the lower right, frees the lower left
    # once it has dropped, and frees the upper right entirely
    x = np.linspace(0, 1, 200)
    ax.plot(x, 1.0 - x)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    assert S._emptiest_corner(ax) in ('lower left', 'upper right')
    ax.clear()
    # a curve hugging the top: the two lower corners are free, and ties go to the lower
    # left (the earlier entry), where no reference line is labelled
    ax.plot(x, np.full_like(x, 0.98))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    assert S._emptiest_corner(ax) == 'lower left'
    ax.clear()
    # nothing drawn -> the caller's default, never an exception
    assert S._emptiest_corner(ax, default='lower right') == 'lower right'
    plt.close(fig)


def test_a_two_line_panel_title_breaks_after_the_scan_name(data):
    d = data.diags[SCAN_A]
    one, two = S._panel_title(d, SCAN_A), S._panel_title(d, SCAN_A, lines=2)
    assert '\n' not in one and two.count('\n') == 1
    assert two.split('\n')[0] == S.SCAN_SHORT[SCAN_A]
    assert two.split('\n')[1] == one.split(', ', 1)[1]


def test_scan_keys_are_letters_only_and_cover_every_headline_scan():
    import headline as hl
    assert set(S.SCAN_KEY) == set(hl.HEADLINE_SCANS)
    assert set(S.SCAN_SHORT) == set(hl.HEADLINE_SCANS)
    assert set(S.SCAN_COLORS) == set(hl.HEADLINE_SCANS)
    for key in S.SCAN_KEY.values():
        assert re.fullmatch(r'[A-Za-z]+', key), key


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------
def test_rank_spread_is_relative_and_sorted_worst_first(data):
    spread = S._rank_spread(data)
    assert list(spread['scan'])[0] in (SCAN_A, SCAN_B)
    assert spread['rel'].is_monotonic_decreasing
    assert (spread['n_seeds'] > 0).all()


def test_discard_table_reads_at_fractions_of_training_and_is_floored(data):
    disc = S._discard_table(data)
    assert set(disc.columns) == {'scan'} | set(S.PROGRESS_MARKS)
    values = disc[list(S.PROGRESS_MARKS)].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= S.DISCARD_FLOOR).all()
    # the polynomial fixture ends at 99.59 % kept, i.e. ~4e-3 discarded
    poly = disc.set_index('scan').loc[SCAN_A, 1.0]
    assert 1e-3 < poly < 1e-1


def test_probe_endpoints_take_the_measured_width_not_the_parameter_count(data):
    ends = S._probe_endpoints(data).set_index('scan')
    assert ends.loc[SCAN_A, 'width'] == 673        # min(10_000 rows, 673 parameters)
    assert ends.loc[SCAN_A, 'n_rows'] == 10000
    assert ends.loc[SCAN_A, 'floor'] == 1e-12


# ---------------------------------------------------------------------------
# Macros (G4)
# ---------------------------------------------------------------------------
def test_macro_names_are_valid_latex_identifiers_and_unique(data):
    book = S.macros(data)
    assert len(book) > 30
    for name in book.values:
        assert re.fullmatch(r'[A-Za-z]+', name), name
        assert name.startswith('num'), name
    assert len(set(book.values)) == len(book.values)


def test_no_nan_none_or_inf_reaches_a_macro(data):
    book = S.macros(data)
    for name, body in book.values.items():
        text = str(body)
        assert text.strip() not in ('', '--', 'nan', 'None', 'inf'), name
        assert not re.search(r'\b(nan|NaN|inf|Inf|None)\b', text), (name, text)
        C.check_raw(body, name)


def test_every_macro_names_the_analysis_function_it_came_from(data):
    book = S.macros(data)
    assert all(book.sources.get(n) for n in book.values)


def test_a_provisional_scan_stamps_its_macros(data):
    book = S.macros(data)
    key = S.SCAN_KEY[SCAN_B]
    stamped = {n for n in book.provisional if n.startswith(f'num{key}')}
    assert stamped, 'CIFAR-CE macros must be stamped while its re-selection runs'
    fresh = S.SCAN_KEY[SCAN_A]
    assert not any(n.startswith(f'num{fresh}') for n in book.provisional)


def test_macros_carry_the_structural_facts_the_section_rests_on(data):
    book = S.macros(data)
    for name in ('numSpectraScans', 'numSpectraRankLawSteps', 'numSpectraRtolBinds',
                 'numSpectraNoiseFloorRel', 'numPolySvenUsedRankFinal',
                 'numPolySvenEnergyFinal', 'numPolySvenProbeEnergy',
                 'numPolySvenProbeSpan'):
        assert name in book, name
    # the online and the probe measurement of the SAME k must not be confused
    assert book.values['numPolySvenEnergyFinal'] != book.values['numPolySvenProbeEnergy']


def test_written_macro_file_is_sorted_and_flags_the_provisional_ones(data, paper_dirs):
    book = S.macros(data)
    path = book.write()
    text = path.read_text()
    assert 'PROVISIONAL' in text
    names = C.macro_names(path)
    assert names == sorted(names)
    assert len(names) == len(book)


def test_macro_collisions_are_reported_before_the_latex_build(data, paper_dirs):
    book = S.macros(data)
    path = book.write()
    assert S.macro_collisions(path) == {}
    other = C.NUM_DIR / 'numbers_v2_main.tex'
    name = sorted(book.values)[0]
    other.write_text(f'\\newcommand{{\\{name}}}{{1.0}}\n')
    assert S.macro_collisions(path) == {other.name: [name]}


# ---------------------------------------------------------------------------
# Tables (T8, T9)
# ---------------------------------------------------------------------------
BARE_PERCENT = re.compile(r'(?<!\\)%(?=.)')


def _assert_compile_safe(path):
    text = path.read_text()
    assert text.count(r'\begin{tabular}') == text.count(r'\end{tabular}') == 1
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith('%'):
            continue
        assert not BARE_PERCENT.search(line), f'{path.name}:{i} bare % -> {line!r}'
        assert line.count('$') % 2 == 0, f'{path.name}:{i} unbalanced $ -> {line!r}'
        stripped = re.sub(r'\$[^$]*\$', '', line)
        stripped = re.sub(r'\\[A-Za-z]+', '', stripped)
        assert not re.search(r'(?<!\\)_', stripped), f'{path.name}:{i} bare _'
        assert not re.search(r'\b(nan|NaN|inf|None)\b', line), f'{path.name}:{i} NaN'
    return text


@pytest.mark.parametrize('builder', [S.table_mechanism, S.table_mechanism_main,
                                     S.table_probe_energy, S.table_low4,
                                     S.table_probe_widths])
def test_every_table_is_compile_safe_and_has_one_row_per_record(builder, data,
                                                                paper_dirs):
    path, rows = builder(data)
    assert path is not None and rows
    text = _assert_compile_safe(path)
    body = text.split(r'\midrule', 1)[1]
    assert body.count(r'\\') >= len(rows)


def test_the_mechanism_table_keeps_the_two_measurements_apart(data, paper_dirs):
    path, rows = S.table_mechanism(data)
    by_scan = {r['Scan']: r for r in rows}
    # CIFAR-CE has no probe cache: the probe column must be a dash, never the online value
    assert str(by_scan[S.SCAN_SHORT[SCAN_B]]['Probe top-$k$']) == '--'
    assert '61.82' in str(by_scan[S.SCAN_SHORT[SCAN_A]]['Probe top-$k$'])   # 0.5882+0.03


def test_the_mechanism_table_prints_finished_over_attempted(data, paper_dirs):
    _path, rows = S.table_mechanism(data)
    counts = {r['Scan']: str(r['Seeds']) for r in rows}
    assert counts[S.SCAN_SHORT[SCAN_A]] == '3/3'
    # the re-selected scan rests on fewer seeds than the pass intends, and says so
    assert counts[S.SCAN_SHORT[SCAN_B]] == '2/3'


def test_the_low4_table_reports_a_tie_as_a_tie(data, paper_dirs):
    _path, rows = S.table_low4(data)
    assert rows
    for r in rows:
        assert 'Verdict' in r and r['Verdict']
        if 'not resolved' in r['Verdict']:
            assert 'smallest' in r['Verdict'] or 'larger' in r['Verdict']
    # every row carries both the rank and the paired interval; a rank alone is not a win
    assert all(str(r['Rank']).endswith('of 4') for r in rows)
    assert all('$[' in str(r['95% interval']) for r in rows)


def test_a_table_writes_its_provenance_outside_the_manuscript(data, paper_dirs):
    path, _rows = S.table_mechanism(data)
    assert C.MANUSCRIPT in path.parents
    side = list((paper_dirs / 'lab').rglob('*.provenance.json'))
    assert side, 'a generated file must carry a provenance sidecar'
    assert not list(C.MANUSCRIPT.rglob('*.provenance.json'))
    import json
    rec = json.loads(side[0].read_text())
    assert rec['functions'] and rec['selection_sha256_16'] is not None
    assert rec['provisional'] == [SCAN_B]


# ---------------------------------------------------------------------------
# The figure registry: draw / save split, knobs (campaign/FIGURE_API_CONTRACT.md)
# ---------------------------------------------------------------------------
MNIST_LR = 'mnist_scan_labelRegression'
MNIST_CE = 'mnist_scan_ce'


def _grid_frame(d, ks=(8, 16, 32, 128), rtols=(0.3, 0.03, 1e-4)):
    """A :func:`spectra_figs.used_rank_grid`-shaped frame: one row per ``(k, rtol)`` cell.

    The selected cell must be IN the grid or the heatmap draws no red square, so the
    fixture's ``k`` / ``rtol`` values are the ones the synthetic diags carry.
    """
    rows = []
    for k in ks:
        for rtol in rtols:
            rows.append({'k': k, 'rtol': rtol, 'lr': d.lr,
                         'used': min(float(k), 8.0),
                         'used_frac_k': min(1.0, 8.0 / k),
                         # one cell rests on fewer seeds than the grid intended: that is
                         # the hatched-cell path of _grid_heatmap
                         'n_seeds': 3 if rtol > 1e-4 else 1, 'n_records': 3})
    return pd.DataFrame(rows)


def _probe_entries(n_seeds=2, n_ck=5, width=24):
    """``ckpt_tools.load_spectra``-shaped trajectories, without touching the cache.

    The offline probe figures are the ones that read ``analysis/ckpt_spectra/``; a unit
    test must not, or it fails for a reason this module is not responsible for (and it
    would be reading a read-only results tree).
    """
    steps = np.array([0, 1, 4, 16, 64])[:n_ck]
    out = []
    for s in range(n_seeds):
        svals = np.geomspace(1.0, 1e-6, width)[None, :] * np.ones((len(steps), 1))
        utr = np.tile(np.geomspace(1.0, 1e-3, width), (len(steps), 1))
        out.append({'model_seed': 2000 + s, 'step': steps, 'svals': svals, 'utr': utr,
                    'method': 'Sven'})
    return out


@pytest.fixture
def probe_cache(monkeypatch):
    def load_spectra(scan, method=None, model_seed=None, n_probe=-1, epochs_only=None,
                     out_dir=None):
        ents = _probe_entries()
        if model_seed is not None:
            ents = [e for e in ents if str(e['model_seed']) == str(model_seed)]
        return ents
    monkeypatch.setattr(S.ct, 'load_spectra', load_spectra)
    return load_spectra


@pytest.fixture
def rich(data):
    """The synthetic :class:`Data` widened until all 13 figures have something to draw.

    ``data`` is enough for the tables; the figures also need the two MNIST scans (the
    LR-vs-CE panel), a ``(k, rtol)`` grid and a second scan with probe metrics.
    """
    lr = _diag(MNIST_LR, k=64, B=64, rtol=1e-4, lr=0.5, title='MNIST (label regression)')
    ce = _diag(MNIST_CE, k=32, B=64, rtol=0.03, lr=0.5, title='MNIST (cross-entropy)')
    data.diags[MNIST_LR] = lr
    data.diags[MNIST_CE] = ce
    data.arrays[MNIST_LR] = _arrays(lr, 3, 64.0, 64.0, 1.0, 1.0)
    data.arrays[MNIST_CE] = _arrays(ce, 3, 60.0, 21.0, 0.99, 0.72)
    data.seed_target.update({MNIST_LR: 3, MNIST_CE: 3})
    data.grids = {SCAN_A: _grid_frame(data.diags[SCAN_A]),
                  MNIST_CE: _grid_frame(ce)}
    data.probe_methods[MNIST_CE] = ['Sven', 'Adam', 'MuonW', 'HIG']
    data.probe[MNIST_CE] = _probe_table(MNIST_CE)
    return data


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An EMPTY override file: a test must see the builders' own defaults."""
    path = tmp_path / 'figure_overrides.yaml'
    path.write_text('{}\n')
    monkeypatch.setattr(F, 'OVERRIDES_PATH', path)
    monkeypatch.setattr(F, '_CACHE', {'mtime': None, 'data': {}})
    return path


@pytest.fixture
def registry(monkeypatch, rich):
    """Only THIS module's specs in the registry, plus its context.

    The four figure modules are converted independently; a test of this one must not go
    down with a half-written builder in another.
    """
    reg = {}
    for name, spec in S.FIGURE_SPECS.items():
        s = copy.copy(spec)
        s.name, s.module, s.group = name, 'spectra', S.GROUP
        reg[name] = s
    monkeypatch.setattr(F, '_REGISTRY', reg)
    monkeypatch.setattr(F, 'context', lambda module, root=None, reload=False: rich)
    return reg


def _no_writing(monkeypatch):
    """Make every write path of the module fail loudly, for the "draw writes nothing" tests."""
    def boom(*a, **k):
        raise AssertionError('a draw function wrote something')
    monkeypatch.setattr(C, 'save_fig', boom)
    monkeypatch.setattr(C, 'write_provenance', boom)
    monkeypatch.setattr(C, 'write_table', boom)


@pytest.mark.parametrize('name', list(S.FIGURE_SPECS))
def test_every_figure_draws_without_writing_and_returns_axes_and_provenance(
        name, rich, store, probe_cache, monkeypatch):
    """The split the notebook rests on: ``draw`` returns ``(fig, meta)`` and writes nothing.

    ``meta['axes']`` is what the notebook hands the user to edit and what
    ``figspec.apply_opts`` acts on; ``meta['provenance']`` is the record the figure is
    saved with, and ``figspec.save_figure`` refuses a figure without it.
    """
    import matplotlib.pyplot as plt
    _no_writing(monkeypatch)
    spec = S.FIGURE_SPECS[name]
    opts = F.figure_opts(name, spec.defaults)
    fig, meta = spec.draw(rich, opts)
    axes = F._axes_list(meta['axes'])
    assert axes and all(hasattr(ax, 'get_xlabel') for ax in axes)
    assert all(ax.figure is fig for ax in axes)
    rec = meta['provenance']
    assert rec['functions'] and rec['selection_sha256_16'] is not None
    assert rec['scan_dirs'], name
    plt.close(fig)


def test_the_generic_cosmetics_reach_every_figure_through_the_registry(registry, store):
    """One figure through ``figspec.draw_figure``: the knobs and the generic cosmetics."""
    import matplotlib.pyplot as plt
    F.set_overrides('online_mechanism', ylabel='Kept', legend={'ncol': 2})
    fig, meta, opts = F.draw_figure('online_mechanism')
    assert opts['smooth'] == S.SMOOTH_STEPS      # the builder's own default, untouched
    assert [ax.get_ylabel() for ax in meta['axes']] == ['Kept'] * 4
    leg = meta['axes'][3].get_legend()           # the per-seed panel's own key, restyled
    assert getattr(leg, '_ncols', getattr(leg, '_ncol', None)) == 2
    plt.close(fig)


@pytest.mark.parametrize('name,knob,expected', [
    ('online_spectra', {'scans': [SCAN_A]}, 1),
    ('online_spectra', {'scans': [SCAN_A, MNIST_CE]}, 2),
    ('probe_metrics', {'metrics': ['cond']}, 2),
])
def test_a_knob_visibly_changes_what_is_drawn(name, knob, expected, rich, store,
                                              probe_cache):
    """A pinned knob must reach the panels, not just the options dict."""
    import matplotlib.pyplot as plt
    spec = S.FIGURE_SPECS[name]
    fig, meta = spec.draw(rich, F.figure_opts(name, spec.defaults, **knob))
    assert len(F._axes_list(meta['axes'])) == expected
    plt.close(fig)


def test_the_geometry_knobs_change_the_figure_size(rich, store):
    import matplotlib.pyplot as plt
    spec = S.FIGURE_SPECS['spectrum_truncation']
    base = spec.draw(rich, F.figure_opts('spectrum_truncation', spec.defaults))[0]
    wide = spec.draw(rich, F.figure_opts('spectrum_truncation', spec.defaults,
                                         fraction=0.49, aspect=0.5, extra_h=1.0))[0]
    assert wide.get_size_inches()[0] > base.get_size_inches()[0]
    assert wide.get_size_inches()[1] > base.get_size_inches()[1]
    plt.close(base)
    plt.close(wide)


@pytest.mark.parametrize('name', list(S.PROBE_FIGURES))
def test_a_probe_figure_without_a_cache_says_so(name, data, store):
    """``build()`` skips these; a notebook asking for one deserves a sentence, not a
    ``KeyError`` from inside a helper three frames down."""
    data.probe_methods, data.probe = {}, {}
    spec = S.FIGURE_SPECS[name]
    with pytest.raises(RuntimeError, match='no cached probe spectra'):
        spec.draw(data, F.figure_opts(name, spec.defaults))


def test_every_spec_declares_yaml_representable_knobs_and_shadows_none():
    """A knob whose value is a python object cannot be pinned from a notebook, and one
    that shadows a generic cosmetic would be applied twice."""
    F.check_defaults(S.FIGURE_SPECS)
    for name, spec in S.FIGURE_SPECS.items():
        assert spec.doc.strip() and '\n' not in spec.doc.strip().split('\n')[0]
        assert not set(spec.defaults) & set(F.COMMON_KEYS), name
        round_trip = yaml.safe_load(yaml.safe_dump(spec.defaults))
        assert round_trip == spec.defaults, name
        for key in ('ncol', 'fraction', 'aspect', 'extra_h', 'sharex', 'sharey'):
            assert key in spec.defaults, (name, key)


def test_the_figure_names_are_the_pdf_stems():
    """The registry key IS the output stem, so ``figures_iclr/spectra/<key>.pdf``."""
    assert list(S.FIGURE_SPECS) == [
        'spectrum_truncation', 'online_spectra', 'online_utr', 'online_mechanism',
        'online_norms', 'mnist_lr_vs_ce', 'used_rank_grid', 'probe_spectra',
        'probe_spectra_all', 'probe_metrics', 'probe_metrics_all', 'probe_norms',
        'probe_energy']
    assert set(S.PROBE_FIGURES) <= set(S.FIGURE_SPECS)


def test_context_reads_the_passes_once_per_root(monkeypatch):
    calls = []
    monkeypatch.setattr(S, '_CONTEXT', {})
    monkeypatch.setattr(S, 'load',
                        lambda root=None, verbose=True: calls.append(root) or 'DATA')
    assert S.context() == 'DATA'
    assert S.context() == 'DATA'
    assert calls == [None]                      # expensive: read once, then cached
    S.context(reload=True)
    S.context(root='/other/root')
    assert calls == [None, None, '/other/root']


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------
def test_the_module_declares_its_assets():
    assert len(S.FIGURE_SPECS) == 13 and S.TABLES
    for name, spec in S.FIGURE_SPECS.items():
        assert callable(spec.draw) and spec.draw.__doc__, name
    for fn in S.TABLES:
        assert callable(fn) and fn.__doc__
    assert S.GROUP == 'spectra'


def test_the_dry_run_names_the_files_the_build_actually_writes():
    """It used to name the BUILDERS (``fig_online_utr.pdf``) while the build wrote
    ``online_utr.pdf``: thirteen file names that never existed."""
    report = S.build(dry_run=True, verbose=False)
    assert report['figures'] == [f'{name}.pdf' for name in S.FIGURE_SPECS]
    assert not any(n.startswith('fig_') for n in report['figures'])
    assert report['tables'] == [f'{fn.__name__}.tex' for fn in S.TABLES]
    assert report['status'] == 'dry run'


def test_build_writes_every_figure_once_through_save_fig(rich, store, probe_cache,
                                                         paper_dirs):
    report = S.build(data=rich, figures=True, tables=False, numbers=False,
                     verbose=False)
    written = sorted(p.name for p in (C.FIG_DIR / 'spectra').glob('*.pdf'))
    assert written == sorted(f'{name}.pdf' for name in S.FIGURE_SPECS)
    assert len(report['figures']) == len(S.FIGURE_SPECS)
    # every figure carries a provenance sidecar, and none of them lands in the manuscript
    side = sorted(p.name for p in (paper_dirs / 'lab').rglob('*.pdf.provenance.json'))
    assert side == sorted(f'{name}.pdf.provenance.json' for name in S.FIGURE_SPECS)
    assert not list(C.MANUSCRIPT.rglob('*.provenance.json'))


def test_build_skips_the_probe_figures_when_nothing_is_cached(rich, store, paper_dirs):
    rich.probe_methods, rich.probe = {}, {}
    report = S.build(data=rich, figures=True, tables=False, numbers=False, verbose=False)
    online = [n for n in S.FIGURE_SPECS if n not in S.PROBE_FIGURES]
    assert len(report['figures']) == len(online)
    assert sorted(p.stem for p in (C.FIG_DIR / 'spectra').glob('*.pdf')) == sorted(online)
    assert [n for n in report['notes'] if 'no cached probe spectra' in n]


def test_build_refuses_a_provisional_scan_under_strict_mode(data, monkeypatch):
    monkeypatch.setenv('PAPER_ASSETS_STRICT', '1')
    with pytest.raises(RuntimeError, match='PAPER_ASSETS_STRICT'):
        S.build(data=data, figures=False, tables=False, numbers=False, verbose=False)


def test_build_writes_the_declared_tables_and_macros(data, paper_dirs):
    report = S.build(data=data, figures=False, verbose=False)
    assert len(report['tables']) == len(S.TABLES)
    assert report['macros'] > 30
    assert report['provisional'] == [SCAN_B]
    assert (C.NUM_DIR / 'numbers_v2_spectra.tex').is_file()


# ---------------------------------------------------------------------------
# Integration: only where the campaign's data is actually present
# ---------------------------------------------------------------------------
requires_results = pytest.mark.skipif(
    not (C.SELECTION_PATH.is_file() and Path(C.results_root()).is_dir()),
    reason='needs bench/best_configs.json and the results root')


@requires_results
def test_the_rank_law_holds_on_every_logged_step():
    """``min(k, rtol-rank) == num_nonzero_svs`` -- the check that says the saved spectrum
    really is the one the step used.  This is claim C9 and it is cheap to verify."""
    d = S.load(verbose=False)
    if not d.diags:
        pytest.skip('no diag pass on disk')
    assert d.rank_law_ok, 'the saved spectra disagree with the optimizer own rank count'
    assert d.rank_law_steps > 1000


@requires_results
def test_the_generated_macro_file_shares_no_name_with_another_module():
    """``numbers_v2.tex`` ``\\input``s all four module files, so a name defined twice is a
    LaTeX error rather than a silent overwrite.  ``\\numCifar{LR,CE}SvenK`` belongs to
    ``large`` (App. J); this module's rank cap is ``\\num<Scan>SvenKCap``."""
    path = C.NUM_DIR / 'numbers_v2_spectra.tex'
    if not path.is_file():
        pytest.skip('numbers_v2_spectra.tex not generated yet')
    assert S.macro_collisions(path) == {}


@requires_results
def test_the_generated_spectra_macros_are_finite_and_traceable():
    path = C.NUM_DIR / 'numbers_v2_spectra.tex'
    if not path.is_file():
        pytest.skip('numbers_v2_spectra.tex not generated yet '
                    '(cd analysis && python -m paper_assets.spectra)')
    for line in path.read_text().splitlines():
        if not line.startswith('\\newcommand'):
            continue
        value = line.split('}{', 1)[1].rsplit('}', 1)[0]
        assert value.strip() not in ('', '--', 'nan', 'None')
        assert '%' in line, f'a macro without a source comment: {line!r}'
