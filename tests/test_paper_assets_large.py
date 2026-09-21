"""CPU tests for ``analysis/paper_assets/large.py`` -- the CIFAR / Fig-5 / transformer /
profile assets of the ICLR revision.

Three kinds of check, and the split is deliberate:

* **pure functions on synthetic frames** -- the macro-name speller
  (:func:`paper_assets.large.words`, :func:`frac_token`), the Fig-5 collapse threshold
  and its cost summary, and the LaTeX-text guard.  Every expected value is written down
  by hand, so these pin the ARITHMETIC and the DIRECTION of a claim (masking makes the
  step slower, not faster) rather than the plumbing;
* **the table emitters on a stub build input** -- a hand-built object with just the
  attributes a table reads, so a table can be rendered without the 24,000-record results
  root.  What is asserted is what would break the paper: a ``nan`` / ``None`` reaching a
  cell, an unescaped ``_`` or ``%`` (the two ways a generated table silently fails to
  compile), and the ``f/a`` column being present on every results row;
* **the generated artefacts, when they exist** -- the real
  ``iclr_manuscript/numbers_v2_large.tex`` is re-read and validated (letters-only macro
  names, no duplicates, no non-finite body).  Skipped in a fresh clone, which is why the
  synthetic checks above exist as well.

Nothing here touches ``experiment_results/`` or a profile root.
"""

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))

from paper_assets import common as C          # noqa: E402
from paper_assets import large                # noqa: E402

MACRO_FILE = C.NUM_DIR / 'numbers_v2_large.tex'
TABLE_DIR = C.TABLE_DIR


# ---------------------------------------------------------------------------
# Macro-name spelling (TeX forbids digits in a command name)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value, expected', [
    (0, 'Zero'), (1, 'One'), (5, 'Five'), (13, 'Thirteen'), (20, 'Twenty'),
    (25, 'TwentyFive'), (50, 'Fifty'), (99, 'NinetyNine'), (100, 'Hundred'),
])
def test_words(value, expected):
    assert large.words(value) == expected


@pytest.mark.parametrize('frac, expected', [
    (0.05, 'PctFive'), (0.1, 'PctTen'), (0.25, 'PctTwentyFive'), (0.5, 'PctFifty'),
    (1.0, 'PctHundred'), (0.75, 'PctSeventyFive'),
])
def test_frac_token(frac, expected):
    assert large.frac_token(frac) == expected


@pytest.mark.parametrize('value', [0, 1, 7, 12, 40, 64, 100, 128, 256])
def test_words_are_letters_only(value):
    """Every fragment a macro name can be built from has to be letters only."""
    assert re.fullmatch(r'[A-Za-z]+', large.words(value))
    assert re.fullmatch(r'[A-Za-z]+', large.frac_token(value / 256.0))


def test_macro_names_of_this_module_are_valid():
    """The names the module composes are accepted by the macro writer."""
    book = C.Macros(module='large-test')
    for scan in ('CifarLR', 'CifarCE', 'FigFive', 'Nanogpt', 'GptTwo'):
        book.add(f'num{scan}SvenVal', 1.25, source='test')
    for arch in large.ARCH_KEY.values():
        for backend in large.SVEN_BACKENDS.values():
            book.add(f'numProf{arch}{backend}MsStep', 1.0, source='test')
    assert len(book) == 5 + len(large.ARCH_KEY) * len(large.SVEN_BACKENDS)
    text = C.write_macros(Path(_tmp('names.tex')), book)
    for line in Path(text).read_text().splitlines():
        if line.startswith('\\newcommand'):
            assert re.match(r'^\\newcommand\{\\num[A-Za-z]+\}\{', line), line


_TMP = {}


def _tmp(name, tmp_path_factory=None):
    """A scratch path outside the manuscript repo (never write there from a test)."""
    import tempfile
    root = _TMP.setdefault('dir', Path(tempfile.mkdtemp(prefix='paper_assets_large_')))
    return root / name


# ---------------------------------------------------------------------------
# LaTeX text
# ---------------------------------------------------------------------------
def test_text_escapes_an_underscore_and_keeps_math():
    """``profile_results_v3`` in text mode is a hard LaTeX error; a method label is math."""
    assert large._text('profile_results_v3') == r'profile\_results\_v3'
    C.check_raw(large._text('profile_results_v3'), 'test')
    kept = large._text('Sven (Gram, full $J$)')
    assert kept == 'Sven (Gram, full $J$)'
    C.check_raw(kept, 'test')


def test_text_result_is_accepted_as_a_macro_body():
    book = C.Macros(module='large-test')
    book.add('numProfRoot', large._text('profile_results_v3'), source='test')
    assert book.values['numProfRoot'] == r'profile\_results\_v3'


# ---------------------------------------------------------------------------
# Fig 5: the collapse threshold and the cost summary
# ---------------------------------------------------------------------------
def paramfrac_frame(acc, ms_per_step=None, peak=None, finished=None):
    """A synthetic :func:`large_figs.paramfrac_table` result over five fractions."""
    fracs = [0.05, 0.1, 0.25, 0.5, 1.0]
    n = len(fracs)
    out = pd.DataFrame({
        'param_fraction': fracs, 'actual': fracs,
        'val': [1.0] * n, 'val_std': [0.0] * n, 'val_min': [1.0] * n,
        'test': [1.0] * n, 'test_std': [0.0] * n,
        'val_acc': acc, 'val_acc_std': [0.0] * n,
        'test_acc': acc, 'test_acc_std': [0.0] * n,
        'peak_mem_mb': peak if peak is not None else [100, 120, 140, 160, 200],
        'peak_mem_mb_std': [0.0] * n,
        'ms_per_step': (ms_per_step if ms_per_step is not None
                        else [340, 350, 360, 390, 180]),
        'ms_per_step_std': [0.0] * n,
        'finished': finished if finished is not None else [3] * n,
        'attempted': [3] * n, 'n_diverged': [0] * n, 'n_failed': [0] * n,
        'mask_mode': ['elementwise'] * 4 + ['none'],
    })
    ref = out[np.isclose(out['actual'], 1.0)].iloc[0]
    out['ms_per_step_vs_full'] = out['ms_per_step'] / ref['ms_per_step']
    out['peak_mem_mb_vs_full'] = out['peak_mem_mb'] / ref['peak_mem_mb']
    return out


def test_paramfrac_threshold_finds_the_collapse():
    """Quality holds to f = 0.25 and is gone at 0.1 -- the state the re-run produced."""
    tbl = paramfrac_frame([0.20, 0.27, 0.69, 0.687, 0.684])
    hold, collapse = large.paramfrac_threshold(tbl)
    assert hold == pytest.approx(0.25)
    assert collapse == pytest.approx(0.1)


def test_paramfrac_threshold_when_nothing_collapses():
    """Every fraction within 10% of f = 1: the hold point is the smallest fraction run
    and there is no collapse to report."""
    tbl = paramfrac_frame([0.66, 0.67, 0.68, 0.69, 0.70])
    hold, collapse = large.paramfrac_threshold(tbl)
    assert hold == pytest.approx(0.05)
    assert not np.isfinite(collapse)


def test_paramfrac_threshold_without_a_reference_point():
    """No f = 1 run yet -> no threshold, rather than a threshold against a masked run."""
    tbl = paramfrac_frame([0.2, 0.3, 0.69, 0.68, 0.68]).iloc[:4]
    hold, collapse = large.paramfrac_threshold(tbl)
    assert not np.isfinite(hold) and not np.isfinite(collapse)


def test_paramfrac_cost_summary_says_masking_is_slower():
    """The direction of Fig 5's cost finding: a masked step is DEARER than the full one,
    and the only saving is memory."""
    tbl = paramfrac_frame([0.2, 0.3, 0.69, 0.68, 0.68])
    cost = large.paramfrac_cost_summary(tbl)
    assert cost['full_ms'] == pytest.approx(180)
    assert cost['slow_min'] == pytest.approx(340 / 180)
    assert cost['slow_max'] == pytest.approx(390 / 180)
    assert cost['slow_min'] > 1.0
    assert cost['mem_frac_min'] == pytest.approx(0.5)


def test_paramfrac_cost_summary_on_an_empty_table():
    cost = large.paramfrac_cost_summary(pd.DataFrame())
    assert set(cost) and all(not np.isfinite(v) for v in cost.values())


# ---------------------------------------------------------------------------
# The table emitters: a stub input, and the LaTeX the emitter produces
# ---------------------------------------------------------------------------
def losses(final, acc, n_epochs=3):
    """A ``losses`` dict in the record schema: ``val`` / ``test`` carry the untrained
    entry at index 0, so they are one longer than ``train``."""
    return {
        'train': [2.0, 1.5, final],
        'val': [3.0, 2.0, 1.5, final], 'test': [3.1, 2.1, 1.6, final + 0.01],
        'val_acc': [0.1, 0.3, 0.5, acc], 'test_acc': [0.1, 0.3, 0.5, acc - 0.005],
        'train_eval': [3.0, 2.0, 1.4, final - 0.1],
        'train_times': [[0.5] * 4] * n_epochs, 'total_time': 10.0,
        'peak_gpu_mem_mb': 100.0,
    }


def runs_frame(method='Sven', n_seeds=5, final=0.5, acc=0.7):
    return pd.DataFrame([
        {'run_id': f'{method}-{s}', 'method': method, 'model_seed': 4100 + s,
         'losses': losses(final + 0.01 * s, acc), 'diverged': False, 'failed': False,
         'status': 'ok', 'final_val_loss': final + 0.01 * s}
        for s in range(n_seeds)])


def conf_frame(methods=('Sven', 'MuonW')):
    rows = []
    for i, m in enumerate(methods):
        rows.append({
            'method': m, 'display': m, 'config': 'k=128, lr=0.5, rtol=0.001',
            'val_conf': 0.4 + 0.1 * i, 'val_conf_std': 0.01,
            'test_conf': 0.41 + 0.1 * i, 'test_conf_std': 0.01,
            'acc_conf': 0.7 - 0.05 * i, 'acc_conf_std': 0.004,
            'treval_conf': 0.06 + 0.01 * i, 'treval_conf_std': 0.002,
            'fin_conf': 5, 'att_conf': 5, 'elig_conf': True,
        })
    return pd.DataFrame(rows)


def eff_frame(methods=('Sven', 'MuonW')):
    return pd.DataFrame([
        {'method': m, 'display': m, 'config': 'k=128', 'epoch_s': 60.0 - 50 * i,
         'epoch_s_std': 0.5, 'ms_per_step': 170.0 - 150 * i,
         'peak_gpu_mem_mb': 22000.0 - 21000 * i, 'peak_gpu_mem_mb_std': 0.0,
         'wall_s': 1200.0 - 1000 * i, 'fin_timing': 5, 'att_timing': 5,
         'gpu_timing': 'A100'}
        for i, m in enumerate(methods)])


def profile_frame():
    """A tiny :func:`profile_helpers.load_profiles` frame: one architecture, the four
    Sven backends and two baselines, plus a two-point ``k`` sweep."""
    rows = []
    for method, step, capture, solve, mem, family in (
            ('gram_hooks', 10.0, 4.0, 6.0, 20.0, 'sven'),
            ('gram_full', 15.0, 7.0, 8.0, 48.0, 'sven'),
            ('gram_chunked', 150.0, 142.0, 8.0, 45.0, 'sven'),
            ('classic', 11.0, 7.5, 3.5, 48.0, 'sven'),
            ('Adam', 4.0, np.nan, np.nan, 20.0, 'baseline'),
            ('SGD', 3.8, np.nan, np.nan, 19.0, 'baseline')):
        rows.append({'arch': 'mnist', 'study': 'methods', 'method': method,
                     'family': family, 'status': 'ok', 'step_ms': step,
                     'capture_ms': capture, 'solve_ms': solve, 'peak_mb': mem,
                     'rel_time': step / 4.0, 'rel_mem': mem / 19.0,
                     'steadiness': 1.1, 'k': 64, 'B': 64, 'width': None,
                     'n_params': 27562, 'pf': 1.0, 'mb': 1, 'mask_mode': 'none',
                     'chunk_fraction': np.nan, 'n_groups': np.nan, 'error': None})
    for k, step in ((4, 10.1), (64, 10.2)):
        rows.append({'arch': 'mnist', 'study': 'k', 'method': 'gram_hooks',
                     'family': 'sven', 'status': 'ok', 'step_ms': step,
                     'capture_ms': 4.0, 'solve_ms': 6.0, 'peak_mb': 20.0,
                     'rel_time': step / 4.0, 'rel_mem': 20 / 19.0, 'steadiness': 1.1,
                     'k': k, 'B': 64, 'width': None, 'n_params': 27562, 'pf': 1.0,
                     'mb': 1, 'mask_mode': 'none', 'chunk_fraction': np.nan,
                     'n_groups': np.nan, 'error': None})
    return pd.DataFrame(rows)


class StubInputs:
    """Just enough of :class:`paper_assets.large.Inputs` to render the tables."""

    def __init__(self):
        methods = ('Sven', 'MuonW')
        self.results_root = None
        # a selection payload shaped like the real one, so the lr-sensitivity table's
        # "selected" star and the figure's ring are exercised (record-level key 'SVD')
        self.payload = {'scans': {large.NANOGPT: {'methods': {
            'SVD': {'method': 'SVD', 'hparams': {'lr': 0.1, 'k': 64, 'rtol': 0.001}},
            'MuonW': {'method': 'MuonW', 'hparams': {'lr': 0.0001}}}}}}
        self.nanogpt_lr = pd.DataFrame([
            {'method': m, 'display': m, 'lr': lr, 'final_val_loss': v,
             'final_val_loss_std': 0.019, 'final_test_loss': v + 0.25,
             'final_train_eval': v - 0.3, 'val_ppl': float(np.exp(v)),
             'rank_eff': 64.0 if m == 'Sven' else np.nan, 'epoch_s': 6.7,
             'peak_gpu_mem_mb': 615.4, 'finished': 5, 'attempted': 5,
             'counts': '5/5', 'config': f'lr={lr:g}'}
            for m, lr, v in (('Sven', 0.05, 1.818), ('Sven', 0.1, 1.724),
                             ('Sven', 1.0, 4.665), ('MuonW', 0.0001, 1.818),
                             ('MuonW', 0.003, 2.019))])
        self.notes = []
        self.provisional = ['stub: nothing is real here']
        self.conf = {s: conf_frame(methods) for s in (*large.CIFAR, large.NANOGPT)}
        self.eff = {s: eff_frame(methods) for s in (*large.CIFAR, large.NANOGPT)}
        self.runs_conf = {s: {m: runs_frame(m) for m in methods}
                          for s in (*large.CIFAR, large.NANOGPT)}
        self.paired_nanogpt = pd.DataFrame([
            {'method': 'MuonW', 'display': 'MuonW', 'mean': -0.095, 'ci_low': -0.11,
             'ci_high': -0.085, 'n': 5, 'sven_better': 5}])
        self.profiles = profile_frame()
        self.profile_archs = ['mnist']
        self.profile_widths = []
        self.profile_compare = pd.DataFrame()
        self.profile_info = {'name': 'profile_results_v3', 'root': '/dev/null',
                             'complete': False, 'n_found': 700, 'n_expected': 720,
                             'per_config': {}, 'has_cifar': False, 'provisional': True}
        table = paramfrac_frame([0.2, 0.27, 0.69, 0.687, 0.684])
        table.attrs['mask_note'] = large.lf.MASK_NOTE
        self.fig5_groups = [{
            'key': (128, 0.5, 0.001), 'df': None, 'table': table, 'is_selected': True,
            'role': 'selected', 'diff': [], 'headline': {'label': 'k=128, lr=0.5'},
            'cfg': {'label': 'k=128, lr=0.5, rtol=0.001', 'n_seeds': 3,
                    'fractions': list(table['param_fraction']), 'num_epochs': 20,
                    'loss': 'label_regression', 'kappa': 2, 'loss_label': 'label reg.',
                    'varies': [], 'seeds': [4000, 4001, 4002], 'n_runs': 15,
                    'mask_note': large.lf.MASK_NOTE, 'fixed_config_note': 'note'},
        }]
        self.fig5_selected = self.fig5_groups[0]
        self.fig5_target_seeds = 3
        self.fig5_incomplete = False
        self.gpt2 = pd.DataFrame([
            {'run_id': 'gpt2-sven', 'method': 'Sven', 'display': 'Sven', 'lr': 0.1,
             'k': 16, 'val': 5.198, 'test': 5.107, 'wall_h': 9.25,
             'ms_per_step': 2517.0, 'peak_gpu_mem_mb': 36050.0, 'diverged': False,
             'failed': False, 'status': 'ok', 'n_params': 163109376,
             'steps_per_epoch': 13125, 'batch_size': 16, 'eval_every_steps': 500,
             'losses': {'eval_step_idx': list(range(500, 13500, 500))}}])
        self.gpt2_best = pd.DataFrame([
            {'method': 'Sven', 'display': 'Sven', 'lr': 0.1, 'k': 16, 'val': 5.198,
             'test': 5.107, 'val_ppl': 181.0, 'wall_h': 9.25, 'ms_per_step': 2517.0,
             'peak_gpu_mem_mb': 36050.0, 'run_id': 'gpt2-sven', 'n_configs': 5,
             'n_diverged': 0, 'finished': 5, 'attempted': 5}])
        self.gpt2_lr = pd.DataFrame([
            {'method': 'Sven', 'display': 'Sven', 'lr': lr, 'k': 16, 'val': v,
             'test': v - 0.09, 'wall_h': 9.25, 'peak_gpu_mem_mb': 36050.0,
             'ms_per_step': 2517.0, 'diverged': False, 'failed': False, 'status': 'ok'}
            for lr, v in ((0.02, 6.32), (0.1, 5.198), (1.0, 10.2))])

    # the two methods the tables call
    def scan_provisional(self, scan):
        return scan == 'cifar10_resnet_ce_scan'

    def profile_note(self):
        return 'profile root profile\\_results\\_v3; 700/720 configurations, PARTIAL'

    def timing_cost(self, scan, method='Sven'):
        return {'epoch_s': 60.0, 'ms_per_step': 170.0, 'peak_mb': 22000.0,
                'wall_s': 1200.0, 'fin': 5, 'att': 5}


#: the tables that must render on ANY input.  ``profile_scaling`` and ``profile_v2_v3``
#: are legitimately empty when the profile root has no width sweep / no second root, and
#: ``build`` skips an empty table rather than writing a headers-only file.
TABLE_FUNCTIONS = ('cifar', 'fig5', 'transformers', 'transformers_nanogpt_lr',
                   'transformers_gpt2', 'transformers_gpt2_lr', 'profile_methods',
                   'gram_cost')


@pytest.fixture(scope='module')
def stub():
    return StubInputs()


@pytest.fixture(scope='module')
def rendered(stub):
    """``{name: (rows, latex)}`` for every table this module owns, on the stub."""
    out = {}
    for name, fn in large.TABLES:
        rows, meta = fn(stub)
        if not rows:
            continue
        out[name] = (rows, C.booktabs(rows, notes=meta.get('notes', ()),
                                      midrules=meta.get('midrules', ()),
                                      caption=large.TABLE_CAPTIONS.get(name),
                                      label=f'tab:{name}'))
    return out


def test_every_table_renders(rendered):
    """Each declared table produces rows on a stub input (an empty one is a bug, not a
    state: the ones that legitimately can be empty are skipped by ``build``)."""
    assert set(TABLE_FUNCTIONS) <= set(rendered), sorted(rendered)


def test_no_nan_or_none_reaches_a_cell(rendered):
    """A missing number prints as ``--``; the strings ``nan`` / ``None`` never do."""
    for name, (rows, _latex) in rendered.items():
        for row in rows:
            for column, value in row.items():
                text = str(value).strip()
                # '' is legitimate: a group column is blank after its first row
                assert text.lower() not in ('nan', 'none', 'inf', '-inf'), \
                    f'{name}: {column} = {value!r}'
                assert 'nan' not in text.lower(), f'{name}: {column} = {value!r}'


def test_tables_are_compile_safe(rendered):
    """The two ways a generated table silently breaks the build: a bare ``_`` in text
    mode and a ``%`` that comments out the rest of the line (including its ``\\\\``)."""
    for name, (_rows, latex) in rendered.items():
        for line in latex.splitlines():
            if line.startswith('%'):
                continue                      # a comment is a comment
            body = re.sub(r'\$[^$]*\$', '', line)
            assert not re.search(r'(?<!\\)_', body), f'{name}: bare _ in {line!r}'
            assert not re.search(r'(?<!\\)%', body), f'{name}: bare % in {line!r}'
            assert line.count('$') % 2 == 0, f'{name}: unbalanced $ in {line!r}'


def test_tables_have_one_cell_per_column(rendered):
    """Every row of the emitted tabular has the same number of ``&`` separators as the
    header -- a ragged generated table does not compile."""
    for name, (_rows, latex) in rendered.items():
        body = [l for l in latex.splitlines()
                if l.endswith(r'\\') and not l.startswith('%')]
        widths = {l.count('&') for l in body}
        assert len(widths) == 1, f'{name}: ragged rows {widths}'


def test_results_tables_print_finished_over_attempted(rendered):
    """Every results row says how many runs are behind it (ANALYSIS_CONTRACTS)."""
    for name in ('cifar', 'fig5', 'transformers'):
        rows = rendered[name][0]
        assert any(k.startswith('fin') for k in rows[0]), f'{name}: no fin/att column'
        for row in rows:
            counts = [v for k, v in row.items() if k.startswith('fin')]
            assert counts and re.fullmatch(r'\d+/\d+', str(counts[0])), row


def test_cifar_table_carries_both_objectives_and_a_midrule(stub):
    rows, meta = large.table_cifar(stub)
    objectives = [r['Objective'] for r in rows if r['Objective']]
    assert len(objectives) == len(large.CIFAR)
    assert meta['midrules'], 'the two objectives must be separated by a midrule'
    assert any('PROVISIONAL' in n for n in meta['notes'])


def test_fig5_table_reports_the_actual_fraction_and_the_cost_ratio(stub):
    rows, meta = large.table_fig5(stub)
    assert [r['f'] for r in rows] == ['0.05', '0.1', '0.25', '0.5', '1']
    assert all('f actual' in r for r in rows)
    full = rows[-1]
    assert full['ms/step x full'] == '1'
    # masking is dearer per step than no mask at all: every masked row is above 1
    masked = [float(str(r['ms/step x full'])) for r in rows[:-1]]
    assert all(v > 1 for v in masked), masked
    assert any('resampled every step' in n for n in meta['notes'])


def test_gram_cost_table_carries_the_k_independence_check(stub):
    rows, meta = large.table_gram_cost(stub)
    hooks = next(r for r in rows if 'hooks' in r['Backend'])
    assert float(hooks['k spread']) == pytest.approx(10.2 / 10.1, rel=1e-3)
    assert hooks['Capture share'] == r'40\%'
    assert any('eigendecomposition' in n for n in meta['notes'])
    # T19 carried eleven columns and the \resizebox that made it fit put the body text
    # near 6 pt, so 'x SGD' and 'k range' were dropped -- both are in App. Q at full
    # size, and the table note says where.
    assert 'k range' not in hooks and 'x SGD' not in hooks
    assert any('tab:profile_methods' in n for n in meta['notes'])


def test_profile_methods_table_lists_sven_first(stub):
    rows, _meta = large.table_profile_methods(stub)
    assert 'Sven' in rows[0]['Method']
    assert rows[0]['Architecture'] == 'MNIST MLP'
    assert all(r['Architecture'] == '' for r in rows[1:])


def test_transformer_tables_keep_the_single_seed_framing(stub):
    _rows, meta = large.table_gpt2(stub)
    assert any('one model seed' in n for n in meta['notes'])
    rows, _ = large.table_nanogpt(stub)
    sven = next(r for r in rows if r['Method'] == 'Sven')
    assert sven['Sven - this'] == '--', 'Sven is not paired against itself'


def test_nanogpt_lr_table_stars_the_selection_and_blocks_by_method(stub):
    rows, meta = large.table_nanogpt_lr(stub)
    assert [r['Method'] for r in rows].count('Sven') == 1, 'one header per method block'
    assert rows[0]['Method'] == 'Sven', 'Sven leads (headline.method_order)'
    starred = [r for r in rows if r'\star' in r['lr']]
    assert len(starred) == 2, 'one star per method (the selection of record)'
    # the record-level key 'SVD' in the selection file must reach the 'Sven' rows
    assert starred[0]['lr'].startswith('0.1'), starred[0]['lr']
    assert any('selection of record' in n for n in meta['notes'])
    assert rows[0]['Rank used'] == '64', 'Sven reports the rank it inverted'
    assert rows[-1]['Rank used'] == '--', 'a first-order method has no rank'
    assert meta['midrules'] == [3], 'one rule between the two method blocks'


def test_selected_lrs_survives_a_scan_with_no_selection(stub):
    class NoSelection:
        payload = {'scans': {}}
    assert large.selected_lrs(NoSelection()) == {}
    assert large.selected_lrs(stub) == {'Sven': 0.1, 'MuonW': 0.0001}


def test_headroom_expands_a_log_axis_without_flipping_it():
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_ylim(10.0, 16.0)
    large._headroom(ax, 0.06)
    lo, hi = ax.get_ylim()
    assert lo < 10.0 < 16.0 < hi
    assert hi / 16.0 == pytest.approx(10.0 / lo, rel=1e-9), 'symmetric in log space'
    ax.set_yscale('linear')
    ax.set_ylim(10.0, 2.0)                   # inverted: nothing sensible to expand
    large._headroom(ax)
    assert ax.get_ylim() == (10.0, 2.0), 'a degenerate axis is left alone'
    plt.close(fig)


def test_profile_change_table_is_empty_without_two_roots(stub):
    rows, meta = large.table_profile_change(stub)
    assert rows == []
    assert any('only one profile root' in n for n in meta['notes'])


# ---------------------------------------------------------------------------
# Small helpers used by the figures
# ---------------------------------------------------------------------------
def test_methods_with_data_drops_a_backend_that_never_ran(stub):
    assert large._methods_with_data(stub.profiles, 'mnist', 'k',
                                   list(large.SVEN_BACKENDS)) == ['gram_hooks']
    assert large._has_study(stub.profiles, 'mnist', 'k')
    assert not large._has_study(stub.profiles, 'mnist', 'width')


def test_plot_curves_positive_drops_the_untrained_point():
    """A log time axis cannot show ``t = 0``, and leaving it in draws a horizontal
    lead-in from the left spine to the first real evaluation."""
    import matplotlib.pyplot as plt
    runs = {'Sven': runs_frame('Sven', n_seeds=3)}
    fig, ax = plt.subplots()
    drawn = large.plot_curves_positive(ax, runs, 'val', versus='time')
    plt.close(fig)
    assert drawn == {'Sven': 3}
    x = ax.lines[0].get_xdata()
    assert len(x) and min(x) > 0, x


def test_profile_root_prefers_v3_over_v2(tmp_path, monkeypatch):
    """The paper reads the corrected pass even while it is PARTIAL: a contaminated
    number is worse than a provisional one, and the stamp says which it is."""
    v2, v3 = tmp_path / 'profile_results_v2', tmp_path / 'profile_results_v3'
    for root in (v2, v3):
        (root / 'profile_mnist').mkdir(parents=True)
        (root / 'profile_mnist' / 'a.json').write_text('{}')
    monkeypatch.delenv('SV3_PROFILE_ROOT', raising=False)
    monkeypatch.setattr(large, 'PROFILE_PREFERENCE', (v3, v2))
    root, info = large.profile_root()
    assert Path(root) == v3
    assert info['provisional'] and not info['complete']
    assert info['per_config'] == {'profile_mnist': 1}
    assert not info['has_cifar']


def test_profile_root_falls_back_when_v3_is_empty(tmp_path, monkeypatch):
    v2, v3 = tmp_path / 'v2', tmp_path / 'v3'
    (v2 / 'profile_mnist').mkdir(parents=True)
    (v2 / 'profile_mnist' / 'a.json').write_text('{}')
    v3.mkdir()
    monkeypatch.delenv('SV3_PROFILE_ROOT', raising=False)
    monkeypatch.setattr(large, 'PROFILE_PREFERENCE', (v3, v2))
    root, _info = large.profile_root()
    assert Path(root) == v2


# ---------------------------------------------------------------------------
# The generated artefacts, when a build has run
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not MACRO_FILE.is_file(), reason='no build in this tree yet')
def test_generated_macros_are_valid_latex():
    names, bad = [], []
    for line in MACRO_FILE.read_text().splitlines():
        if not line.startswith('\\newcommand'):
            continue
        m = re.match(r'^\\newcommand\{\\(?P<name>[^}]+)\}\{(?P<body>.*)\}\s*(%.*)?$',
                     line)
        assert m, line
        name, body = m.group('name'), m.group('body')
        names.append(name)
        assert re.fullmatch(r'num[A-Za-z]+', name), name
        assert body.strip(), name
        if body.strip().lower() in ('nan', 'none', 'inf', '-inf', '--'):
            bad.append((name, body))
        # a bare % or a text-mode _ in a macro body breaks every page that uses it
        outside = re.sub(r'\$[^$]*\$', '', body)
        assert not re.search(r'(?<!\\)%', outside), (name, body)
        assert not re.search(r'(?<!\\)_', outside), (name, body)
        assert body.count('$') % 2 == 0, (name, body)
    assert names, 'the macro file has no macros'
    assert not bad, f'non-finite macro bodies: {bad}'
    assert len(names) == len(set(names)), 'duplicate macro name'


@pytest.mark.skipif(not MACRO_FILE.is_file(), reason='no build in this tree yet')
def test_generated_macros_have_a_provenance_header():
    head = MACRO_FILE.read_text().splitlines()[:6]
    assert any('GENERATED' in l for l in head)
    assert any('selection:' in l and 'sha256' in l for l in head)


@pytest.mark.parametrize('name', ['cifar', 'fig5', 'transformers', 'profile_methods',
                                  'gram_cost'])
def test_generated_tables_are_compile_safe(name):
    path = TABLE_DIR / f'{name}.tex'
    if not path.is_file():
        pytest.skip('no build in this tree yet')
    text = path.read_text()
    assert text.startswith('% GENERATED')
    assert r'\begin{tabular}' in text and r'\end{tabular}' in text
    for line in text.splitlines():
        if line.startswith('%'):
            continue
        body = re.sub(r'\$[^$]*\$', '', line)
        assert not re.search(r'(?<!\\)_', body), line
        assert not re.search(r'(?<!\\)%', body), line
        assert line.count('$') % 2 == 0, line
    rows = [l for l in text.splitlines() if l.endswith(r'\\') and not l.startswith('%')]
    assert len({l.count('&') for l in rows}) == 1, f'{name}: ragged rows'


@pytest.mark.parametrize('name', [n for n, _f in large.FIGURES])
def test_generated_figures_exist_as_pdf_and_png(name):
    pdf = C.FIG_DIR / large.GROUP / f'{name}.pdf'
    if not (C.FIG_DIR / large.GROUP).is_dir():
        pytest.skip('no build in this tree yet')
    png = C.LAB / large.GROUP / f'{name}.png'
    assert pdf.is_file(), pdf
    assert png.is_file(), f'{png} -- the PNG twin is what a reviewer looks at'
    assert pdf.stat().st_size > 2000
