"""What the robustness fixes changed: legacy results against the fresh campaign (WP5).

`CHANGES_NEEDED.md` section 4.2 phase 7 and section 6 ask for one thing in writing: *the
differences from the legacy tables*.  This module computes them.  The consumer is
``analysis/notebooks/legacy/legacy_vs_fresh.ipynb``; the written account is ``analysis/WHAT_CHANGED.md``.

What "the legacy tables" are
----------------------------
``experiment_results_legacy_2026-09-18/`` is the pre-campaign results root, read-only.  Its
validation losses average per-batch means rather than examples (F10), so every legacy number
here is the **example-weighted repair** from :mod:`repair_legacy`
(``analysis/legacy_repair/{scan}.parquet``, column ``final_val_loss_ew``), not the number the
run recorded.  Regenerate a missing parquet with::

    campaign/run_cpu_tests.sh .venv/bin/python analysis/lib/repair_legacy.py <scan> \\
        --root $(pwd)/experiment_results_legacy_2026-09-18 --n-val 10000

The legacy *selection* is the binding rule of `CHANGES_NEEDED.md` §1 applied to legacy data
(eligible -> fewest diverged -> seed mean), i.e. :meth:`scan_analysis.Scan.configs`, so the
two sides differ in the data, never in the rule.  Legacy scans carry no manifest, so
``attempted`` there is the seed count, not a grid the scan promised to run -- which is
itself one of the things that changed (C-R2).

What may and may not be compared
--------------------------------
* **Polynomial scans compare RANKS ONLY.**  The legacy target is an additive cubic (F1,
  C-D1): a different function, so its losses are not on the same scale as the fresh ones.
  :func:`ranks_only` marks those scans and :func:`diff_view` blanks their loss columns.
* Losses elsewhere are comparable only up to the split change: MNIST / CIFAR legacy runs
  validated on the official test set, the fresh ones on a held-out slice of train (C-E1),
  and toy / polynomial draw their validation set from a different generator (C-D3).  A
  small level shift is therefore expected everywhere; a rank change is the signal.
* Fresh headline numbers come from the selection of record (``bench/best_configs.json``)
  through :mod:`headline`, never re-derived here.  :func:`selection_agreement` proves this
  module's own rule reproduces that selection before any table is believed.
* A fresh rank is computed on the CONFIRMATION mean, except where that configuration is
  not eligible on the confirmation seeds (polynomial L-BFGS finishes 2 of 15), where the
  rule's fallback is the tuning-seed loss.  Such a rank is on a different number than the
  confirmation mean printed beside it, so ``diff_table``'s ``fresh_basis`` records it and
  :func:`diff_view` marks it (:func:`fallback_rows`, :data:`FALLBACK_NOTE`).
* Sven's ranks aggregate in :func:`rank_change_tally`, which NEVER pools the ranks-only
  polynomial points with the same-target ones: they are ranks on another function and are
  also Sven's largest gains, so a pooled mean is mostly a statement about the target.
* :func:`scans_in_both_roots` is an intersection, so :func:`scan_inventory` reports its
  complement -- what the legacy campaign ran and the fresh one has no counterpart for
  (:data:`LEGACY_ONLY_DISPOSITION`).

Attributed causes
-----------------
:data:`CAUSES` is the fixed vocabulary; :func:`causes_for` assigns them per (scan, method)
from facts, not from prose: ``newbase`` from the method's absence in the legacy root,
``grid`` from the per-method distinct-configuration counts on both sides (on the Sven-only
ablations too -- the kappa study's legacy grid is 3 configurations against 42), the rest from
the scan's properties.  The split-family cause is per dataset (:data:`_SPLIT_CAUSE`): MNIST
and CIFAR had an official test set the legacy runs selected on, the character corpus had no
test set at all and what changed there is that one was carved out.  Where a cause can be
tested from the data there is a function for it: :func:`adamw_duplicate_check` (F9),
:func:`failure_visibility` (F6), :func:`split_only_check`, :func:`nanogpt_split_check`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import headline as hl
import repair_legacy as RL
import scan_analysis as sa
import style

HERE = Path(__file__).resolve().parent.parent   # analysis/ (this module lives in analysis/lib/)
REPO = HERE.parent

#: the read-only legacy results root (`CHANGES_NEEDED.md` §4.1)
LEGACY_ROOT = str(REPO / 'experiment_results_legacy_2026-09-18')
#: slim-load cache for the legacy root, which analysis must not write into: the same
#: directory :mod:`repair_legacy` already uses (gitignored)
LEGACY_CACHE = str(HERE / 'legacy_repair' / '_cache')
#: figures of this notebook
PLOT_DIR = HERE / 'plots_v2' / 'legacy_vs_fresh'

#: the selection metric on both sides -- a column this module WRITES (legacy: the
#: example-weighted repair; fresh: the recorded final validation loss), so that one code
#: path ranks both roots.  Never a test quantity (:func:`style.assert_selection_metric`).
METRIC = 'val_sel'

# ---------------------------------------------------------------------------
# Which scans, and how each is compared
# ---------------------------------------------------------------------------
#: multi-method scans with a `_confirm` pass and an entry in the selection of record
HEADLINE_SCANS = hl.HEADLINE_SCANS

#: multi-method studies that sweep an axis: ranked WITHIN each value of that axis, because
#: the best configuration (and Sven's rank) is a function of it
GROUPED_SCANS = {
    'rebuttal_overparam_toy_1d_scan': 'n_data',
    'rebuttal_overparam_polynomial_scan': 'n_data',
    'rebuttal_overparam_mnist_scan': 'n_data',
    'rebuttal_batchsize_polynomial_scan': 'batch_size',
}

#: Sven-only ablations: no ranking is possible, so they contribute the best Sven
#: configuration and its loss on each side.  The value is the extra configuration axis the
#: selection must NOT pool over (added to :data:`scan_analysis.SVEN_CONFIG`).
SVEN_ONLY_SCANS = {
    'toy_1d_microbatch_scan': 'microbatch_size',
    'polynomial_microbatch_scan': 'microbatch_size',
    'mnist_microbatch_ce_scan': 'microbatch_size',
    'mnist_microbatch_labelreg_scan': 'microbatch_size',
    'toy_1d_paramfrac_scan': 'param_fraction',
    'polynomial_paramfrac_scan': 'param_fraction',
    'mnist_paramfrac_ce_scan': 'param_fraction',
    'mnist_paramfrac_labelreg_scan': 'param_fraction',
    'rebuttal_fig5_cifar_paramfrac_scan': 'param_fraction',
    'mnist_kappaScan_labelRegression': 'kappa',
}


def dataset_of(scan):
    """Which benchmark a scan directory is on: ``cifar`` | ``mnist`` | ``polynomial`` |
    ``toy`` | ``nanogpt`` | ``gpt2`` | ``other``.  Drives the scan-level causes."""
    s = scan.lower()
    for name in ('cifar', 'mnist', 'polynomial', 'nanogpt', 'gpt2'):
        if name in s:
            return name
    return 'toy' if 'toy_1d' in s else 'other'


def ranks_only(scan):
    """True where the legacy target is a DIFFERENT function and only ranks may be compared.

    Every polynomial scan: the legacy `RandomPolynomialDataset` adds its factors instead of
    multiplying them and omits the constant, linear and pure degree-d monomials, so it is an
    additive cubic (F1/C-D1, verified by a 19-feature fit to 8e-8).  A loss on it is not a
    loss on the fresh degree-4 target.
    """
    return dataset_of(scan) == 'polynomial'


#: scans present in both roots that this module deliberately does NOT diff, with the reason
EXCLUDED = {
    'exp_gpt2_small_comparison':
        'legacy GPT-2 trained on a corpus whose validation set is a prefix of training '
        '(F4/C-D2) and has 6 runs against 29: not a comparison, a quarantine',
    'exp_nanogpt_speedrun_timing': 'standalone timing pass, not a tuning grid',
    'mnist_scan_ce_timing': 'standalone timing pass, not a tuning grid',
    'mnist_scan_labelRegression_timing': 'standalone timing pass, not a tuning grid',
    'polynomial_scan_timing': 'standalone timing pass, not a tuning grid',
    'toy_1d_scan_timing': 'standalone timing pass, not a tuning grid',
}


def scans_in_both_roots(fresh_root=None):
    """Scan directories that exist under BOTH results roots, from disk."""
    fresh = Path(style.resolve_results_root(fresh_root))
    if not fresh.is_absolute():
        fresh = (HERE / fresh).resolve()
    legacy = Path(LEGACY_ROOT)

    def scans(root):
        return {d.name for d in root.iterdir()
                if d.is_dir() and not d.name.startswith('_') and any(d.glob('*.jsonl'))}

    return sorted(scans(fresh) & scans(legacy))


def scan_kind(scan):
    """``'headline'`` | ``'grouped'`` | ``'sven_only'`` | ``'excluded'`` | ``'unclassified'``."""
    if scan in HEADLINE_SCANS:
        return 'headline'
    if scan in GROUPED_SCANS:
        return 'grouped'
    if scan in SVEN_ONLY_SCANS:
        return 'sven_only'
    if scan in EXCLUDED:
        return 'excluded'
    return 'unclassified'


#: readable names for the scans :mod:`headline` has no title key for (it covers the seven
#: headline scans only); the headline spellings still come from `style.DATASET_TITLES`
EXTRA_TITLES = {
    'rebuttal_overparam_toy_1d_scan': 'Toy 1D, P/N sweep',
    'rebuttal_overparam_polynomial_scan': 'Polynomial, P/N sweep',
    'rebuttal_overparam_mnist_scan': 'MNIST (label reg.), P/N sweep',
    'rebuttal_batchsize_polynomial_scan': 'Polynomial, batch-size sweep',
    'rebuttal_fig5_cifar_paramfrac_scan': 'CIFAR-10 (label reg.), Fig-5 masks',
    'mnist_kappaScan_labelRegression': 'MNIST (label reg.), kappa',
    'toy_1d_microbatch_scan': 'Toy 1D, micro-batch',
    'polynomial_microbatch_scan': 'Polynomial, micro-batch',
    'mnist_microbatch_ce_scan': 'MNIST (CE), micro-batch',
    'mnist_microbatch_labelreg_scan': 'MNIST (label reg.), micro-batch',
    'toy_1d_paramfrac_scan': 'Toy 1D, param fraction',
    'polynomial_paramfrac_scan': 'Polynomial, param fraction',
    'mnist_paramfrac_ce_scan': 'MNIST (CE), param fraction',
    'mnist_paramfrac_labelreg_scan': 'MNIST (label reg.), param fraction',
    'exp_gpt2_small_comparison': 'GPT-2 small (FineWeb)',
}


def scan_title(scan):
    """One spelling for a scan (:func:`headline.scan_title`, then :data:`EXTRA_TITLES`)."""
    title = hl.scan_title(scan)
    return EXTRA_TITLES.get(scan, title) if title == scan else title


# ---------------------------------------------------------------------------
# The fixed cause vocabulary
# ---------------------------------------------------------------------------
#: code -> what it names.  A cause may be attributed to a (scan, method) pair only from this
#: table; :func:`causes_for` is the only place that assigns them.
CAUSES = {
    'split': 'test-set-as-validation removed (MNIST 50k/10k, CIFAR 45k/5k): the legacy runs '
             'selected on the official TEST set and reported the same number, the fresh ones '
             'select on a slice held out of train (C-E1)',
    'testsplit_carved': 'a test split was carved out of the legacy data and validation was '
                        're-cut: the character corpus was 90/10 by position and is now '
                        '80/10/10, so the legacy VALIDATION blocks are the fresh TEST split '
                        '(verified: legacy val[0] = fresh test[0]) and the fresh validation '
                        'set comes out of what legacy trained on. Legacy had no test set at '
                        'all, so it did select and report on one and the same held-out set; '
                        'and legacy training saw more data (nanoGPT n_train 7,842 -> 6,971 '
                        'blocks, n_val 871 on both sides)',
    'ew': 'example-weighted validation metric (legacy averaged per-batch means, F10/C-A1)',
    'bn': 'BatchNorm evaluation fixed (every optimizer now trains with batch statistics, '
          'updates the running statistics exactly once per step and evaluates with them): '
          'legacy Sven validated in TRAIN mode on validation-batch statistics and mutated '
          'the buffers, and repeated forwards inside one step (Sven capture, L-BFGS '
          'closures) applied the momentum twice (F2/F3, C-E2)',
    'target': 'true polynomial target (legacy = additive cubic, F1/C-D1): RANKS ONLY',
    'fixedval': 'synthetic data regenerated: a fixed 10,000-example training pool with '
                'validation and test from SEPARATE generators and the target normalised by '
                'the pool, so the validation set no longer moves with n_data (F22/C-D3)',
    'muon': 'Muon regrouping: hidden 2-D weights only, convolutions flattened, '
            'embeddings/head/1-D to AdamW, adjust_lr_fn=match_rms_adamw (F13/C-B5)',
    'adamw_wd': 'AdamW now decays (legacy AdamW was Adam with wd forced to 0, F9/C-B1)',
    'grid': 'extended grid / interior optimum (C-B3): the fresh scan searches more '
            'configurations for this method',
    'failures': 'failures are records (C-R1/C-R2): legacy exceptions were printed and '
                'dropped, so a half-crashed grid looked like a complete one',
    'droplast': 'drop_last=True for every optimizer and a per-seed data order derived from '
                'the model seed (C-S2/C-S3): same batches for all methods, seed bands now '
                'contain data-order variance',
    'newbase': 'new baseline in the fresh campaign (SGDm everywhere, MuonW/SOAP/AdamW/Muon '
               'on CIFAR, KFAC on MNIST): the field it is ranked against is larger',
}

#: methods that did not exist as baselines in the legacy campaign at all
_ALWAYS_NEW = ('SGDm',)

#: which split-family cause a dataset gets.  MNIST and CIFAR had an official test set that
#: the legacy runs used as their validation set; the character corpus had none, and what
#: changed there is that a test split was carved out (:data:`CAUSES` spells both out).
#: ``gpt2`` is deliberately absent: its legacy corpus had validation as a PREFIX of training
#: (F4/C-D2), which is why it is in :data:`EXCLUDED` and gets no comparison at all, so
#: attributing a split cause to it would describe a diff that is never computed.
_SPLIT_CAUSE = {'mnist': 'split', 'cifar': 'split', 'nanogpt': 'testsplit_carved'}


def cause_text(codes):
    """The codes spelled out, one per line (for a table footer)."""
    return '\n'.join(f'{c}: {CAUSES[c]}' for c in codes if c in CAUSES)


def causes_for(scan, method, in_legacy=True, grid_grew=False):
    """The attributed cause codes for one (scan, method), from :data:`CAUSES`.

    Facts, not prose: ``in_legacy`` comes from the legacy records, ``grid_grew`` from the
    per-method distinct-configuration counts (:func:`grid_counts`).  The scan-level causes
    follow from what the scan is.
    """
    key = style.canonical_method(method)
    data = dataset_of(scan)
    out = ['ew', 'droplast']                      # every scan, every method
    if ranks_only(scan):
        out.append('target')
    out.append(_SPLIT_CAUSE.get(data))
    if data == 'cifar':
        out.append('bn')
    if data in ('toy', 'polynomial'):
        out.append('fixedval')
    if key in ('Muon', 'MuonW'):
        out.append('muon')
    if key == 'AdamW':
        out.append('adamw_wd')
    if not in_legacy or key in _ALWAYS_NEW:
        out.append('newbase')
    if grid_grew:
        out.append('grid')
    if key in ('HIG', 'KFAC', 'JD_UPGrad', 'LBFGS', 'Sven'):
        out.append('failures')
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return tuple(uniq)


# ---------------------------------------------------------------------------
# Loading one side
# ---------------------------------------------------------------------------
def load_side(scan, legacy, plot_dir=None, results_root=None):
    """One scan from one root as a :class:`scan_analysis.Scan`, with :data:`METRIC` set.

    The legacy side's metric is the example-weighted repair joined on ``run_id``; the fresh
    side's is the recorded ``final_val_loss``.  Diverged and failed runs keep NaN on both
    sides (they are counted, never averaged).  ``scan.df.attrs`` carry the join's coverage.
    """
    plot_dir = Path(plot_dir or PLOT_DIR)
    if legacy:
        scobj = sa.load_scan(scan, scan_title(scan), plot_dir, results_root=LEGACY_ROOT,
                             cache_dir=LEGACY_CACHE)
        cov = attach_ew(scobj, scan)
    else:
        scobj = sa.load_scan(scan, scan_title(scan), plot_dir, results_root=results_root)
        scobj.df[METRIC] = scobj.df['final_val_loss']
        cov = {'source': 'recorded final_val_loss (fresh runs are example-weighted, C-E1)',
               'n_runs': len(scobj.df), 'n_metric': int(scobj.df[METRIC].notna().sum()),
               'n_usable': int((~scobj.df['diverged'] & ~scobj.df['failed']).sum()),
               'coverage': 1.0, 'repaired': False}
        _split_views(scobj)
    scobj.df.attrs['metric_source'] = cov
    scobj.metric = METRIC
    return scobj, cov


def _split_views(scobj):
    """Rebuild ``scan.sven`` / ``scan.baseline`` after a column is added to ``scan.df``."""
    d = scobj.df
    scobj.sven = d[d['optimizer'] == 'SVD'].copy()
    scobj.baseline = d[d['optimizer'] != 'SVD'].copy()


def attach_ew(scobj, scan, out_dir=None):
    """Join :mod:`repair_legacy`'s example-weighted final validation loss onto a legacy scan.

    Writes :data:`METRIC`.  Falls back to the RECORDED (batch-averaged) loss for a scan with
    no repair parquet, and says so in the returned dict -- a fallback is a caveat on every
    number of that scan, not a detail.
    """
    d = scobj.df
    usable = ~d['diverged'] & ~d['failed']
    try:
        rep = RL.load_repair(scan, out_dir=out_dir)
    except (FileNotFoundError, OSError):
        d[METRIC] = np.where(usable, d['final_val_loss'], np.nan)
        _split_views(scobj)
        return {'source': 'RECORDED (batch-averaged): no repair parquet for this scan',
                'n_runs': len(d), 'n_metric': int(d[METRIC].notna().sum()),
                'n_usable': int(usable.sum()), 'coverage': np.nan, 'repaired': False}
    ew = rep.drop_duplicates('run_id').set_index('run_id')['final_val_loss_ew']
    joined = d['run_id'].map(ew)
    d[METRIC] = np.where(usable, joined, np.nan)
    _split_views(scobj)
    n_usable = int(usable.sum())
    n_cov = int((usable & joined.notna()).sum())
    corr = pd.to_numeric(rep.get('rel_corr_final'), errors='coerce') \
        if 'rel_corr_final' in rep else pd.Series(dtype=float)
    corr = corr[np.isfinite(corr)]
    # a usable run with no repaired value would silently shrink its configuration's seed
    # count, so report the coverage rather than assume it
    return {'source': f'example-weighted repair ({RL.OUT_DIR}/{scan}.parquet)',
            'n_runs': len(d), 'n_metric': int(np.isfinite(d[METRIC]).sum()),
            'n_usable': n_usable, 'coverage': n_cov / n_usable if n_usable else np.nan,
            'repaired': True,
            'median_corr_%': float(100 * corr.median()) if len(corr) else np.nan,
            'max_corr_%': float(100 * corr.max()) if len(corr) else np.nan}


# ---------------------------------------------------------------------------
# The rule, applied to one side
# ---------------------------------------------------------------------------
def _sven_keys(scan, df):
    keys = list(sa.SVEN_CONFIG)
    extra = SVEN_ONLY_SCANS.get(scan)
    if extra and extra in df.columns:
        keys.append(extra)
    return keys


def best_per_method(scobj, scan, group=None, group_value=None):
    """The selected configuration of every method under the binding rule, one row each.

    ``group`` restricts to one value of a sweep axis (``n_data``, ``batch_size``) first, so
    the rule is applied within the axis rather than across it.  Columns: ``method`` (key),
    ``display``, ``config``, ``val`` / ``val_std``, ``finished`` / ``attempted`` /
    ``n_diverged``, ``n_configs`` (distinct configurations searched for this method) and
    ``rank`` (1 = lowest validation loss among the eligible methods).
    """
    df = scobj.df
    if group is not None:
        df = df[df[group] == group_value]
    if df.empty:
        return pd.DataFrame(columns=['method', 'display', 'config', 'val', 'val_std',
                                     'finished', 'attempted', 'n_diverged', 'n_configs',
                                     'rank'])
    sven_df = df[df['optimizer'] == 'SVD']
    base_df = df[df['optimizer'] != 'SVD']
    rows = []
    if len(sven_df):
        keys = _sven_keys(scan, sven_df)
        cfg = scobj.configs(sven_df, keys, metric=METRIC)
        rows.append(_first_eligible('SVD', cfg, keys, n_configs=len(cfg)))
    for opt, sub in base_df.groupby('optimizer', dropna=False):
        cfg = scobj.configs(sub, sa.BASELINE_CONFIG, metric=METRIC)
        rows.append(_first_eligible(opt, cfg, [k for k in sa.BASELINE_CONFIG
                                               if k != 'optimizer'], n_configs=len(cfg)))
    out = pd.DataFrame([r for r in rows if r is not None])
    if out.empty:
        return out
    out['display'] = [style.method_label(m) for m in out['method']]
    out['method'] = [style.canonical_method(m) for m in out['method']]
    score = pd.to_numeric(out['val'], errors='coerce').where(out['eligible'])
    out['rank'] = score.rank(method='min')
    out['n_methods'] = int(score.notna().sum())
    return out.sort_values(['rank', 'val'], na_position='last').reset_index(drop=True)


def _config_str(row, keys, columns):
    """One configuration rendered from a ``configs`` row -- the ONE renderer of this module.

    Follows :func:`headline.config_label`'s conventions (a zero weight decay and the default
    ``kappa = 2`` are not printed), and both roots go through it, so two strings from
    :func:`best_per_method` or :func:`grid_config_strings` may be compared as configurations
    rather than as two renderings.
    """
    bits = []
    for k in keys:
        if k not in columns:
            continue
        v = row[k]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if k == 'weight_decay' and not v:
            continue                                   # wd = 0 is the norm
        if k == 'kappa' and float(v) == 2.0:
            continue                                   # kappa = 2 is the default
        bits.append(f'{k}={v:g}' if isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool) else f'{k}={v}')
    return ', '.join(bits) or '(no swept hyperparameter)'


def grid_config_strings(scobj, scan):
    """``{method: {config string}}`` -- every configuration ON DISK, same renderer as above.

    :func:`grid_counts` says how many; this says which, so "is the fresh winner even on the
    legacy grid?" is answerable.  The strings are keyed the same way on both sides, and a
    configuration whose every seed crashed without a record is absent on the legacy side --
    the same lower bound as :func:`grid_counts`, for the same reason (F6).
    """
    out = {}
    df = scobj.df
    sven_df = df[df['optimizer'] == 'SVD']
    if len(sven_df):
        keys = _sven_keys(scan, sven_df)
        cfg = scobj.configs(sven_df, keys, metric=METRIC)
        out['Sven'] = {_config_str(r, keys, cfg.columns) for _, r in cfg.iterrows()}
    base_keys = [k for k in sa.BASELINE_CONFIG if k != 'optimizer']
    for opt, sub in df[df['optimizer'] != 'SVD'].groupby('optimizer', dropna=False):
        cfg = scobj.configs(sub, sa.BASELINE_CONFIG, metric=METRIC)
        out[style.canonical_method(opt)] = {_config_str(r, base_keys, cfg.columns)
                                            for _, r in cfg.iterrows()}
    return out


def _first_eligible(method, cfg, keys, n_configs):
    """The winning row of one method's ``configs`` table (already sorted by the rule)."""
    if cfg is None or cfg.empty:
        return {'method': method, 'config': '(no runs)', 'val': np.nan, 'val_std': np.nan,
                'lr': np.nan, 'finished': 0, 'attempted': 0, 'n_diverged': 0, 'n_failed': 0,
                'n_configs': 0, 'eligible': False}
    elig = cfg[cfg['eligible']]
    row = (elig if len(elig) else cfg).iloc[0]
    lr = row['lr'] if 'lr' in cfg.columns else np.nan
    return {'method': method, 'config': _config_str(row, keys, cfg.columns),
            'val': float(row['score']), 'val_std': float(row['score_std']),
            'lr': float(lr) if lr is not None and np.isfinite(lr) else np.nan,
            'finished': int(row['finished']), 'attempted': int(row['attempted']),
            'n_diverged': int(row['n_diverged']), 'n_failed': int(row['n_failed']),
            'n_configs': int(n_configs), 'eligible': bool(row['eligible'])}


def grid_counts(scobj, scan):
    """``{method: distinct configurations present}`` -- the tuning budget actually on disk.

    Counted from the records, so a configuration whose every seed crashed without leaving a
    record does not count on the legacy side.  That is the point: it did not exist as far as
    the legacy tables could tell (F6).
    """
    out = {}
    df = scobj.df
    sven_df = df[df['optimizer'] == 'SVD']
    if len(sven_df):
        out['Sven'] = int(len(sven_df.groupby(_sven_keys(scan, sven_df),
                                              dropna=False).size()))
    for opt, sub in df[df['optimizer'] != 'SVD'].groupby('optimizer', dropna=False):
        keys = [k for k in sa.BASELINE_CONFIG if k in sub.columns]
        out[style.canonical_method(opt)] = int(len(sub.groupby(keys, dropna=False).size()))
    return out


# ---------------------------------------------------------------------------
# The fresh side of a headline scan: the selection of record
# ---------------------------------------------------------------------------
def _sel_lr(scan, method, payload=None):
    """The selected learning rate of one method, from the selection of record."""
    sels = hl.selection_methods(scan, payload=payload)
    for name, sel in sels.items():
        if style.canonical_method(name) == style.canonical_method(method):
            lr = (sel.get('hparams') or {}).get('lr')
            return float(lr) if lr is not None else np.nan
    return np.nan


def fresh_headline_table(scan, payload=None, results_root=None):
    """The fresh numbers of a headline scan from :func:`headline.confirmation_table`.

    Never a re-selection: the configuration, its tuning-seed numbers and its confirmation
    numbers all come from ``bench/best_configs.json`` through :mod:`headline`, which is the
    one implementation of those tables.
    """
    tbl = hl.confirmation_table(scan, payload=payload, results_root=results_root)
    rank_tune = pd.to_numeric(tbl['val_tune'], errors='coerce').where(
        tbl['elig_tune']).rank(method='min')
    rank_conf = pd.to_numeric(tbl['val_conf'], errors='coerce').where(
        tbl['elig_conf']).rank(method='min')
    out = pd.DataFrame({
        'method': tbl['method'], 'display': tbl['display'], 'config_fresh': tbl['config'],
        'val_tune': tbl['val_tune'], 'val_tune_std': tbl['val_tune_std'],
        'fin_tune': tbl['fin_tune'], 'att_tune': tbl['att_tune'],
        'div_tune': tbl['div_tune'],
        'val_conf': tbl['val_conf'], 'val_conf_std': tbl['val_conf_std'],
        'test_conf': tbl.get('test_conf'), 'acc_conf': tbl.get('acc_conf'),
        'fin_conf': tbl['fin_conf'], 'att_conf': tbl['att_conf'],
        'rank_tune': rank_tune, 'rank_conf': rank_conf,
        'elig_conf': tbl['elig_conf'],
        # the selected learning rate as a NUMBER: comparing configuration strings across the
        # two sides compares two renderers, not two configurations
        'lr': [_sel_lr(scan, m, payload) for m in tbl['method']],
    })
    out.attrs.update(tbl.attrs)
    return out


def selection_agreement(scans=None, results_root=None, payload=None):
    """Does THIS module's rule reproduce the selection of record?  One row per (scan, method).

    The legacy side has no selection file, so the only way to know that the legacy numbers
    were produced by the same rule as the fresh ones is to run this module's code path on the
    fresh scans and diff it against ``bench/best_configs.json``.  ``agrees`` compares the
    selected configuration's seed-mean validation loss (the quantity the rule minimises),
    not the config string, because exact ties between equivalent Sven configurations are
    common by construction (:meth:`scan_analysis.Scan.configs`).
    """
    rows = []
    for scan in (scans or HEADLINE_SCANS):
        scobj, _ = load_side(scan, legacy=False, results_root=results_root)
        mine = best_per_method(scobj, scan).set_index('method')
        for method, sel in hl.selection_methods(scan, payload=payload).items():
            key = style.canonical_method(method)
            got = mine.loc[key] if key in mine.index else None
            mine_val = float(got['val']) if got is not None else np.nan
            ref = float(sel['seed_mean_final_val'])
            rel = abs(mine_val - ref) / abs(ref) if np.isfinite(mine_val) and ref else np.nan
            rows.append({'scan': scan, 'method': key,
                         'val_mine': mine_val, 'val_selection': ref, 'rel_diff': rel,
                         'agrees': bool(np.isfinite(rel) and rel <= 1e-9)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------
def diff_table(scan, legacy=None, fresh=None, payload=None, results_root=None):
    """Legacy best vs fresh best, one row per method, for a headline or ungrouped scan.

    ``rank_leg`` / ``rank_tune`` / ``rank_conf`` rank within their own field, whose size the
    campaign changed; ``rank_leg_common`` / ``rank_fresh_common`` rank within the methods
    present in BOTH roots, which is the like-for-like comparison and the number
    :func:`sven_rank_summary` quotes.  ``d_rank`` is the change in the common-field rank
    (negative = Sven moved up).
    """
    leg = legacy if legacy is not None else load_side(scan, legacy=True)[0]
    fre = fresh if fresh is not None else load_side(scan, legacy=False,
                                                    results_root=results_root)[0]
    L = best_per_method(leg, scan).set_index('method')
    grids_leg, grids_fresh = grid_counts(leg, scan), grid_counts(fre, scan)
    if scan in HEADLINE_SCANS:
        F = fresh_headline_table(scan, payload=payload,
                                 results_root=results_root).set_index('method')
        F['n_configs_fresh'] = [grids_fresh.get(m, 0) for m in F.index]
    else:
        f = best_per_method(fre, scan).set_index('method')
        F = pd.DataFrame({
            'display': f['display'], 'config_fresh': f['config'],
            'val_tune': f['val'], 'val_tune_std': f['val_std'],
            'fin_tune': f['finished'], 'att_tune': f['attempted'],
            'div_tune': f['n_diverged'], 'rank_tune': f['rank'],
            'val_conf': np.nan, 'val_conf_std': np.nan, 'test_conf': np.nan,
            'acc_conf': np.nan, 'fin_conf': 0, 'att_conf': 0, 'rank_conf': np.nan,
            'elig_conf': False, 'n_configs_fresh': [grids_fresh.get(m, 0) for m in f.index],
            'lr': f['lr'],
        })
    F = F.rename(columns={'lr': 'lr_fresh'})
    keep = ['config', 'val', 'val_std', 'lr', 'finished', 'attempted', 'n_diverged',
            'eligible']
    out = F.join(L[keep].rename(columns={
        'config': 'config_leg', 'val': 'val_leg', 'val_std': 'val_leg_std', 'lr': 'lr_leg',
        'finished': 'fin_leg', 'attempted': 'att_leg', 'n_diverged': 'div_leg',
        'eligible': 'elig_leg'}), how='outer')
    out.index.name = 'method'
    out = out.reset_index()
    out['display'] = [d if isinstance(d, str) else style.method_label(m)
                      for m, d in zip(out['method'], out['display'])]
    out['in_legacy'] = out['method'].isin(L.index)
    out['in_fresh'] = out['method'].isin(F.index)
    out['n_configs_leg'] = [grids_leg.get(m, 0) for m in out['method']]
    out['n_configs_fresh'] = out['n_configs_fresh'].fillna(0).astype(int)
    # did the rule pick the SAME learning rate on both sides?  (numbers, not strings)
    out['same_lr'] = [bool(np.isfinite(a) and np.isfinite(b) and float(a) == float(b))
                      for a, b in zip(out['lr_leg'], out['lr_fresh'])]
    out['rank_leg'] = pd.to_numeric(out['val_leg'], errors='coerce').where(
        out['elig_leg'].fillna(False)).rank(method='min')
    # like for like: rank inside the methods both campaigns ran
    both = out['in_legacy'] & out['in_fresh']
    # What the fresh rank is computed ON.  The confirmation mean of a configuration that is
    # NOT eligible on the confirmation seeds (polynomial L-BFGS: 2 of 15 finished) is a mean
    # over the survivors, so ranking on it would rank a method by its luckiest two runs.
    # The rule's own answer is to fall back to the tuning-seed loss -- but the row still
    # PRINTS the confirmation mean, so a rank that came from the fallback is a rank on a
    # different number than the cell beside it, and `fresh_basis` says which.  Every table
    # and figure that consumes `val_fresh_rankon` must carry the marker with it.
    conf_ok = out['elig_conf'].fillna(False) & out['val_conf'].notna()
    fresh_score = out['val_conf'].where(conf_ok)
    fresh_score = fresh_score.where(fresh_score.notna(), out['val_tune'])
    out['val_fresh_rankon'] = fresh_score
    out['fresh_basis'] = np.where(
        ~np.isfinite(pd.to_numeric(fresh_score, errors='coerce')), '--',
        np.where(conf_ok, 'confirm', 'tuning'))
    out['rank_leg_common'] = out['val_leg'].where(both & out['elig_leg'].fillna(False)) \
        .rank(method='min')
    out['rank_fresh_common'] = fresh_score.where(both).rank(method='min')
    # the fresh TUNING-seed rank in the same field: the like-for-like protocol (one problem
    # instance, the seeds the selection was made on), which separates "the data changed"
    # from "the protocol changed to fresh confirmation seeds"
    out['rank_tune_common'] = out['val_tune'].where(both).rank(method='min')
    out['d_rank'] = out['rank_fresh_common'] - out['rank_leg_common']
    out['d_rank_tune'] = out['rank_tune_common'] - out['rank_leg_common']
    out['n_common'] = int(both.sum())
    # the full fresh field (every method the fresh campaign ran on this scan), which is what
    # a rank in the paper is out of
    out['n_fresh_methods'] = int(fresh_score.where(out['in_fresh']).notna().sum())
    out['causes'] = [','.join(causes_for(scan, m, in_legacy=inl,
                                         grid_grew=nf > nl > 0))
                     for m, inl, nl, nf in zip(out['method'], out['in_legacy'],
                                               out['n_configs_leg'],
                                               out['n_configs_fresh'])]
    out.attrs.update(scan=scan, title=scan_title(scan), ranks_only=ranks_only(scan),
                     kind=scan_kind(scan),
                     legacy_metric=leg.df.attrs.get('metric_source', {}),
                     confirm=scan in HEADLINE_SCANS)
    order = pd.to_numeric(out['rank_fresh_common'], errors='coerce').fillna(
        pd.to_numeric(out['rank_tune'], errors='coerce') + 100)
    return out.assign(_o=order).sort_values('_o', na_position='last') \
              .drop(columns='_o').reset_index(drop=True)


#: appended to a fresh rank that was computed on the TUNING-seed loss because the
#: confirmation pass of that row is not eligible (:func:`diff_table`'s ``fresh_basis``)
FALLBACK_MARK = ' *'

#: what :data:`FALLBACK_MARK` means, for a table footer
FALLBACK_NOTE = (f'{FALLBACK_MARK.strip()} the fresh rank of this row is computed on its '
                 f'TUNING-seed loss, not on the confirmation mean printed beside it: too '
                 f'few confirmation runs finished for the configuration to be eligible '
                 f'(style.config_eligible), so its confirmation mean is a mean over the '
                 f'survivors and ranking on it would rank the method by its luckiest runs.')


def fallback_rows(tbl):
    """The rows whose fresh rank came from the tuning-seed fallback, as a small table.

    One row per substitution, with both numbers and the confirmation counts that caused it,
    so the notebook can print the substitution instead of leaving the reader to infer it
    from a marker.  Empty when every ranked row is ranked on its confirmation mean.
    """
    if 'fresh_basis' not in tbl:
        return pd.DataFrame(columns=['method', 'display', 'val_conf', 'fin_conf',
                                     'att_conf', 'val_tune', 'val_ranked_on'])
    sub = tbl[(tbl['fresh_basis'] == 'tuning') & tbl['val_conf'].notna()]
    return pd.DataFrame({
        'method': sub['method'], 'display': sub['display'],
        'val_conf': sub['val_conf'], 'fin_conf': sub['fin_conf'],
        'att_conf': sub['att_conf'], 'val_tune': sub['val_tune'],
        'val_ranked_on': sub['val_fresh_rankon'],
    }).reset_index(drop=True)


def diff_view(tbl, spec='.4g'):
    """:func:`diff_table` as it is read: legacy | fresh tuning | fresh confirmation | causes.

    A ranks-only scan (legacy additive cubic) shows its legacy loss as ``n/a (different
    target)``: the rank is the only comparable quantity there.

    Both fresh rank columns carry :data:`FALLBACK_MARK` and a ``fresh rank basis`` column
    where the rank was computed on the tuning-seed loss rather than on the confirmation mean
    the same row prints (:func:`fallback_rows`, :data:`FALLBACK_NOTE`).
    """
    ranks_only = bool(tbl.attrs.get('ranks_only'))
    has_conf = bool(tbl.attrs.get('confirm')) and tbl['val_conf'].notna().any()
    has_acc = 'acc_conf' in tbl and pd.to_numeric(tbl['acc_conf'], errors='coerce').notna().any()
    rows = []
    for _, r in tbl.iterrows():
        row = {'method': r['display']}
        row['legacy config'] = r['config_leg'] if isinstance(r['config_leg'], str) else '--'
        row['legacy val'] = ('n/a (different target)' if ranks_only and r['in_legacy']
                             else hl.fmt_pm(r['val_leg'], r['val_leg_std'], spec))
        row['legacy fin/att'] = ('--' if not r['in_legacy']
                                 else hl.fmt_counts(r['fin_leg'], r['att_leg'])
                                 + ('' if r['elig_leg'] else ' (not eligible)'))
        row['legacy rank'] = rank_cell(r['rank_leg_common'], r['n_common'])
        row['fresh config'] = r['config_fresh'] if isinstance(r['config_fresh'], str) else '--'
        row['fresh val (tuning)'] = hl.fmt_pm(r['val_tune'], r['val_tune_std'], spec)
        row['fresh rank (tuning)'] = rank_cell(r.get('rank_tune_common'), r['n_common'])
        if has_conf:
            row['fresh val (confirm)'] = hl.fmt_pm(r['val_conf'], r['val_conf_std'], spec)
            row['fresh test'] = hl.fmt_pm(r.get('test_conf'), None, spec)
            if has_acc:
                row['fresh test acc'] = hl.fmt_pm(r.get('acc_conf'), None, '.3g')
            row['fresh fin/att'] = hl.fmt_counts(r['fin_conf'], r['att_conf'])
        else:
            row['fresh fin/att'] = hl.fmt_counts(r['fin_tune'], r['att_tune'])
        # a rank computed on the tuning-seed loss instead of on the confirmation mean this
        # row prints is marked, and the basis is spelled out in its own column
        basis = r.get('fresh_basis', 'confirm')
        mark = FALLBACK_MARK if (has_conf and basis == 'tuning') else ''
        row['fresh rank'] = rank_cell(r['rank_fresh_common'], r['n_common']) + mark
        row['rank change'] = ('--' if not np.isfinite(r['d_rank'])
                              else ('=' if r['d_rank'] == 0 else f"{r['d_rank']:+.0f}"))
        # the rank in the FULL fresh field, which the new baselines made larger: the number
        # the paper quotes ("9th of 11"), beside the like-for-like one
        row['fresh rank (all methods)'] = rank_cell(
            r.get('rank_conf') if np.isfinite(r.get('rank_conf', np.nan))
            else r.get('rank_tune'), r.get('n_fresh_methods', np.nan)) + mark
        if has_conf:
            row['fresh rank basis'] = (
                'confirm' if basis == 'confirm' else
                f"tuning ({hl.fmt_counts(r['fin_conf'], r['att_conf'])} confirmation runs "
                f"finished: not eligible)" if basis == 'tuning' else '--')
        if not has_conf:
            row.pop('fresh rank (tuning)', None)
        row['grid (leg->fresh)'] = f"{int(r['n_configs_leg'])}->{int(r['n_configs_fresh'])}"
        row['causes'] = r['causes']
        rows.append(row)
    view = pd.DataFrame(rows)
    view.attrs.update(tbl.attrs)
    return view


def rank_cell(rank, n):
    """``'2/14'`` -- a rank is unreadable without the size of the field it is a rank in."""
    if rank is None or not np.isfinite(rank):
        return '--'
    if n is None or not np.isfinite(n):
        return f'{int(rank)}'
    return f'{int(rank)}/{int(n)}'


# ---------------------------------------------------------------------------
# Grouped studies (a sweep axis) and the Sven-only ablations
# ---------------------------------------------------------------------------
def grouped_rank_table(scan, legacy=None, fresh=None, results_root=None, method='Sven'):
    """One method's rank at every value of a swept axis, legacy vs fresh.

    The axis is :data:`GROUPED_SCANS`'s column.  Ranks are over the methods present in both
    roots at that axis value, so "2nd of 11" and "2nd of 12" are not silently compared.
    """
    axis = GROUPED_SCANS[scan]
    leg = legacy if legacy is not None else load_side(scan, legacy=True)[0]
    fre = fresh if fresh is not None else load_side(scan, legacy=False,
                                                    results_root=results_root)[0]
    values = sorted(set(pd.Series(leg.df[axis]).dropna().unique())
                    & set(pd.Series(fre.df[axis]).dropna().unique()))
    rows = []
    for v in values:
        L = best_per_method(leg, scan, group=axis, group_value=v).set_index('method')
        F = best_per_method(fre, scan, group=axis, group_value=v).set_index('method')
        common = [m for m in L.index if m in F.index]
        lr = L.loc[common, 'val'].where(L.loc[common, 'eligible']).rank(method='min')
        fr = F.loc[common, 'val'].where(F.loc[common, 'eligible']).rank(method='min')
        row = {axis: v, 'n_common': len(common),
               'rank_leg': lr.get(method, np.nan), 'rank_fresh': fr.get(method, np.nan),
               'val_leg': L['val'].get(method, np.nan),
               'val_fresh': F['val'].get(method, np.nan),
               'config_leg': L['config'].get(method, '--'),
               'config_fresh': F['config'].get(method, '--'),
               'best_leg': lr.idxmin() if lr.notna().any() else '--',
               'best_fresh': fr.idxmin() if fr.notna().any() else '--'}
        row['d_rank'] = row['rank_fresh'] - row['rank_leg']
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(scan=scan, title=scan_title(scan), axis=axis, method=method,
                     ranks_only=ranks_only(scan))
    return out


def sven_only_table(scans=None, results_root=None):
    """The Sven-only ablations: best Sven configuration and loss on each side, one row each.

    No rank exists (one method), so what changed is the selected configuration and the
    level.  ``axis`` names the ablation the selection is NOT pooled over.

    The two grids are counted on both sides (:func:`grid_counts`) and the ``grid`` cause is
    attributed from them, exactly as in :func:`diff_table`: on the kappa study the legacy
    grid is 3 configurations against 42 and the fresh winner does not exist in it, so a
    level change there is not a like-for-like measurement and the row must say so.
    """
    rows = []
    for scan in (scans or SVEN_ONLY_SCANS):
        leg, cov_l = load_side(scan, legacy=True)
        fre, cov_f = load_side(scan, legacy=False, results_root=results_root)
        L = best_per_method(leg, scan).set_index('method')
        F = best_per_method(fre, scan).set_index('method')
        grids_leg, grids_fresh = grid_counts(leg, scan), grid_counts(fre, scan)
        cfgs_leg = grid_config_strings(leg, scan)
        for key in sorted(set(L.index) | set(F.index)):
            n_leg, n_fresh = grids_leg.get(key, 0), grids_fresh.get(key, 0)
            rows.append({
                'scan': scan, 'title': scan_title(scan),
                'axis': SVEN_ONLY_SCANS.get(scan, ''), 'method': key,
                'config_leg': L['config'].get(key, '--'),
                'val_leg': L['val'].get(key, np.nan),
                'fin_leg': L['finished'].get(key, 0), 'att_leg': L['attempted'].get(key, 0),
                'div_leg': L['n_diverged'].get(key, 0),
                'config_fresh': F['config'].get(key, '--'),
                'val_fresh': F['val'].get(key, np.nan),
                'fin_fresh': F['finished'].get(key, 0),
                'att_fresh': F['attempted'].get(key, 0),
                'div_fresh': F['n_diverged'].get(key, 0),
                'n_configs_leg': n_leg, 'n_configs_fresh': n_fresh,
                'grid_grew': bool(n_fresh > n_leg > 0),
                # the fresh winner may not even be ON the legacy grid, which makes the
                # level change a search-budget statement and not a data-change one
                'fresh_config_in_legacy_grid': bool(
                    isinstance(F['config'].get(key), str)
                    and F['config'].get(key) in cfgs_leg.get(key, set())),
                'ranks_only': ranks_only(scan),
                'repaired': bool(cov_l.get('repaired')),
                'causes': ','.join(causes_for(scan, key, in_legacy=key in L.index,
                                              grid_grew=n_fresh > n_leg > 0)),
            })
        del leg, fre
    return pd.DataFrame(rows)


def sven_rank_summary(diffs=None, grouped=None):
    """The cross-scan table: Sven's rank, legacy -> fresh, in the common field.

    ``rank_leg`` / ``rank_fresh`` are out of ``n_common`` methods (present in both roots).
    For a grouped study the rank is reported at every axis value, so the row count is the
    axis, not the scan.
    """
    rows = []
    for tbl in (diffs or {}).values():
        r = tbl[tbl['method'] == 'Sven']
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({'scan': tbl.attrs['scan'], 'title': tbl.attrs['title'], 'axis': '',
                     'point': '', 'n_common': r['n_common'],
                     'rank_leg': r['rank_leg_common'], 'rank_fresh': r['rank_fresh_common'],
                     'd_rank': r['d_rank'],
                     'val_leg': np.nan if tbl.attrs.get('ranks_only') else r['val_leg'],
                     'val_fresh': np.nan if tbl.attrs.get('ranks_only')
                     else r['val_fresh_rankon'],
                     'ranks_only': bool(tbl.attrs.get('ranks_only'))})
    for tbl in (grouped or {}).values():
        axis = tbl.attrs['axis']
        for _, r in tbl.iterrows():
            rows.append({'scan': tbl.attrs['scan'], 'title': tbl.attrs['title'],
                         'axis': axis, 'point': f'{axis}={r[axis]:g}',
                         'n_common': r['n_common'], 'rank_leg': r['rank_leg'],
                         'rank_fresh': r['rank_fresh'], 'd_rank': r['d_rank'],
                         'val_leg': np.nan if tbl.attrs.get('ranks_only') else r['val_leg'],
                         'val_fresh': np.nan if tbl.attrs.get('ranks_only')
                         else r['val_fresh'],
                         'ranks_only': bool(tbl.attrs.get('ranks_only'))})
    return pd.DataFrame(rows)


def rank_change_tally(summary):
    """The aggregate of :func:`sven_rank_summary`, SPLIT by whether the target is the same.

    An aggregate over every row of that table is not a statement about the robustness
    fixes: the polynomial rows are ranks on a DIFFERENT objective (the legacy additive
    cubic, F1/C-D1), and on the fresh grids they are also Sven's largest improvements, so
    they dominate any pooled mean.  The two groups are therefore never added up here.

    One row per group (``same target`` / ``ranks only (different target)``) plus ``all``,
    with ``n``, ``better`` / ``same`` / ``worse``, ``sum_d_rank`` and ``mean_d_rank``.  The
    grouping is read off ``summary['ranks_only']``, the flag :func:`sven_rank_summary`
    already writes on every row, so a sentence quoting this cannot drift from the table.
    """
    s = summary.dropna(subset=['d_rank']).copy()
    s['ranks_only'] = s['ranks_only'].fillna(False).astype(bool)
    rows = []
    for label, sub in (('same target', s[~s['ranks_only']]),
                       ('ranks only (different target)', s[s['ranks_only']]),
                       ('all', s)):
        d = sub['d_rank']
        rows.append({'group': label, 'n': int(len(d)),
                     'better': int((d < 0).sum()), 'same': int((d == 0).sum()),
                     'worse': int((d > 0).sum()),
                     'sum_d_rank': float(d.sum()) if len(d) else np.nan,
                     'mean_d_rank': float(d.mean()) if len(d) else np.nan,
                     'headline': int((sub['axis'] == '').sum()),
                     'swept': int((sub['axis'] != '').sum())})
    out = pd.DataFrame(rows)
    out.attrs['note'] = ('the two groups are not added up: the ranks-only rows are ranks on '
                         'the legacy additive cubic, a different objective function')
    return out


# ---------------------------------------------------------------------------
# What is NOT in the intersection of the two roots
# ---------------------------------------------------------------------------
#: legacy scan directories with no fresh counterpart, and what each one's results now rest
#: on.  Every disposition is quoted from `EXPERIMENTS.md` section 10 ("Cut, parked, and
#: stale") and `campaign/CONTRACTS.md`, not decided here: :func:`scans_in_both_roots`
#: intersects the roots, so without this table a legacy figure that now has no fresh data
#: simply disappears from the account.
LEGACY_ONLY_DISPOSITION = {
    'exp_critbatch_mnist':
        'CUT by the user (09-18): the critical-batch study confounds retained rank with '
        'batch size (F23) and crossings were measured once per epoch. NO fresh data; the '
        'legacy critical-batch figure has no replacement and cannot be quoted.',
    'exp_critbatch_nanogpt':
        'CUT by the user (09-18), same reason. NO fresh data; the legacy critical-batch '
        'figure has no replacement and cannot be quoted.',
    'cifar10_resnet_kappaScan_labelReg':
        'CUT: 1 seed at a tentative set point. The kappa story now runs on MNIST at 5 '
        'seeds with a matched-effective-step grid (mnist_kappaScan_labelRegression); there '
        'is NO fresh CIFAR kappa scan.',
    'cifar10_resnet_ce_kappaScan':
        'CUT: 1 seed at a tentative set point; replaced on MNIST only '
        '(mnist_kappaScan_labelRegression). NO fresh CIFAR kappa scan.',
    'cifar10_resnet_paramFrac_scan_labelReg':
        'CUT: superseded by rebuttal_fig5_cifar_paramfrac_scan at 3 seeds, which exists in '
        'both roots and IS diffed here (the Sven-only table). The legacy Fig-5 numbers come '
        'from that 1-seed scan, not from this one.',
    'cifar10_resnet_ce_paramFrac_scan':
        'CUT: superseded by rebuttal_fig5_cifar_paramfrac_scan (label regression only), so '
        'the cross-entropy parameter-fraction curve has NO fresh counterpart.',
    'exp_finetune_cifar_smallN':
        'PARKED to the extension phase (user, 09-18): its legacy runs trained BatchNorm on '
        '250-2000 images, the defect C-E2 names. NO fresh data; the fine-tuning claim is '
        'unsupported until the parked 408 runs are launched.',
}


def scan_inventory(fresh_root=None):
    """Every scan directory in EITHER root, with which roots hold it and its disposition.

    :func:`scans_in_both_roots` is an intersection, so anything outside it vanishes from
    the diff.  This is the complement: ``where`` is ``both`` | ``legacy only`` |
    ``fresh only``, ``n_records`` counts the ``.jsonl`` files on each side, and
    ``disposition`` says what a legacy-only scan's results now rest on
    (:data:`LEGACY_ONLY_DISPOSITION`).  A fresh-only directory is normally an auxiliary
    PASS of a headline scan (``_confirm`` / ``_diag`` / ``_timing``), which the fresh
    protocol added and the legacy campaign had no equivalent of.
    """
    fresh = Path(style.resolve_results_root(fresh_root))
    if not fresh.is_absolute():
        fresh = (HERE / fresh).resolve()
    legacy = Path(LEGACY_ROOT)

    def scans(root):
        return {d.name: len(list(d.glob('*.jsonl'))) for d in root.iterdir()
                if d.is_dir() and not d.name.startswith('_') and any(d.glob('*.jsonl'))}

    L, F = scans(legacy), scans(fresh)
    rows = []
    for name in sorted(set(L) | set(F)):
        where = 'both' if name in L and name in F else (
            'legacy only' if name in L else 'fresh only')
        pass_of = ''
        if where == 'fresh only':
            for suffix in ('_confirm', '_diag', '_timing'):
                if name.endswith(suffix) and name[: -len(suffix)] in F:
                    pass_of = f'{suffix.lstrip("_")} pass of {name[: -len(suffix)]}'
        rows.append({'scan': name, 'title': scan_title(name), 'where': where,
                     'n_records_leg': L.get(name, 0), 'n_records_fresh': F.get(name, 0),
                     'compared as': scan_kind(name) if where == 'both' else '--',
                     'disposition': LEGACY_ONLY_DISPOSITION.get(name, pass_of)})
    out = pd.DataFrame(rows)
    out.attrs['n_both'] = int((out['where'] == 'both').sum())
    out.attrs['n_legacy_only'] = int((out['where'] == 'legacy only').sum())
    out.attrs['n_fresh_only'] = int((out['where'] == 'fresh only').sum())
    return out


# ---------------------------------------------------------------------------
# Causes that can be tested from the data
# ---------------------------------------------------------------------------
def adamw_duplicate_check(scobj, wd=None, extra_keys=(), tol=1e-9):
    """Was ``AdamW`` the same run as ``Adam``?  (F9: legacy forced ``weight_decay=0``.)

    Pairs the two optimizers' runs on ``(lr, model_seed)`` -- plus ``extra_keys`` for a scan
    that also sweeps ``n_data`` / ``batch_size`` -- and reports the largest relative
    difference in the final validation loss.  ``wd`` restricts AdamW to one weight-decay
    value: the legacy rebuttal scans hold BOTH the old ``wd = 0`` runs (which are Adam under
    another name) and the ``wd = 0.01`` replacements written on 2026-09-17, and mixing them
    hides the duplication.  ``n_identical == n_pairs`` is the duplication; a decaying AdamW
    must differ.
    """
    df = scobj.df
    keys = ['lr', 'model_seed'] + [k for k in extra_keys if k in df.columns]
    cols = keys + [METRIC]
    a = df[df['optimizer'] == 'Adam'][cols].dropna()
    w = df[df['optimizer'] == 'AdamW']
    wds_all = sorted(float(v) for v in
                     set(pd.to_numeric(w.get('weight_decay', pd.Series(dtype=float)),
                                       errors='coerce').dropna())) if len(w) else []
    if wd is not None and 'weight_decay' in w.columns:
        w = w[pd.to_numeric(w['weight_decay'], errors='coerce') == float(wd)]
    w = w[cols].dropna()
    if a.empty or w.empty:
        return {'n_pairs': 0, 'max_rel_diff': np.nan, 'n_identical': 0, 'wd': wd,
                'wds_present': wds_all}
    m = a.merge(w, on=keys, suffixes=('_adam', '_adamw'))
    rel = (m[f'{METRIC}_adamw'] - m[f'{METRIC}_adam']).abs() / m[f'{METRIC}_adam'].abs()
    return {'n_pairs': int(len(m)), 'max_rel_diff': float(rel.max()) if len(m) else np.nan,
            'n_identical': int((rel <= tol).sum()), 'wd': wd, 'wds_present': wds_all}


def nanogpt_split_check(scan='exp_nanogpt_speedrun', run_id=None, results_root=None,
                        tol=1e-4):
    """What the split change on the character corpus actually was, from the records.

    The `split` cause of MNIST / CIFAR ("the legacy runs validated on the official test
    set") does not describe the character corpus, which never had a test set.  The legacy
    corpus was 90/10 by position and the fresh one is 80/10/10
    (:func:`datasets.contiguous_split_bounds`), so the prediction is:

    * the legacy validation blocks are the **fresh test** split -- the same text, so the
      two curves start from the same initialisation on the same data and ``val[0]``
      (legacy) must equal ``test[0]`` (fresh);
    * the fresh **validation** set is new, carved out of what legacy trained on, so
      ``val[0]`` differs between the roots;
    * ``n_val`` is unchanged and ``n_train`` shrinks by exactly ``n_test``.

    Returns the three curve values, the two relative differences and ``verdict``.  The
    tolerance is loose on purpose: the two roots ran on different GPU types (F-cross-GPU),
    so "the same data" means agreeing far better than the other split does, not bitwise.
    """
    import json

    def first(root):
        d = Path(root) / scan
        files = sorted(d.glob(f'{run_id}.jsonl')) if run_id else sorted(d.glob('*.jsonl'))
        if not files:
            raise FileNotFoundError(f'no records in {d}')
        rec = json.loads(files[0].read_text().splitlines()[0])
        losses = rec.get('losses') or {}
        return {'run_id': rec.get('run_id'), 'n_train': rec.get('n_train'),
                'n_val': rec.get('n_val'), 'n_test': rec.get('n_test'),
                'val0': (losses.get('val') or [np.nan])[0],
                'test0': (losses.get('test') or [np.nan])[0]}

    fresh_root = Path(style.resolve_results_root(results_root))
    if not fresh_root.is_absolute():
        fresh_root = (HERE / fresh_root).resolve()
    leg, fre = first(LEGACY_ROOT), first(fresh_root)

    def rel(a, b):
        return abs(a - b) / abs(b) if np.isfinite(a) and np.isfinite(b) and b else np.nan

    leg_val_vs_fresh_test = rel(leg['val0'], fre['test0'])
    leg_val_vs_fresh_val = rel(leg['val0'], fre['val0'])
    same_test = np.isfinite(leg_val_vs_fresh_test) and leg_val_vs_fresh_test <= tol
    same_val = np.isfinite(leg_val_vs_fresh_val) and leg_val_vs_fresh_val <= tol
    return {
        'scan': scan, 'run_id': leg['run_id'],
        'n_train_leg': leg['n_train'], 'n_train_fresh': fre['n_train'],
        'n_val_leg': leg['n_val'], 'n_val_fresh': fre['n_val'],
        'n_test_leg': leg['n_test'], 'n_test_fresh': fre['n_test'],
        'val0_leg': leg['val0'], 'val0_fresh': fre['val0'], 'test0_fresh': fre['test0'],
        'rel_legval_vs_freshtest': leg_val_vs_fresh_test,
        'rel_legval_vs_freshval': leg_val_vs_fresh_val,
        'legacy_val_is_fresh_test': bool(same_test and not same_val),
        'n_val_unchanged': leg['n_val'] == fre['n_val'],
        'train_shrank_by_test': (leg['n_train'] is not None and fre['n_test'] is not None
                                 and leg['n_train'] - fre['n_train'] == fre['n_test']),
        'verdict': ('legacy validation blocks ARE the fresh test split; the fresh '
                    'validation set is new, carved out of legacy train'
                    if same_test and not same_val else
                    'the two roots share their validation set' if same_val else
                    'neither split matches: check the corpus'),
    }


def seed_completeness(scobj, scan, methods=None):
    """Per method: configurations observed, and how many of their seed-runs left no record.

    This is the signature of F6 on the legacy root.  A legacy exception was printed and
    dropped, so the crashed run has no file at all; the legacy analysis then inferred the
    expected seed count FROM THE FILES IT FOUND and a configuration with 4 survivors looked
    complete.  ``n_missing`` counts, per method, ``seeds - records`` summed over the
    configurations that did leave at least one record -- the visible part of the loss.  It
    cannot count a configuration that vanished entirely: that is what the manifest (C-R2)
    exists for, and the legacy root has none.
    """
    df = scobj.df
    n_seeds = len(scobj.seeds)
    rows = []
    for opt, sub in df.groupby('optimizer', dropna=False):
        key = style.canonical_method(opt)
        if methods is not None and key not in {style.canonical_method(m) for m in methods}:
            continue
        per_cfg = sub.groupby(sub['run_id'].map(style.config_key)).size()
        rows.append({'method': key, 'display': style.method_label(opt),
                     'n_configs': int(len(per_cfg)), 'n_runs': int(len(sub)),
                     'n_seeds': n_seeds,
                     'n_full_configs': int((per_cfg >= n_seeds).sum()),
                     'n_partial_configs': int((per_cfg < n_seeds).sum()),
                     'n_missing': int((n_seeds - per_cfg).clip(lower=0).sum()),
                     'n_diverged': int(sub['diverged'].sum()),
                     'n_failed': int(sub['failed'].sum())})
    out = pd.DataFrame(rows)
    return out.sort_values('n_missing', ascending=False).reset_index(drop=True)


def failure_visibility(legacy, fresh, methods=('HIG', 'KFAC', 'LBFGS', 'Sven')):
    """How many runs each method LEFT BEHIND on each side, and how many are failures.

    Legacy exceptions were printed and dropped (F6/C-R1), so a legacy method's missing runs
    are invisible: ``records`` is all the legacy tables could see.  ``n_diverged`` on the
    fresh side is what the same blow-ups now look like.
    """
    rows = []
    for m in methods:
        key = style.canonical_method(m)
        out = {'method': key, 'display': style.method_label(m)}
        for tag, scobj in (('leg', legacy), ('fresh', fresh)):
            if scobj is None:
                continue
            d = scobj.df
            sub = d[[style.canonical_method(o) == key for o in d['optimizer']]]
            out[f'records_{tag}'] = int(len(sub))
            out[f'diverged_{tag}'] = int(sub['diverged'].sum()) if len(sub) else 0
            out[f'failed_{tag}'] = int(sub['failed'].sum()) if len(sub) else 0
            out[f'usable_{tag}'] = int((~sub['diverged'] & ~sub['failed']).sum()) \
                if len(sub) else 0
        rows.append(out)
    return pd.DataFrame(rows)


def split_only_check(diff, methods=('Adam', 'SGD', 'RMSprop', 'Shampoo')):
    """The prediction "these methods move only because the split moved", as a table.

    For plain first-order baselines on MNIST nothing else in the fixed list applies (no
    BatchNorm, no Muon regrouping, no weight decay): the selected LEARNING RATE should come
    out the same although the grid around it grew, and the rank should barely move, while
    the level may shift because the validation examples are now different data.  The test is
    on the numbers (``same_lr``), not on the configuration strings, which are rendered by
    two different pieces of code.
    """
    keys = [style.canonical_method(m) for m in methods]
    sub = diff[diff['method'].isin(keys)]
    return pd.DataFrame({
        'method': sub['display'], 'legacy lr': sub['lr_leg'], 'fresh lr': sub['lr_fresh'],
        'same lr': sub['same_lr'],
        'grid': [f'{a}->{b}' for a, b in zip(sub['n_configs_leg'], sub['n_configs_fresh'])],
        'legacy val': sub['val_leg'], 'fresh val (tuning)': sub['val_tune'],
        'rel change': (sub['val_tune'] - sub['val_leg']) / sub['val_leg'],
        'rank change': sub['d_rank'], 'causes': sub['causes'],
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def savefig(fig, name, plot_dir=None):
    """Save a figure as PDF **and** PNG under ``plots_v2/legacy_vs_fresh/``."""
    d = Path(plot_dir or PLOT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    out = []
    for ext in ('pdf', 'png'):
        p = d / f'{stem}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        out.append(p)
    return out


def plot_rank_slope(summary, ax, label_col='label', spread=0.16, label_size=8):
    """Sven's rank legacy -> fresh, one line per scan / axis point (1 = best, at the top).

    Several points usually share a rank -- half the swept studies end at 1 or 2 -- so the
    endpoints of a group are fanned out by ``spread`` and the label rides on its own
    endpoint.  Without that the labels sit on top of each other and the figure says nothing.
    """
    s = summary.dropna(subset=['rank_leg', 'rank_fresh']).reset_index(drop=True)
    labels = (s[label_col] if label_col in s.columns else s['title']).astype(str)

    def fan(values):
        """Per group of equal values, offsets centred on the value."""
        out = np.zeros(len(values), dtype=float)
        for v in sorted(set(values)):
            idx = [i for i, x in enumerate(values) if x == v]
            for j, i in enumerate(idx):
                out[i] = (j - (len(idx) - 1) / 2) * spread
        return out

    ya = np.asarray(s['rank_leg'], float) + fan(list(s['rank_leg']))
    yb = np.asarray(s['rank_fresh'], float) + fan(list(s['rank_fresh']))
    for i in range(len(s)):
        d = s['rank_fresh'].iloc[i] - s['rank_leg'].iloc[i]
        color = '#b2182b' if d > 0 else ('#2166ac' if d < 0 else '0.45')
        ax.plot([0, 1], [ya[i], yb[i]], '-o', color=color, lw=1.6, ms=4.5, zorder=3,
                alpha=0.9)
        ax.annotate(f'  {labels.iloc[i]}', (1, yb[i]), fontsize=label_size, va='center',
                    color=color, annotation_clip=False)
    lo = min(ya.min(), yb.min()) - 0.4
    hi = max(ya.max(), yb.max()) + 0.4
    ax.set_xlim(-0.12, 1.06)
    ax.set_ylim(hi, lo)                       # 1 = best, at the top
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['legacy', 'fresh'])
    ax.set_yticks([v for v in range(1, int(np.ceil(hi)) + 1)])
    ax.set_ylabel("Sven's rank (methods in both roots)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_method_deltas(tbl, ax, ylabel='Final validation loss (seed mean)'):
    """Per-method legacy -> fresh validation loss on one scan (log axis, method colours).

    Only the methods BOTH campaigns ran can appear: a bar pair is a comparison, and a fresh
    method with no legacy twin would be half a bar.
    """
    s = tbl.dropna(subset=['val_leg']).copy()
    s['fresh'] = s['val_fresh_rankon']
    s = s.dropna(subset=['fresh'])
    x = np.arange(len(s))
    ax.bar(x - 0.2, s['val_leg'], width=0.38, label='legacy (repaired)', color='0.65')
    ax.bar(x + 0.2, s['fresh'], width=0.38, label='fresh (bar colour = method)',
           color=[style.method_color(m) for m in s['method']])
    ax.set_xticks(x)
    ax.set_xticklabels(s['display'], rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    return ax
