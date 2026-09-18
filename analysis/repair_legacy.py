"""Example-weighted validation curves for the LEGACY results, offline (C-A1).

The training loops averaged the per-batch validation means
(``np.mean`` over batches, ``experiment_utils.py:276,375``), so the last,
short batch of every epoch carried the same weight as a full one.  The
validation loader never dropped it -- ``DataLoader(val_dataset,
batch_size=batch_size, shuffle=False)`` (``generic_scan.py:542``) -- so with
``n_val = 10000`` and ``B = 256`` sixteen examples were weighted like 256.
Fable measured the damage: a fraction of a percent in the headline scans, up to
33% for individual runs in the full-batch studies (FABLE_CRITIQUES.md section 8).

Every legacy run stored its per-batch validation losses in
``{scan}/diag/{run_id}.npz`` (``val_batch``, all epochs concatenated), and the
batch lengths follow from ``n_val`` and the batch size alone.  So the curves are
*exactly* repairable without rerunning anything:

    l_ew[e] = sum_b n_b * l[e, b] / n_val,   n_b = B except n_val - (nb-1) B last.

This module recomputes them, writes NEW columns next to the results (under
``analysis/legacy_repair/``, NEVER into the results root, which is read-only for
the campaign) and reports the size of the correction per scan.  Accuracy is not
repairable: it was never stored per batch.

Two checks stand between the arithmetic and the numbers being trusted:

* the EQUAL-weight mean of the same blocks must reproduce the stored curve
  (``eq_check``, computed for every run, asserted on a sample of the runs that
  could produce a comparison, per batch size) -- that is what proves the reshape
  into epochs x batches is the right one;
* ``ceil(n_val / B) == n_batches`` for every batch size in the scan pins ``n_val``
  to an interval; :func:`n_val_window` reports it and the run asserts the value
  used lies inside.  On ``rebuttal_batchsize_polynomial_scan`` (B from 8 to 256)
  the interval is 9993-10000, i.e. the assumed 10000 is essentially forced.

    analysis/repair_legacy.py toy_1d_scan polynomial_scan        # writes + prints
    python -c "import repair_legacy as R; R.repair_scan('toy_1d_scan')"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:       # importable as `repair_legacy` from anywhere
    sys.path.insert(0, str(_HERE))

import style  # noqa: E402

#: Where the repaired columns go.  Deliberately NOT under the results root.
OUT_DIR = _HERE / 'legacy_repair'

#: Validation-set sizes of the legacy scans, from the dataset classes and the scan
#: configs as they stood at the campaign branch point (``git show HEAD:``):
#: ``Toy1DRegressionDataset(n_val=10_000)`` and ``RandomPolynomialDataset(n_val=10_000)``
#: (also set explicitly by ``rebuttal_overparam_polynomial_scan.yaml``), MNIST and
#: CIFAR-10 validate on the official 10k test split.  Records written after
#: 2026-09-17 carry ``n_val`` themselves and are used in preference to this table;
#: a scan that is in neither needs ``n_val=`` (and every value is checked against
#: :func:`n_val_window`).
LEGACY_N_VAL = {
    'toy_1d_scan': 10_000,
    'polynomial_scan': 10_000,
    'mnist_scan_ce': 10_000,
    'mnist_scan_labelRegression': 10_000,
    'cifar10_resnet_ce_scan': 10_000,
    'cifar10_resnet_scan_labelRegression': 10_000,
    'rebuttal_batchsize_polynomial_scan': 10_000,
    'rebuttal_overparam_polynomial_scan': 10_000,
}

#: The six headline scans plus the two large-/full-batch studies where Fable saw
#: individual runs move by 10-30%.
HEADLINE = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_ce', 'mnist_scan_labelRegression',
            'cifar10_resnet_ce_scan', 'cifar10_resnet_scan_labelRegression')
DEFAULT_SCANS = HEADLINE + ('rebuttal_batchsize_polynomial_scan',
                            'rebuttal_overparam_polynomial_scan')


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
def n_batches(n_val, batch_size):
    """Batches the validation loader produced: ``ceil(n_val / B)`` (never dropped)."""
    return int(math.ceil(int(n_val) / int(batch_size)))


def batch_weights(n_val, batch_size):
    """Examples per validation batch: ``[B, ..., B, n_val - (nb-1) B]``."""
    n_val, B = int(n_val), int(batch_size)
    w = np.full(n_batches(n_val, B), float(B))
    w[-1] = n_val - B * (len(w) - 1)
    return w


def n_val_window(pairs):
    """``(lo, hi)``: the ``n_val`` values consistent with every ``(batch_size,
    n_batches)`` pair observed in a scan, since ``ceil(n_val/B) == nb`` means
    ``(nb-1) B < n_val <= nb B``.  ``(None, None)`` for no pairs."""
    los, his = [], []
    for B, nb in pairs:
        los.append((int(nb) - 1) * int(B) + 1)
        his.append(int(nb) * int(B))
    return (max(los), min(his)) if los else (None, None)


def repair_curve(val_batch, n_val, batch_size):
    """``(equal_weight, example_weight, n_batches)`` curves from one run's
    concatenated per-batch validation losses.

    The array is ``n_evals * nb`` long; the equal-weight curve is what the run
    recorded, and reproducing it is the proof that this reshape is correct.
    """
    v = np.asarray(val_batch, dtype=np.float64).ravel()
    nb = n_batches(n_val, batch_size)
    if nb <= 0 or v.size == 0 or v.size % nb:
        raise ValueError(f'val_batch of {v.size} is not a whole number of {nb}-batch epochs '
                         f'(n_val={n_val}, batch_size={batch_size})')
    blocks = v.reshape(-1, nb)
    w = batch_weights(n_val, batch_size)
    return blocks.mean(axis=1), blocks @ w / w.sum(), nb


def _rel(new, old):
    """Relative correction ``|new - old| / |old|``, NaN where ``old`` is not usable."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.abs(new - old) / np.abs(old)
    return np.where(np.isfinite(r), r, np.nan)


def _nanmax(values):
    """``np.nanmax`` that returns NaN (quietly) for an all-NaN series: a run that
    diverged has no relative correction to report, and that is not a warning."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else np.nan


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _val_batch(row, scan_dir):
    """One run's ``val_batch``, from its npz (reading only that member) or, for the
    old inline layout, from its JSONL."""
    diag = row.get('diag_file')
    if diag and isinstance(diag, str):
        with np.load(Path(scan_dir) / diag) as z:
            if 'val_batch' not in z.files:
                return None
            return z['val_batch']
    f = Path(scan_dir) / f"{row['run_id']}.jsonl"
    if not f.is_file():
        return None
    with open(f) as fh:
        rec = json.loads(fh.readline())
    return (rec.get('losses') or {}).get('val_batch')


def repair_run(row, scan_dir, n_val):
    """Repaired columns for one run, or ``{'error': ...}`` if it cannot be repaired.

    ``val_ew`` is INDEX-ALIGNED with the stored ``val`` curve: ``len(val_ew) ==
    len(losses['val'])`` and ``val_ew[i]`` repairs ``val[i]``.  Every legacy scan has
    one leading, pre-training entry which was never stored per batch (the blocks are
    matched to the stored curve's TAIL, so both layouts work); it is NaN in ``val_ew``
    and ``pre_training_eval`` says whether there is one.  Plot ``val_ew`` on the same
    epoch axis as ``val``, not shifted.
    """
    out = {'run_id': row['run_id'], 'batch_size': row.get('batch_size'), 'n_val': int(n_val)}
    try:
        vb = _val_batch(row, scan_dir)
        if vb is None:
            raise ValueError('no val_batch stored')
        eq, ew, nb = repair_curve(vb, n_val, row['batch_size'])
        stored = np.asarray((row.get('losses') or {}).get('val', []), dtype=float)
        offset = stored.size - eq.size
        if offset not in (0, 1):
            raise ValueError(f'{eq.size} recomputed epochs against a stored curve of {stored.size}')
        ref = stored[offset:]
        # Padded back to the stored curve's length: a consumer overlays val_ew on the
        # epoch axis built from losses['val'], and an off-by-one there is invisible.
        aligned = np.concatenate([np.full(offset, np.nan), ew])
        assert aligned.size == stored.size, f'{aligned.size} against a stored {stored.size}'
        out.update({
            'n_val_batches': nb, 'n_evals': int(eq.size), 'pre_training_eval': bool(offset),
            'val_ew': aligned.tolist(),
            'final_val_loss': float(ref[-1]), 'final_val_loss_ew': float(ew[-1]),
            'rel_corr_final': float(_rel(ew[-1], ref[-1])),
            'rel_corr_max': _nanmax(_rel(ew, ref)),
            # The equal-weight mean of the same blocks must BE the stored curve.
            'eq_check': _nanmax(_rel(eq, ref)),
            'n_nonfinite': int((~np.isfinite(np.asarray(vb, float))).sum()),
        })
    except Exception as exc:                        # a broken run is data, not a crash
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------
def resolve_root(root=None):
    """The results root as an ABSOLUTE path: ``style.resolve_results_root`` plus the
    fact that its default (``'../experiment_results'``) is relative to ``analysis/``,
    while this module is also run as a script from the repo root."""
    p = Path(style.resolve_results_root(root))
    return p if p.is_absolute() else (_HERE / p).resolve()


def resolve_n_val(name, df, n_val=None):
    """``n_val`` for a scan: the argument, else the value the records carry
    (schema 2), else :data:`LEGACY_N_VAL`."""
    if n_val is not None:
        return int(n_val)
    if 'n_val' in df.columns:
        vals = {int(v) for v in df['n_val'].dropna().unique()}
        if len(vals) == 1:
            return vals.pop()
        if len(vals) > 1:
            raise ValueError(f'{name}: records disagree about n_val: {sorted(vals)}')
    if name in LEGACY_N_VAL:
        return LEGACY_N_VAL[name]
    raise KeyError(f'{name}: no n_val on the records and none in LEGACY_N_VAL; pass n_val=')


def repair_scan(name, results_root=None, n_val=None, workers=16, check=5, rtol=2e-5,
                cache_dir=None, verbose=True) -> pd.DataFrame:
    """Repair every run of one scan.  One row per run, ``attrs`` carry the checks.

    ``workers`` threads read the npz files (Lustre is latency- not bandwidth-bound:
    ~1 GB over the eight scans, but tens of thousands of round trips).  ``check``
    runs PER BATCH SIZE are asserted to reproduce their stored curve under equal
    weighting before the repair is believed (``attrs['n_checked']`` counts them; the
    whole-scan maximum is in ``attrs['eq_check_max']`` either way).
    """
    root = resolve_root(results_root)
    scan_dir = root / name
    # The slim cache goes next to the OUTPUT: the legacy root stays untouched.
    df = style.load_results(name, results_root=str(root), slim=True,
                            cache_dir=str(cache_dir or (OUT_DIR / '_cache')))
    n_val = resolve_n_val(name, df, n_val)
    rows = df.to_dict('records')
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        out = list(pool.map(lambda r: repair_run(r, scan_dir, n_val), rows))
    res = pd.DataFrame(out)
    for col in ('error', 'n_val_batches', 'n_evals', 'val_ew', 'final_val_loss',
                'final_val_loss_ew', 'rel_corr_final', 'rel_corr_max', 'eq_check'):
        if col not in res.columns:      # every run failed (or none did): keep the schema
            res[col] = np.nan
    keep = [c for c in ('_scan', 'optimizer', 'model_seed', 'lr', 'k', 'rtol', 'n_data', 'n_train')
            if c in df.columns]
    res = res.merge(df[['run_id', *keep]], on='run_id', how='left')
    if '_scan' not in res.columns:
        res['_scan'] = name

    ok = res[res['error'].isna()] if 'error' in res.columns else res
    pairs = sorted({(int(b), int(nb)) for b, nb in zip(ok['batch_size'], ok['n_val_batches'])}) \
        if len(ok) else []
    lo, hi = n_val_window(pairs)
    if lo is not None and not (lo <= n_val <= hi):
        raise AssertionError(f'{name}: n_val={n_val} is outside the window {lo}-{hi} implied by '
                             f'(batch_size, n_batches) {pairs}')
    # Believe the repair only once the equal-weight recomputation reproduces the record.
    # A run whose stored curve is all non-finite (it diverged) has eq_check = NaN and
    # checks nothing -- NaN > rtol is False -- so the sample is drawn from the runs that
    # DID produce a comparison, and every batch size present must contribute one: the
    # reshape into epochs x batches depends on (n_val, batch_size).  ``check=0`` skips it.
    n_checked = 0
    if int(check) > 0 and len(ok):
        cand = ok[pd.to_numeric(ok['eq_check'], errors='coerce').notna()]
        blank = sorted(set(ok['batch_size']) - set(cand['batch_size']))
        if not len(cand) or blank:
            raise AssertionError(f'{name}: no run could be checked against its stored curve '
                                 f'(batch sizes {blank or "all"} have none with a finite '
                                 f'validation curve): the repair is unverified')
        sample = cand.sort_values('run_id').groupby('batch_size', dropna=False).head(int(check))
        bad = sample[sample['eq_check'] > rtol]
        if len(bad):
            raise AssertionError(f'{name}: equal-weight recomputation does not reproduce the stored '
                                 f'curve for {list(bad.run_id)} (max rel. dev '
                                 f'{bad.eq_check.max():.2e} > {rtol:.1e})')
        n_checked = len(sample)
    res.attrs.update(scan=name, n_val=n_val, n_val_window=(lo, hi), batch_pairs=pairs,
                     eq_check_max=_nanmax(ok['eq_check']) if len(ok) else np.nan,
                     n_checked=n_checked, n_errors=int(len(res) - len(ok)))
    if verbose:
        print(f'[repair] {name}: {len(ok)}/{len(res)} runs repaired, n_val={n_val} '
              f'(window {lo}-{hi}), eq_check_max={res.attrs["eq_check_max"]:.2e} '
              f'over {n_checked} asserted run(s)')
    return res


def correction_summary(res) -> pd.DataFrame:
    """Median / 95th percentile / max relative correction of the FINAL validation
    loss, per scan -- the table to put beside Fable's (FABLE_CRITIQUES section 8)."""
    rows = []
    for name, d in res.groupby('_scan', dropna=False, sort=False):
        r = np.asarray(d['rel_corr_final'], dtype=float)
        r = r[np.isfinite(r)]
        bs = sorted({int(b) for b in d['batch_size'].dropna().unique()})
        rows.append({'scan': name, 'n_runs': len(d), 'n_scored': r.size,
                     'batch_sizes': ','.join(str(b) for b in bs),
                     'median_%': 100 * np.median(r) if r.size else np.nan,
                     'p95_%': 100 * np.percentile(r, 95) if r.size else np.nan,
                     'max_%': 100 * r.max() if r.size else np.nan,
                     'eq_check_max': _nanmax(d['eq_check']) if 'eq_check' in d else np.nan,
                     'n_errors': int(d['error'].notna().sum()) if 'error' in d else 0})
    return pd.DataFrame(rows)


def write(res, out_dir=None, name=None):
    """Write the repaired columns under :data:`OUT_DIR` (parquet, csv fallback)."""
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or res.attrs.get('scan') or 'repair'
    try:
        path = out_dir / f'{name}.parquet'
        res.to_parquet(path, index=False)
    except Exception as exc:                        # no pyarrow -> csv with JSON curves
        print(f'[repair] parquet unavailable ({type(exc).__name__}: {exc}); writing csv')
        path = out_dir / f'{name}.csv'
        flat = res.copy()
        flat['val_ew'] = [json.dumps(v) if isinstance(v, list) else v for v in flat['val_ew']]
        flat.to_csv(path, index=False)
    return path


def load_repair(name, out_dir=None) -> pd.DataFrame:
    """Read back one scan's repaired columns (for a notebook join on ``run_id``)."""
    out_dir = Path(out_dir or OUT_DIR)
    p = out_dir / f'{name}.parquet'
    if p.is_file():
        return pd.read_parquet(p)
    d = pd.read_csv(out_dir / f'{name}.csv')
    d['val_ew'] = [json.loads(v) if isinstance(v, str) else v for v in d['val_ew']]
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('scans', nargs='*', default=list(DEFAULT_SCANS),
                    help=f'scan directories (default: {" ".join(DEFAULT_SCANS)})')
    ap.add_argument('--root', default=None, help='results root (default $SV3_RESULTS_ROOT)')
    ap.add_argument('--out', default=None, help=f'output directory (default {OUT_DIR})')
    ap.add_argument('--n-val', type=int, default=None, help='override n_val for every scan')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--check', type=int, default=5, help='runs to assert the equal-weight check on')
    args = ap.parse_args(argv)

    summaries = []
    for name in (args.scans or list(DEFAULT_SCANS)):
        res = repair_scan(name, results_root=args.root, n_val=args.n_val,
                          workers=args.workers, check=args.check)
        print(f'[repair] wrote {write(res, args.out, name)}')
        summaries.append(correction_summary(res))
    out_dir = Path(args.out or OUT_DIR)
    summary = pd.concat(summaries, ignore_index=True)
    # Several invocations are needed to stay inside the CPU-test time limit, so the
    # summary of the scans this call did REPLACES their rows and keeps the others.
    csv = out_dir / 'corrections.csv'
    if csv.is_file():
        old = pd.read_csv(csv)
        summary = pd.concat([old[~old['scan'].isin(summary['scan'])], summary], ignore_index=True)
    summary.to_csv(csv, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
