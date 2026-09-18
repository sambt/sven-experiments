"""Loading + plotting for the optimizer memory / step-time profile.

Input: the per-configuration JSON files written by ``experiments/optimizer_profile.py``
under ``profile_results_v2/<config>/``.  :func:`load_profiles` flattens them into ONE tidy
DataFrame (a row per configuration); every plot function here takes that frame, so the
notebooks are thin drivers.

Conventions
    step_ms      steady-state mean step time (start-up steps and >3 MAD spikes dropped)
    peak_mb      max over measured steps of ``torch.cuda.max_memory_allocated``
    overhead_mb  peak_mb minus the model's own parameter memory
    rel_time     step_ms / the reference first-order method (Adam, else AdamW) at the SAME
                 architecture, batch size and width;  rel_mem = peak_mb / SGD (SGD only)
    status       ok | oom | infeasible | error  -- non-ok rows are kept and drawn as markers
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import METHOD_COLORS, method_color

RESULTS_ROOT = '../profile_results_v2'
MB = 1e6

SVEN = ['gram_hooks', 'gram_full', 'gram_chunked', 'classic']
LABELS = {
    'gram_hooks': 'Sven (Gram, hooks)', 'gram_full': 'Sven (Gram, full $J$)',
    'gram_chunked': 'Sven (Gram, chunked)', 'classic': r'Sven (classic, rand. SVD)',
    'JD': 'JD (UPGrad)', 'LBFGS1': 'LBFGS (1 it.)', 'LBFGS3': 'LBFGS (3 it.)',
}
# Baselines take the global optimizer colours (style.METHOD_COLORS).  The Sven variants
# only ever appear side by side here: the default backend (Gram, hooks) is Sven's black,
# the others magenta / teal / brown, outside the baseline hues.  LBFGS (3 it.) is a
# darker shade of the LBFGS grey.
COLORS = {
    'gram_hooks': method_color('Sven'), 'gram_full': '#C2185B', 'gram_chunked': '#00796B', 'classic': '#A0522D',
    **{m: c for m, c in METHOD_COLORS.items() if m != 'Sven'},
    'LBFGS1': method_color('LBFGS'), 'LBFGS3': '#4D4D4D',
}
MARKERS = {'gram_hooks': 'o', 'gram_full': 's', 'gram_chunked': 'D', 'classic': '^'}
ARCH_TITLES = {'toy_1d': 'Toy 1D MLP', 'polynomial': 'Polynomial MLP', 'mnist': 'MNIST MLP',
               'mnist_width': 'MNIST MLP (width sweep)', 'cifar_resnet18': 'CIFAR-10 ResNet18',
               'nanogpt': 'nanoGPT', 'nanogpt_width': 'nanoGPT (width sweep)'}
ARCH_ORDER = ['toy_1d', 'polynomial', 'mnist', 'nanogpt', 'cifar_resnet18']


def label(m): return LABELS.get(m, m)
def color(m): return COLORS[m] if m in COLORS else method_color(m)
def is_sven(m): return m in SVEN


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _row(r: dict) -> dict:
    p, meta = r.get('params', {}), r.get('meta') or {}
    t, mem = r.get('time') or {}, r.get('memory') or {}
    st = t.get('step_ms') or {}
    g = lambda d, k: (d.get(k) if d else None)
    out = {
        'run_id': r['run_id'], 'arch': r.get('arch'), 'config_name': r.get('config_name'), 'study': r['study'],
        'method': r['method'], 'family': 'sven' if is_sven(r['method']) else 'baseline', 'status': r['status'],
        'error': r.get('error'), 'n_params': r.get('n_params'), 'B': p.get('batch_size'),
        'rows': meta.get('rows'), 'k': meta.get('k'), 'k_fraction': p.get('k_fraction'),
        'pf': p.get('param_fraction', 1.0), 'mask_mode': p.get('mask_mode') or 'none',
        'mb': p.get('microbatch_size', 1), 'chunk_fraction': p.get('chunk_fraction'),
        'chunk_numel': meta.get('chunk_numel'), 'n_groups': meta.get('n_groups'), 'width': p.get('width'),
        'step_ms': st.get('steady_mean'), 'step_ms_median': st.get('median'), 'step_ms_trimmed': st.get('trimmed_mean'),
        'step_ms_mean': st.get('mean'), 'step_ms_p10': st.get('p10'), 'step_ms_p90': st.get('p90'), 'n_steps': st.get('n'),
        'wall_ms': g(t.get('wall_ms'), 'steady_mean'), 'capture_ms': g(t.get('capture_ms'), 'steady_mean'),
        'solve_ms': g(t.get('solve_ms'), 'steady_mean'),
        'model_mb': (r.get('baseline_model_bytes') or 0) / MB,
        'peak_mb': (mem.get('peak_alloc_bytes_max') or np.nan) / MB,
        'peak_reserved_mb': (mem.get('peak_reserved_bytes_max') or np.nan) / MB,
        'resident_mb': (mem.get('resident_bytes_mean') or np.nan) / MB,
        'peak_capture_mb': (mem.get('peak_capture_bytes_max') or np.nan) / MB,
        'analytic_jac_mb': (meta.get('analytic_jacobian_bytes') or np.nan) / MB,
        'analytic_gram_mb': (meta.get('analytic_gram_bytes') or np.nan) / MB,
        'oom_peak_mb': (r.get('peak_alloc_bytes_at_failure') or np.nan) / MB,
        'gpu': (r.get('env') or {}).get('gpu'), 'gpu_total_mb': ((r.get('env') or {}).get('gpu_total_bytes') or np.nan) / MB,
        '_path': r.get('_path'),
    }
    out['overhead_mb'] = out['peak_mb'] - out['model_mb']
    out['steadiness'] = (out['step_ms_p90'] / out['step_ms_p10']) if out['step_ms_p10'] else np.nan
    return out


def load_profiles(root=RESULTS_ROOT, configs=None, use_cache=True) -> pd.DataFrame:
    """Every profile JSON under ``root`` as one tidy frame (cached on the file listing)."""
    root = Path(root)
    files = sorted(f for f in root.glob('*/*.json') if configs is None or f.parent.name in configs)
    sig = (len(files), max((f.stat().st_mtime_ns for f in files), default=0), tuple(configs or ()))
    cache = root / '_profile_cache.pkl'
    if use_cache and cache.is_file():
        try:
            blob = pickle.loads(cache.read_bytes())
            if blob['sig'] == sig:
                return blob['df']
        except Exception:
            pass
    rows = []
    for f in files:
        r = json.loads(f.read_text()); r['_path'] = str(f)
        rows.append(_row(r))
    df = add_relative(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if use_cache and rows:
        cache.write_bytes(pickle.dumps({'sig': sig, 'df': df}))
    print(f'{len(df)} configurations from {root}/ ({df["status"].value_counts().to_dict() if len(df) else {}})')
    return df


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """rel_time (vs Adam, else AdamW) and rel_mem (vs SGD) at matching arch / B / width."""
    df = df.copy()
    key = ['arch', 'B', 'width']
    ok = df[(df.status == 'ok') & (df.pf == 1.0) & (df.mb == 1)]
    def ref(methods, col):
        out = {}
        for m in methods:
            for k_, v in ok[ok.method == m].groupby(key, dropna=False)[col].mean().items():
                out.setdefault(k_, v)
        return out
    # rel_mem is "/ SGD" and only SGD: a silent fallback to Adam made the label a lie
    rt, rm = ref(['Adam', 'AdamW'], 'step_ms'), ref(['SGD'], 'peak_mb')
    keys = list(zip(df.arch, df.B, df.width))
    norm = lambda k_: tuple(None if (isinstance(x, float) and np.isnan(x)) else x for x in k_)
    rt = {norm(k_): v for k_, v in rt.items()}; rm = {norm(k_): v for k_, v in rm.items()}
    df['rel_time'] = [r / rt.get(norm(k_), np.nan) if r == r else np.nan for r, k_ in zip(df.step_ms, keys)]
    df['rel_mem'] = [r / rm.get(norm(k_), np.nan) if r == r else np.nan for r, k_ in zip(df.peak_mb, keys)]
    return df


def raw_steps(row, which='step_ms'):
    """The raw per-step series of one configuration (read back from its JSON)."""
    return np.asarray(json.loads(Path(row['_path']).read_text())['raw'].get(which, []), dtype=float)


def legend_below(fig, ax=None, ncol=4, fontsize=9):
    """One shared legend under the figure (keeps it off the data and the failure markers).

    Collects the handles of EVERY axes in the figure (``ax`` is accepted for backwards
    compatibility and ignored), so a method or failure marker that only appears in a
    later panel still gets an entry.
    """
    seen = {}
    for a in fig.axes:
        h, l = a.get_legend_handles_labels()
        for label, handle in zip(l, h):
            seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               fontsize=fontsize, frameon=False)


def save(fig, name, plot_dir):
    plot_dir = Path(plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / name, bbox_inches='tight')
    return plot_dir / name


def _fmt(v, unit=''):
    if v != v: return '--'
    return (f'{v:.3g}' if v < 100 else f'{v:,.0f}') + unit


# ---------------------------------------------------------------------------
# Study 1: method comparison
# ---------------------------------------------------------------------------
def method_table(df, arch, study='methods') -> pd.DataFrame:
    """One row per method at the set point: time, memory, ratios, status."""
    d = df[(df.arch == arch) & (df.study == study)].copy()
    d['Method'] = d.method.map(label)
    note = d.apply(lambda r: '' if r.status == 'ok' else
                   (f'OOM' if r.status == 'oom' else 'infeasible' if r.status == 'infeasible' else 'error'), axis=1)
    out = pd.DataFrame({
        'Method': d.Method, 'Step (ms)': d.step_ms.round(2), 'x Adam': d.rel_time.round(2),
        'Peak mem (MB)': d.peak_mb.round(1), 'x SGD': d.rel_mem.round(2),
        'Capture (ms)': d.capture_ms.round(2), 'Solve+apply (ms)': d.solve_ms.round(2),
        'p90/p10': d.steadiness.round(2), 'Note': note})
    out['_s'] = d.family.map({'sven': 0, 'baseline': 1}).values
    return out.sort_values(['_s', 'Step (ms)'], na_position='last').drop(columns='_s').reset_index(drop=True)


def plot_method_bars(df, arch, ax, value='step_ms', study='methods', log=True):
    """Horizontal bars, one per method, sorted; non-ok configurations are annotated."""
    d = df[(df.arch == arch) & (df.study == study)].copy()
    d = d.sort_values(value, ascending=True, na_position='first')
    y = np.arange(len(d))
    vals = d[value].fillna(0).values
    ax.barh(y, vals, color=[color(m) for m in d.method], edgecolor='k', lw=0.4)
    ax.set_yticks(y); ax.set_yticklabels([label(m) for m in d.method], fontsize=9)
    for yi, (_, r) in zip(y, d.iterrows()):
        if r.status != 'ok':
            ax.text(ax.get_xlim()[0] if not log else max(vals[vals > 0].min() if (vals > 0).any() else 1, 1e-3),
                    yi, {'oom': ' OOM', 'infeasible': ' infeasible', 'error': ' n/a'}[r.status], va='center', fontsize=8,
                    color='crimson')
        else:
            ax.text(r[value], yi, ' ' + _fmt(r[value]), va='center', fontsize=8)
    if log: ax.set_xscale('log')
    ax.set_xlabel({'step_ms': 'Steady-state step time (ms)', 'peak_mb': 'Peak GPU memory (MB)',
                   'overhead_mb': 'Peak memory above model (MB)', 'rel_time': r'Step time / Adam',
                   'rel_mem': 'Peak memory / SGD'}.get(value, value))
    ax.set_title(ARCH_TITLES.get(arch, arch)); ax.grid(axis='x', ls='--', alpha=0.5)


def plot_heatmap(df, ax, value='rel_time', study='methods', archs=None, methods=None, fmt='{:.1f}'):
    """methods x architectures grid of a relative cost; OOM / infeasible / n/a cells are labelled."""
    d = df[df.study == study]
    archs = archs or [a for a in ARCH_ORDER if a in set(d.arch)]
    methods = methods or (SVEN + [m for m in COLORS if m not in SVEN])
    methods = [m for m in methods if m in set(d.method)]
    M = np.full((len(methods), len(archs)), np.nan); txt = [['' for _ in archs] for _ in methods]
    for i, m in enumerate(methods):
        for j, a in enumerate(archs):
            r = d[(d.method == m) & (d.arch == a)]
            if r.empty: txt[i][j] = '--'; continue
            r = r.iloc[0]
            if r.status == 'ok': M[i, j] = r[value]; txt[i][j] = fmt.format(r[value])
            else: txt[i][j] = {'oom': 'OOM', 'infeasible': 'inf.', 'error': 'n/a'}[r.status]
    im = ax.imshow(np.log10(M), cmap='viridis_r', aspect='auto')
    for i in range(len(methods)):
        for j in range(len(archs)):
            ax.text(j, i, txt[i][j], ha='center', va='center', fontsize=8,
                    color='crimson' if np.isnan(M[i, j]) else 'w' if np.log10(M[i, j]) > np.nanmean(np.log10(M)) else 'k')
    ax.set_xticks(range(len(archs))); ax.set_xticklabels([ARCH_TITLES.get(a, a) for a in archs], rotation=25, ha='right')
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([label(m) for m in methods], fontsize=9)
    return im


# ---------------------------------------------------------------------------
# Sweeps (batch size, chunk fraction, param fraction, microbatch, k, width)
# ---------------------------------------------------------------------------
def plot_sweep(df, arch, study, x, ax, value='step_ms', methods=None, style_by=None, logx=True, logy=True,
               analytic=None, ideal=None):
    """``value`` vs ``x`` for one study, a line per method (and per ``style_by`` value, e.g. mask_mode).

    Non-ok points are drawn as red crosses at the top of the axis so OOM walls are visible.
    ``analytic``: a column to overlay as a dotted line (e.g. 'analytic_jac_mb').
    ``ideal``: 'linear' draws the proportional-to-x reference through each method's x=max point.
    """
    d = df[(df.arch == arch) & (df.study == study)]
    methods = methods or [m for m in (SVEN + list(COLORS)) if m in set(d.method)]
    methods = list(dict.fromkeys(methods))
    ls_cycle = ['-', '--', ':', '-.']
    n_bad_rows = 0
    for m in methods:
        dm = d[d.method == m]
        groups = [(None, dm)] if style_by is None else list(dm.groupby(style_by, dropna=False))
        for gi, (gval, g) in enumerate(groups):
            if style_by is not None and gval == 'none':
                continue
            if style_by is not None:       # the unmasked pf=1 point belongs to every mask_mode line
                g = pd.concat([g, dm[dm[style_by] == 'none']])
            g = g.sort_values(x)
            ok = g[g.status == 'ok']
            lab = label(m) + (f' [{gval}]' if style_by is not None else '')
            ax.plot(ok[x], ok[value], marker=MARKERS.get(m, 'o'), ms=5, lw=2.2 if is_sven(m) else 1.4,
                    ls=ls_cycle[gi % 4] if style_by is not None else '-', color=color(m), label=lab)
            bad = g[g.status != 'ok']
            if len(bad):   # failures: method-coloured X with a red edge, stacked above the data
                top = (np.nanmax(d[value]) if d[value].notna().any() else 1) * (1.6 * 1.35 ** n_bad_rows)
                n_bad_rows += 1
                ax.plot(bad[x], [top] * len(bad), 'X', color=color(m), mec='crimson', mew=1.2, ms=9,
                        label=f'{label(m)}: ' + '/'.join(sorted(set(bad.status))))
            if analytic and ok[analytic].notna().any():
                ax.plot(ok[x], ok[analytic], ls=':', lw=1, color=color(m))
            if ideal == 'linear' and len(ok) > 1:
                x0, y0 = ok[x].iloc[-1], ok[value].iloc[-1]
                ax.plot(ok[x], y0 * ok[x] / x0, ls=':', lw=1, color=color(m), alpha=0.6)
    if logx: ax.set_xscale('log', base=2)
    if logy: ax.set_yscale('log')
    ax.set_xlabel({'B': 'Batch size $B$', 'chunk_fraction': 'Chunk size / $P$', 'pf': 'Parameter fraction',
                   'mb': 'Micro-batch size', 'k_fraction': '$k$ / rows', 'n_params': 'Parameters $P$'}.get(x, x))
    ax.set_ylabel({'step_ms': 'Step time (ms)', 'peak_mb': 'Peak GPU memory (MB)', 'overhead_mb': 'Peak memory above model (MB)',
                   'capture_ms': 'Capture time (ms)', 'solve_ms': 'Solve + apply time (ms)'}.get(value, value))
    ax.set_title(ARCH_TITLES.get(arch, arch)); ax.grid(True, which='both', ls='--', alpha=0.4)


def plot_pareto(df, arch, ax):
    """Chunk-fraction trade-off: peak memory vs step time, annotated with the chunk fraction."""
    d = df[(df.arch == arch) & (df.study == 'chunk_fraction') & (df.status == 'ok')].sort_values('chunk_fraction')
    ax.plot(d.peak_mb, d.step_ms, '-D', color=color('gram_chunked'), lw=2, label=label('gram_chunked'))
    for _, r in d.iterrows():
        ax.annotate(f"$f$={r.chunk_fraction:g}\n({int(r.n_groups)} grp)", (r.peak_mb, r.step_ms), fontsize=8,
                    textcoords='offset points', xytext=(6, 4))
    ref = df[(df.arch == arch) & (df.study == 'methods') & (df.status == 'ok')]
    for m in ['gram_hooks', 'gram_full', 'classic', 'Adam']:
        r = ref[ref.method == m]
        if len(r): ax.plot(r.peak_mb, r.step_ms, MARKERS.get(m, '*'), ms=10, color=color(m), label=label(m), mec='k')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Peak GPU memory (MB)'); ax.set_ylabel('Step time (ms)')
    ax.set_title(ARCH_TITLES.get(arch, arch) + ': chunk-size trade-off'); ax.grid(True, which='both', ls='--', alpha=0.4)


def plot_phase_bars(df, ax, archs=None, study='methods'):
    """Stacked capture vs solve+apply time for each Sven variant, grouped by architecture."""
    d = df[(df.study == study) & (df.family == 'sven') & (df.status == 'ok')]
    archs = archs or [a for a in ARCH_ORDER if a in set(d.arch)]
    xs, labs = [], []; x = 0
    for a in archs:
        for m in SVEN:
            r = d[(d.arch == a) & (d.method == m)]
            if r.empty: continue
            r = r.iloc[0]
            ax.bar(x, r.capture_ms, color=color(m), edgecolor='k', lw=0.4)
            ax.bar(x, r.solve_ms, bottom=r.capture_ms, color=color(m), alpha=0.35, hatch='//', edgecolor='k', lw=0.4)
            xs.append(x); labs.append(f"{label(m).replace('Sven ', '')}"); x += 1
        x += 0.8
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=60, ha='right', fontsize=8)
    ax.set_yscale('log'); ax.set_ylabel('Time per step (ms)')
    ax.bar(0, 0, color='grey', label='capture (loss + Jacobian/Gram)'); ax.bar(0, 0, color='grey', alpha=0.35, hatch='//', label='solve + apply')
    ax.legend(fontsize=9)


def fit_exponent(x, y):
    """Slope of log y vs log x over the upper half of the range (the asymptotic regime)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 3: return np.nan
    o = np.argsort(x); x, y = x[o][len(x) // 2 - 1:], y[o][len(y) // 2 - 1:]
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def scaling_table(df, arch, value='step_ms') -> pd.DataFrame:
    """Per method: value at the smallest / largest model, the growth factor and the fitted exponent in P."""
    d = df[(df.arch == arch) & (df.study == 'width')]
    rows = []
    for m, g in d.groupby('method'):
        ok = g[g.status == 'ok'].sort_values('n_params')
        bad = g[g.status != 'ok'].sort_values('n_params')
        rows.append({'Method': label(m), 'smallest P': _fmt(ok[value].iloc[0]) if len(ok) else '--',
                     'largest ok P': f"{int(ok.n_params.iloc[-1]):,}" if len(ok) else '--',
                     'at largest': _fmt(ok[value].iloc[-1]) if len(ok) else '--',
                     'growth': round(ok[value].iloc[-1] / ok[value].iloc[0], 1) if len(ok) > 1 else np.nan,
                     'exponent': round(fit_exponent(ok.n_params, ok[value]), 2),
                     'first failure': (f"{bad.status.iloc[0]} @ P={int(bad.n_params.iloc[0]):,}" if len(bad) and bad.n_params.notna().any()
                                       else (bad.status.iloc[0] if len(bad) else ''))})
    return pd.DataFrame(rows).sort_values('exponent').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def plot_steadiness(df, ax, threshold=1.3):
    """p90/p10 of the per-step times for every ok configuration; flags the unsteady ones."""
    d = df[df.status == 'ok'].sort_values('steadiness')
    ax.plot(np.arange(len(d)), d.steadiness, '.', ms=3, color='k')
    ax.axhline(threshold, color='crimson', ls='--', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('configuration (sorted)'); ax.set_ylabel('step time p90 / p10')
    bad = d[d.steadiness > threshold]
    ax.set_title(f'{len(bad)} of {len(d)} configurations above {threshold}')
    return bad[['arch', 'study', 'method', 'B', 'step_ms', 'step_ms_p10', 'step_ms_p90', 'steadiness']]


def plot_raw_steps(row, ax):
    """Raw per-step times of one configuration with the summary statistics overlaid."""
    s = raw_steps(row)
    ax.plot(s, '.-', color='k', lw=0.8)
    for v, c, l in [(row.step_ms, 'crimson', 'steady mean'), (row.step_ms_median, 'C0', 'median'), (row.step_ms_mean, 'C2', 'plain mean')]:
        ax.axhline(v, color=c, ls='--', lw=1, label=f'{l} {v:.2f} ms')
    ax.set_xlabel('measured step'); ax.set_ylabel('ms'); ax.legend(fontsize=8)
    ax.set_title(f"{row.arch} / {label(row.method)}")


def status_report(df) -> pd.DataFrame:
    """Every configuration that did not complete, with the reason."""
    d = df[df.status != 'ok']
    return d[['arch', 'study', 'method', 'B', 'pf', 'mask_mode', 'mb', 'width', 'status', 'error']].reset_index(drop=True)
