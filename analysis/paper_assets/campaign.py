"""``paper_assets.campaign`` -- the campaign's own totals: how many runs were trained, on
what hardware, and for how many GPU-hours.

Owned by the **Experiment Details** appendix (``iclr_manuscript/sections_v2/
app_exp_details.tex``, Sec. "Compute and Wall-Time Measurement") and by the
**Reproducibility** appendix.  Those two fragments quote four campaign-wide totals and two
per-scan costs; before this module they were typed by hand behind a ``BUILD GUARD`` comment,
which breaks as soon as a refresh adds runs -- and the parallel refresh (the CIFAR-CE
``rtol`` extension, the Fig-5 re-run at the selected configuration) does add runs.

Nothing here reads a curve or a loss: the only quantities are counts and ``wall_time_s``,
read from the FIRST line of every ``<run_id>.jsonl`` record under the results root.  A
record exists for a failed run too (``EXPERIMENTS.md`` section 7), so the totals are the
cost of the campaign as it ran and not the cost of its successes.

GPU-hours are **process time**: one run owns one device (or one MIG partition of one) for
``wall_time_s``, summed.  Short runs were packed several to a device, so the wall-clock
*allocation* is smaller than this figure -- the appendix says so, and the packing factors
are in ``campaign/stage0_reports/gpu.probe.md``.

Run it standalone (no other module's assets are touched)::

    cd analysis && ../.venv/bin/python -m paper_assets.campaign

or, once ``'campaign'`` is added to :data:`paper_assets.common.MODULES`, as part of
``python -m paper_assets``.  The macro file it writes, ``numbers_v2_campaign.tex``, is
picked up by :func:`common.write_numbers_index` only in that second case; until then the
appendix fragment ``\\input``s it itself, guarded by ``\\ifdefined``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import style

from . import common as C

GROUP = 'campaign'

#: The three result-dependent companion passes of phase 5 (``EXPERIMENTS.md`` section 4).
#: A directory whose name ends in one of these is a *pass* over an existing scan's selected
#: configurations, not a hyperparameter grid, so it is excluded from the grid count while
#: its GPU-hours still count towards the total.
PASS_SUFFIXES = ('_timing', '_diag', '_confirm')

#: A whole 80 GB device vs a 20 GB partition of a 40 GB one.  The campaign used exactly
#: these two, and the appendix describes them generically (no vendor model names appear in
#: the paper: anonymity, and they would date it).
def device_class(gpu_name):
    """``'mig'`` for a partitioned device, ``'full'`` for a whole one, ``'other'`` if the
    record does not say."""
    if not gpu_name:
        return 'other'
    return 'mig' if 'MIG' in str(gpu_name) else 'full'


def _first_line(path):
    """The record's provenance line, or ``None`` if the file is empty/truncated."""
    try:
        with open(path) as fh:
            line = fh.readline()
    except OSError:
        return None
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def collect(root=None, verbose=True):
    """One row per record: ``(scan, run_id, status, gpu_name, wall_time_s)``.

    Reads only the first line of each record, so this is ~25k small reads rather than a
    load of the curves; it takes about a minute on a quiet node.
    """
    root = Path(root or style.resolve_results_root())
    if not root.is_absolute():
        root = (C.ANALYSIS / root).resolve()
    rows, bad = [], []
    for scan_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if scan_dir.name.startswith('_'):               # _cache
            continue
        for rec_path in sorted(scan_dir.glob('*.jsonl')):
            rec = _first_line(rec_path)
            if rec is None:
                bad.append(str(rec_path.relative_to(root)))
                continue
            rows.append({
                'scan': scan_dir.name,
                'run_id': rec.get('run_id') or rec_path.stem,
                'status': rec.get('status'),
                'gpu_name': rec.get('gpu_name'),
                'device': device_class(rec.get('gpu_name')),
                'wall_time_s': float(rec.get('wall_time_s') or 0.0),
                'slurm_job_id': rec.get('slurm_job_id'),
            })
    if verbose and bad:
        print(f'[campaign] {len(bad)} unreadable record(s): {bad[:5]}')
    if verbose:
        print(f'[campaign] {len(rows)} records under {root}')
    return rows


def totals(rows):
    """The campaign account: counts, GPU-hours overall / per device class / per scan."""
    def gpu_h(subset):
        return sum(r['wall_time_s'] for r in subset) / 3600.0

    scans = sorted({r['scan'] for r in rows})
    grids = [s for s in scans if not s.endswith(PASS_SUFFIXES)]
    passes = [s for s in scans if s.endswith(PASS_SUFFIXES)]
    per_scan = {s: {'runs': sum(1 for r in rows if r['scan'] == s),
                    'gpu_h': gpu_h([r for r in rows if r['scan'] == s])}
                for s in scans}
    total_h = gpu_h(rows)
    out = {
        'runs': len(rows),
        'grid_runs': sum(1 for r in rows if not r['scan'].endswith(PASS_SUFFIXES)),
        'pass_runs': sum(1 for r in rows if r['scan'].endswith(PASS_SUFFIXES)),
        'n_grids': len(grids),
        'n_passes': len(passes),
        'gpu_h': total_h,
        'gpu_h_mig': gpu_h([r for r in rows if r['device'] == 'mig']),
        'gpu_h_full': gpu_h([r for r in rows if r['device'] == 'full']),
        'gpu_models': sorted({str(r['gpu_name']) for r in rows if r['gpu_name']}),
        'n_jobs': len({r['slurm_job_id'] for r in rows if r['slurm_job_id']}),
        'per_scan': per_scan,
        'diverged_recorded': sum(1 for r in rows if r['status'] == 'diverged'),
        'oom_or_error': sum(1 for r in rows if r['status'] in ('oom', 'error')),
    }
    # the two costs the appendix contrasts: the whole GPT-2-small comparison (29 runs) and
    # the CIFAR-10 cross-entropy grid (785), whose shares of the total are comparable
    for key, scan in (('gpt2', 'exp_gpt2_small_comparison'),
                      ('cifar_ce', 'cifar10_resnet_ce_scan')):
        got = per_scan.get(scan, {'runs': 0, 'gpu_h': 0.0})
        out[f'{key}_runs'] = got['runs']
        out[f'{key}_gpu_h'] = got['gpu_h']
        out[f'{key}_share'] = got['gpu_h'] / total_h if total_h else 0.0
    return out


def _thousands(n):
    """``24894`` -> ``24{,}894``: a TeX-safe thousands separator that keeps the digits
    together in math and in text."""
    return C.Raw(f'{int(round(n)):,}'.replace(',', '{,}'))


def _macros(acc):
    book = C.Macros(module=GROUP)
    src = 'paper_assets.campaign.totals'
    book.add('numCampaignRuns', _thousands(acc['runs']), source=src,
             note='records under the results root, failures included')
    book.add('numCampaignGridRuns', _thousands(acc['grid_runs']), source=src,
             note='the hyperparameter grids alone')
    book.add('numCampaignPassRuns', _thousands(acc['pass_runs']), source=src,
             note='the timing + diagnostics + confirmation passes')
    book.add('numCampaignGrids', C.fmt_int(acc['n_grids']), source=src,
             note='grid directories (the phase-5 passes excluded)')
    book.add('numCampaignPasses', C.fmt_int(acc['n_passes']), source=src)
    book.add('numCampaignGpuHours', _thousands(acc['gpu_h']), source=src,
             note='sum of wall_time_s, one device per process')
    book.add('numCampaignGpuHoursMig', _thousands(acc['gpu_h_mig']), source=src,
             note='on 20 GB partitions of a 40 GB device')
    book.add('numCampaignGpuHoursFull', _thousands(acc['gpu_h_full']), source=src,
             note='on whole 80 GB devices')
    book.add('numCampaignCifarCeGpuHours', _thousands(acc['cifar_ce_gpu_h']), source=src,
             note=f'{acc["cifar_ce_runs"]} runs of the CIFAR-10 CE grid')
    book.add('numCampaignCifarCeRuns', C.fmt_int(acc['cifar_ce_runs']), source=src)
    book.add('numCampaignGptTwoGpuHours', _thousands(acc['gpt2_gpu_h']), source=src,
             note=f'{acc["gpt2_runs"]} runs of the GPT-2-small comparison')
    return book


def build(root=None, figures=True, tables=True, numbers=True, dry_run=False):
    """The module protocol shared by every ``paper_assets`` module."""
    C.banner(GROUP, f'root={root or "default"}')
    rows = collect(root)
    if not rows:
        print('[campaign] no records found -- writing nothing')
        return {'status': 'no records', 'figures': (), 'tables': (), 'n_macros': 0}
    acc = totals(rows)
    print(f'[campaign] {acc["runs"]} runs  '
          f'({acc["grid_runs"]} on {acc["n_grids"]} grids + '
          f'{acc["pass_runs"]} in {acc["n_passes"]} passes)')
    print(f'[campaign] {acc["gpu_h"]:.0f} GPU-h total = '
          f'{acc["gpu_h_mig"]:.0f} partitioned + {acc["gpu_h_full"]:.0f} whole-device')
    print(f'[campaign] devices: {acc["gpu_models"]}; {acc["n_jobs"]} scheduler jobs')
    print(f'[campaign] recorded diverged {acc["diverged_recorded"]}, '
          f'oom/error {acc["oom_or_error"]}')
    print(f'[campaign] GPT-2-small {acc["gpt2_runs"]} runs = {acc["gpt2_gpu_h"]:.0f} GPU-h '
          f'({100 * acc["gpt2_share"]:.1f}% of the total); '
          f'CIFAR-10 CE grid {acc["cifar_ce_runs"]} runs = '
          f'{acc["cifar_ce_gpu_h"]:.0f} GPU-h ({100 * acc["cifar_ce_share"]:.1f}%)')
    print('\n--- GPU-hours per scan ---')
    for scan, got in sorted(acc['per_scan'].items(),
                            key=lambda kv: -kv[1]['gpu_h']):
        print(f'{scan:48s} {got["runs"]:6d} runs {got["gpu_h"]:9.1f} GPU-h')
    book = _macros(acc)
    if dry_run or not numbers:
        print(f'\n[campaign] dry run: {len(book)} macro(s) not written')
        return {'status': 'dry run', 'figures': (), 'tables': (), 'n_macros': len(book)}
    prov = C.provenance(functions=('paper_assets.campaign.totals',),
                        scans=sorted(acc['per_scan']),
                        note='counts and wall_time_s only; no loss is read here',
                        provisional=C.provisional_scans(),
                        runs=acc['runs'], gpu_hours=round(acc['gpu_h'], 1),
                        gpu_models=acc['gpu_models'])
    path = book.write(provenance=prov)
    print(f"\n[campaign] {path.relative_to(C.REPO)} <- {len(book)} macros")
    return {'status': 'ok', 'figures': (), 'tables': (), 'n_macros': len(book)}


if __name__ == '__main__':
    sys.exit(0 if build().get('status') in ('ok', 'dry run') else 1)
