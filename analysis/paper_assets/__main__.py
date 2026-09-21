"""CLI: build the paper's assets.

    cd analysis && ../.venv/bin/python -m paper_assets
    cd analysis && ../.venv/bin/python -m paper_assets --only main,spectra
    cd analysis && ../.venv/bin/python -m paper_assets --figures-only
    cd analysis && ../.venv/bin/python -m paper_assets --list

After the parallel refresh (CIFAR-CE re-selection, the Fig-5 re-run, ``profile_results_v3``)
closes, re-running this command with no arguments updates every asset the paper uses and
rewrites ``iclr_manuscript/numbers_v2.tex``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback

from . import common


def _parse(argv=None):
    p = argparse.ArgumentParser(prog='python -m paper_assets',
                                description=__doc__.splitlines()[0])
    p.add_argument('--only', default='', help='comma-separated modules '
                                              f'({", ".join(common.MODULES)})')
    p.add_argument('--figures-only', action='store_true')
    p.add_argument('--tables-only', action='store_true')
    p.add_argument('--numbers-only', action='store_true')
    p.add_argument('--dry-run', action='store_true',
                   help='report what each module would build; write nothing')
    p.add_argument('--list', action='store_true', help='list the modules and exit')
    p.add_argument('--check', action='store_true',
                   help='after building, report which generated assets changed since '
                        'the last run (the answer to "did the refresh move the paper?")')
    p.add_argument('--results-root', default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    if args.list:
        for name in common.MODULES:
            try:
                importlib.import_module(f'paper_assets.{name}')
                state = 'ready'
            except ModuleNotFoundError:
                state = 'NOT IMPLEMENTED'
            print(f'{name:10s} {state}')
        return 0
    modules = [m.strip() for m in args.only.split(',') if m.strip()] or list(common.MODULES)
    unknown = [m for m in modules if m not in common.MODULES]
    if unknown:
        print(f'unknown module(s) {unknown}; known: {list(common.MODULES)}')
        return 2
    what = {'figures': not (args.tables_only or args.numbers_only),
            'tables': not (args.figures_only or args.numbers_only),
            'numbers': not (args.figures_only or args.tables_only)}
    common.banner('__main__', f'modules={modules} build={what} '
                              f'dry_run={args.dry_run}')
    prov = common.provisional_scans()
    if prov:
        print(f'[paper_assets] PROVISIONAL scans (an input is still refreshing): {prov}')
        if not common.allow_provisional():
            print('[paper_assets] PAPER_ASSETS_STRICT=1 -- refusing to write')
            return 3
    reports, failed = {}, []
    for name in modules:
        try:
            mod = importlib.import_module(f'paper_assets.{name}')
        except ModuleNotFoundError:
            print(f'[paper_assets] {name}: module not written yet -- skipping')
            reports[name] = {'status': 'not implemented yet'}
            continue
        try:
            reports[name] = mod.build(root=args.results_root, dry_run=args.dry_run, **what)
        except Exception:                                   # one module must not sink all
            traceback.print_exc()
            failed.append(name)
            reports[name] = {'status': 'FAILED'}
    if not args.dry_run:
        path, written = common.write_numbers_index()
        print(f'[paper_assets] {path.relative_to(common.REPO)} <- {written}')
    if args.check:
        added, changed, removed = common.check_assets()
        print('\n--- assets changed since the last run ---')
        for label, names in (('added', added), ('changed', changed),
                             ('removed', removed)):
            # a figure PDF's bytes carry a timestamp, so it always reads as changed;
            # the tables and macro files are the ones that mean something
            tex = [n for n in names if n.endswith('.tex')]
            print(f'{label}: {len(names)} file(s), {len(tex)} of them .tex')
            for n in tex[:40]:
                print(f'    {n}')
    print('\n--- summary ---')
    for name, rep in reports.items():
        if not isinstance(rep, dict):
            print(f'{name}: {rep}')
            continue
        figs = len(rep.get('figures', ()) or ())
        tabs = len(rep.get('tables', ()) or ())
        macs = rep.get('n_macros', 0)
        note = rep.get('status', '')
        print(f'{name:10s} figures={figs:2d} tables={tabs:2d} macros={macs:3d} {note}')
    if failed:
        print(f'FAILED: {failed}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
