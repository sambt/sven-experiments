#!/usr/bin/env python
"""Did a rebuild change a paper figure, or only its timestamp?

    tools/cmp_figures.py                 # every group, against the stored snapshot
    tools/cmp_figures.py main spectra    # just these groups
    tools/cmp_figures.py --snapshot      # (re)take the snapshot from what is on disk now
    tools/cmp_figures.py --snapshot-dir /path/to/other/snapshot

A figure PDF embeds its creation date, so `md5sum` calls every rebuild a change.  This
scrubs `/CreationDate` (and `/ModDate`, if a writer ever adds one) from both copies and
compares the remaining bytes, which is exact: a figure that reads as IDENTICAL here is
the same picture, and one that reads as CHANGED really is a different picture.  The
`.provenance.json` sidecars are compared too, with their own `generated` timestamp
scrubbed -- a changed sidecar means the figure now claims to come from different inputs.

Exit status 0 when nothing changed, 1 when something did, 2 when the snapshot is missing.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / 'iclr_manuscript' / 'figures_iclr'
#: the sidecars do NOT live beside the PDFs: nothing but the artefact goes into the
#: Overleaf repo, so ``common.write_provenance`` mirrors them under agent_lab/
PROV_DIR = REPO / 'agent_lab' / 'paper_assets' / 'provenance' / 'figures_iclr'
DEFAULT_SNAPSHOT = REPO / 'agent_lab' / 'paper_assets' / 'figs_snapshot'

_DATE = re.compile(rb'/(?:Creation|Mod)Date\s*\((?:[^()\\]|\\.)*\)')


def scrub_pdf(data: bytes) -> bytes:
    """The PDF's bytes without its date fields (and without the xref offsets they move).

    Removing a date changes every following byte offset, so the trailing cross-reference
    table would differ even for an identical picture; the content streams are what matter,
    so the xref section is dropped as well.
    """
    data = _DATE.sub(b'/Date()', data)
    cut = data.rfind(b'startxref')
    return data[:cut] if cut > 0 else data


def scrub_json(text: str) -> str:
    try:
        rec = json.loads(text)
    except ValueError:
        return text
    if isinstance(rec, dict):
        rec.pop('generated', None)
        rec.pop('generated_at', None)
    return json.dumps(rec, sort_keys=True, indent=2)


def _files(group_dir: Path):
    return sorted(p for p in group_dir.rglob('*')
                  if p.is_file() and p.suffix in ('.pdf', '.json'))


def take_snapshot(groups, snapshot: Path):
    n = 0
    for group in groups:
        dst = snapshot / group
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for src in (FIG_DIR / group, PROV_DIR / group):
            if not src.is_dir():
                continue
            for p in _files(src):
                shutil.copy2(p, dst / p.name)
                n += 1
    print(f'[cmp_figures] snapshot: {n} file(s) -> {snapshot.relative_to(REPO)}')
    return 0


def compare(groups, snapshot: Path, verbose=True):
    identical, changed, added, removed = [], [], [], []
    for group in groups:
        pre = snapshot / group
        if not pre.is_dir():
            print(f'[cmp_figures] no snapshot for group {group!r} in '
                  f'{snapshot.relative_to(REPO)}')
            return 2
        live_files = {}
        for d in (FIG_DIR / group, PROV_DIR / group):
            if d.is_dir():
                live_files.update({p.name: p for p in _files(d)})
        live_names = set(live_files)
        pre_names = {p.name for p in _files(pre)}
        for name in sorted(pre_names - live_names):
            removed.append(f'{group}/{name}')
        for name in sorted(live_names - pre_names):
            added.append(f'{group}/{name}')
        for name in sorted(live_names & pre_names):
            a, b = (pre / name).read_bytes(), live_files[name].read_bytes()
            if name.endswith('.pdf'):
                same = scrub_pdf(a) == scrub_pdf(b)
            else:
                same = scrub_json(a.decode()) == scrub_json(b.decode())
            (identical if same else changed).append(f'{group}/{name}')
    if verbose:
        for label, names in (('CHANGED', changed), ('added', added),
                             ('removed', removed)):
            for n in names:
                print(f'  {label:8s} {n}')
    print(f'[cmp_figures] identical={len(identical)} changed={len(changed)} '
          f'added={len(added)} removed={len(removed)}  '
          f'({"OK" if not (changed or added or removed) else "DIFFERENCES"})')
    return 1 if (changed or added or removed) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('groups', nargs='*', help='figure groups (default: all present)')
    p.add_argument('--snapshot', action='store_true', help='take the snapshot instead')
    p.add_argument('--snapshot-dir', default=None)
    args = p.parse_args(argv)
    snapshot = Path(args.snapshot_dir) if args.snapshot_dir else DEFAULT_SNAPSHOT
    groups = args.groups or sorted(d.name for d in FIG_DIR.iterdir() if d.is_dir())
    if args.snapshot:
        snapshot.mkdir(parents=True, exist_ok=True)
        return take_snapshot(groups, snapshot)
    return compare(groups, snapshot)


if __name__ == '__main__':
    sys.exit(main())
