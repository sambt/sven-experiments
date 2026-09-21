"""Shared conventions for ``analysis/paper_assets`` -- the ONE command that builds
every figure, table and number macro the ICLR revision quotes.

Read ``campaign/PAPER_CONTRACTS.md`` and ``campaign/PAPER_PLAN.md`` first.  The rules
this module enforces mechanically:

* **Output goes where the manuscript can see it.**  Figures to
  ``iclr_manuscript/figures_iclr/<group>/<name>.pdf``, generated tables to
  ``iclr_manuscript/tables_v2/<name>.tex``, number macros to
  ``iclr_manuscript/numbers_v2_<module>.tex`` which ``iclr_manuscript/numbers_v2.tex``
  ``\\input``s.  A PNG twin of every figure goes to ``agent_lab/paper_assets/<group>/``
  -- **never** into the manuscript repo, which is Overleaf-synced.
* **No number is typed by hand.**  A module calls the analysis functions of record
  (``analysis/lib/headline.py``, ``headline_figs.py``, ``large_figs.py``, ``spectra_figs.py``,
  ``reviewer_figs.py``, ``profile_helpers.py``) and formats what they return.  Every
  generated file is accompanied by ``<name>.provenance.json`` naming the functions
  called, the scan directories read, the hash of ``bench/best_configs.json`` and the
  timestamp, so a reviewer can tell which refresh a number came from.
* **Macro names are letters only** (TeX forbids digits in a command name) and unique
  across the whole package; a NaN / None / inf never reaches a macro or a table cell.
* **Figures are drawn at their final print size.**  The manuscript's ``\\linewidth`` is
  5.5 in (``iclr2026_conference.sty``); a panel used 3-across at ``0.32\\linewidth`` is
  1.76 in wide, so the figure is laid out at exactly that size with the font sizes the
  page will show.  Nothing is saved with ``bbox_inches='tight'``, which would change the
  width and silently rescale the fonts.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ``analysis/`` is a flat import root (`import style`, `import headline`), exactly as the
# notebooks use it.  Running ``python -m paper_assets.<module>`` from ``analysis/`` already
# puts it on the path; this makes an import from elsewhere work too.
ANALYSIS = Path(__file__).resolve().parent.parent
for _p in (ANALYSIS / "lib", ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import style  # noqa: E402  (after the sys.path fix)

REPO = ANALYSIS.parent

#: the Overleaf-synced manuscript repo.  NEVER commit, push, pull or branch in here.
MANUSCRIPT = REPO / 'iclr_manuscript'
FIG_DIR = MANUSCRIPT / 'figures_iclr'
TABLE_DIR = MANUSCRIPT / 'tables_v2'
NUM_DIR = MANUSCRIPT
#: where the PNG twins and any scratch output go -- outside the manuscript repo
LAB = REPO / 'agent_lab' / 'paper_assets'

#: the selection of record (``tools/select_best.py``'s output)
SELECTION_PATH = REPO / 'bench' / 'best_configs.json'

#: the asset modules, in the order ``numbers_v2.tex`` inputs them.  ``campaign`` is last
#: because it is the only one that reads no curve -- just counts and ``wall_time_s`` over
#: every record in the results root -- and it must be in this tuple so that the ONE
#: command of ``campaign/PAPER_CONTRACTS.md`` regenerates the campaign totals too.
MODULES = ('main', 'reviewer', 'large', 'spectra', 'campaign')

#: ``\linewidth`` of the manuscript in inches (``\textwidth 5.5 true in``)
LINEWIDTH_IN = 5.5

#: the three panel widths the plan uses, as a fraction of ``\linewidth``
PANEL_FRACTIONS = (0.32, 0.49, 1.0)

#: base font size (pt) for a panel drawn at each fraction, so the text on the PAGE is
#: legible next to a 10 pt caption.  A 0.32 panel is 1.76 in wide: anything below ~6 pt
#: is unreadable in print and anything above ~8 pt leaves no room for the data.
_FONT_FOR_FRACTION = {0.32: 7.0, 0.49: 8.0, 1.0: 9.0}

#: the ratio of panel height to panel width used unless a caller overrides it
DEFAULT_ASPECT = 0.80


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec='seconds')


def selection_hash(path=None):
    """``sha256`` (first 16 hex) of the selection file every table is built from."""
    p = Path(path or SELECTION_PATH)
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def selection_generated_at(path=None):
    """The instant ``bench/best_configs.json`` was written, as it records it."""
    try:
        return json.loads(Path(path or SELECTION_PATH).read_text()).get('generated_at')
    except (OSError, ValueError):
        return None


def results_root():
    """The results root the analysis layer resolves (``$SV3_RESULTS_ROOT`` or the default),
    as an absolute path, for the provenance block.

    ``style.DEFAULT_RESULTS_ROOT`` is ``'../experiment_results'`` *relative to*
    ``analysis/``, because that is where every notebook runs.  Resolving it against the
    process CWD instead put the provenance root two directories above the repository
    whenever the build (or a test) was started from the repository root, so a relative
    root is anchored to :data:`ANALYSIS` here.
    """
    root = Path(style.resolve_results_root())
    if not root.is_absolute():
        root = ANALYSIS / root
    return str(root.resolve())


def is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) \
        and not isinstance(value, bool)


def finite(value):
    """Whether ``value`` is a real, finite number (the gate every macro passes)."""
    if value is None or isinstance(value, bool):
        return False
    if not is_number(value):
        return False
    return bool(math.isfinite(float(value)))


# ---------------------------------------------------------------------------
# LaTeX text: escaping, and the ``Raw`` opt-out
# ---------------------------------------------------------------------------
class Raw(str):
    """A table cell / macro body that is already LaTeX (``$\\pm$``, ``\\%``, math).

    Everything else is escaped by :func:`latex_escape`.  A ``Raw`` string is still
    CHECKED (:func:`check_raw`): a bare ``%`` would comment out the rest of the line and
    a bare ``_`` outside math is a hard error in text mode, so the emitter refuses both
    rather than producing a file that does not compile.
    """
    __slots__ = ()


#: Whether a cell that already CONTAINS LaTeX (a backslash or a ``$``) is treated as
#: :class:`Raw` even when its wrapper class was lost.  It is lost routinely: numpy
#: normalises a ``str`` subclass, so a :class:`Raw` cell put into a ``DataFrame`` comes
#: back out a plain ``str`` and would be escaped into ``0.2 \$\textbackslash{}pm\$ 0.01``.
#: The content check is still validated by :func:`check_raw`, so an un-escaped ``%`` or a
#: text-mode ``_`` is refused either way.  Set to False to require explicit :class:`Raw`.
AUTO_RAW = True


def is_raw(value):
    """Whether a table cell should be emitted verbatim rather than escaped."""
    if isinstance(value, Raw):
        return True
    return bool(AUTO_RAW and isinstance(value, str) and re.search(r'[\\$]', value))


_ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def latex_escape(text):
    """Escape the ten TeX specials in ``text`` (a :class:`Raw` string passes through)."""
    if isinstance(text, Raw):
        return str(text)
    out = []
    for ch in str(text):
        out.append(_ESCAPES.get(ch, ch))
    return ''.join(out)


_MATH_SPAN = re.compile(r'\$[^$]*\$')


def check_raw(text, where=''):
    """Refuse a :class:`Raw` fragment that would break the LaTeX build.

    A ``%`` that is not ``\\%`` comments out the rest of the line (the single most common
    way a generated table silently loses its ``\\\\``); a ``_`` or ``^`` outside ``$...$``
    is an error in text mode.  Returns ``text``.
    """
    s = str(text)
    if re.search(r'(?<!\\)%', s):
        raise ValueError(f'{where or "raw cell"}: un-escaped "%" in {s!r} -- '
                         f'write "\\%" or pass a plain string to be escaped')
    outside = _MATH_SPAN.sub('', s)
    bad = [c for c in ('_', '^') if re.search(rf'(?<!\\)\{c}', outside)]
    if bad:
        raise ValueError(f'{where or "raw cell"}: {bad} outside math in {s!r} -- '
                         f'wrap it in $...$ or escape it')
    if s.count('$') % 2:
        raise ValueError(f'{where or "raw cell"}: unbalanced "$" in {s!r}')
    return text


# ---------------------------------------------------------------------------
# Number formatting (the only place a number becomes a string)
# ---------------------------------------------------------------------------
def fmt_sig(value, sig=3, dashes='--'):
    """``value`` at ``sig`` significant figures, with ``1.23e-07`` rendered as LaTeX math.

    Every printed loss goes through here, so 4.8e-07 and 0.1388 are formatted by one rule
    instead of per table.  Non-finite -> ``dashes``.
    """
    if not finite(value):
        return Raw(dashes) if dashes == '--' else dashes
    v = float(value)
    if v == 0:
        return '0'
    exp = math.floor(math.log10(abs(v)))
    if -4 <= exp < 5:
        digits = max(sig - 1 - exp, 0)
        s = f'{v:.{digits}f}'
        # only a FRACTIONAL part may lose trailing zeros: `'2500'.rstrip('0')` is 25
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s or '0'
    mant = v / 10 ** exp
    return Raw(f'${mant:.{max(sig - 1, 0)}f}\\times 10^{{{exp}}}$')


def fmt_pm(mean, std, sig=3, dashes='--'):
    """``mean $\\pm$ std`` at ``sig`` significant figures -- the seed band
    (:data:`paired.SEED_SPREAD_LABEL`), the one spelling used in every table."""
    if not finite(mean):
        return Raw(dashes) if dashes == '--' else dashes
    m = fmt_sig(mean, sig)
    if not finite(std):
        return Raw(str(m))
    # the std is shown at the mean's scale, so 0.1388 +/- 0.0021 does not become
    # "0.1388 +/- 0.00210"
    s = fmt_sig(std, 2)
    return Raw(f'{m} $\\pm$ {s}')


def fmt_int(value, dashes='--'):
    if not finite(value):
        return Raw(dashes) if dashes == '--' else dashes
    return f'{int(round(float(value))):d}'


def fmt_counts(finished, attempted, dashes='--'):
    """``finished/attempted`` -- printed on every row of every results table."""
    if not finite(finished) or not finite(attempted):
        return Raw(dashes) if dashes == '--' else dashes
    return f'{int(finished)}/{int(attempted)}'


def fmt_pct(value, digits=1, signed=False, dashes='--'):
    """A FRACTION as a percentage (``0.0216`` -> ``2.2\\%``)."""
    if not finite(value):
        return Raw(dashes) if dashes == '--' else dashes
    sign = '+' if signed else ''
    return Raw(f'{100 * float(value):{sign}.{digits}f}\\%')


def fmt_pct_value(value, digits=1, signed=False, dashes='--'):
    """A value ALREADY in percent (``96.6`` -> ``96.6\\%``)."""
    if not finite(value):
        return Raw(dashes) if dashes == '--' else dashes
    sign = '+' if signed else ''
    return Raw(f'{float(value):{sign}.{digits}f}\\%')


def fmt_ci(low, high, sig=3, dashes='--'):
    """A two-sided interval as ``[low, high]`` (the t-interval on a paired mean)."""
    if not finite(low) or not finite(high):
        return Raw(dashes) if dashes == '--' else dashes
    return Raw(f'$[{_math_body(fmt_sig(low, sig))}, {_math_body(fmt_sig(high, sig))}]$')


def _math_body(text):
    """``fmt_sig``'s output as the BODY of a math expression (strips the ``$``)."""
    s = str(text)
    return s[1:-1] if s.startswith('$') and s.endswith('$') else s


def fmt_rank(rank, n, dashes='--'):
    """``3 of 14`` -- a rank is meaningless without the size of the field."""
    if not finite(rank) or not finite(n):
        return Raw(dashes) if dashes == '--' else dashes
    return f'{int(rank)} of {int(n)}'


# ---------------------------------------------------------------------------
# Number macros
# ---------------------------------------------------------------------------
_MACRO_NAME = re.compile(r'^[A-Za-z]+$')


@dataclass
class Macros:
    """A module's number macros, validated as they are added.

    ``add(name, value, source=..., note=...)`` records one macro; the name is the plan's
    ``num<Scan><Method><Quantity>`` with **letters only** (TeX forbids digits in a command
    name), a collision raises, and a non-finite value raises rather than writing
    ``\\newcommand{\\numX}{nan}`` into the paper.  ``source`` is the analysis function the
    value came from and is written into the file as a comment beside the macro, which is
    what makes a number traceable without reading this code.
    """
    module: str
    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    provisional: set = field(default_factory=set)

    def add(self, name, value, source='', note='', sig=3, provisional=False):
        """Record one macro.  ``value`` may be a number (formatted at ``sig`` significant
        figures), an already-formatted string, or a :class:`Raw` LaTeX fragment."""
        if not _MACRO_NAME.match(str(name)):
            raise ValueError(f'macro name {name!r} must be letters only '
                             f'(TeX forbids digits in a command name)')
        if name in self.values:
            raise ValueError(f'macro {name!r} defined twice in module {self.module!r} '
                             f'(was {self.values[name]!r} from {self.sources.get(name)!r})')
        if value is None:
            raise ValueError(f'macro {name!r}: value is None -- a missing number must be '
                             f'left out of the paper, not written as a blank')
        if is_number(value):
            if not finite(value):
                raise ValueError(f'macro {name!r}: value is {value!r} (not finite)')
            body = fmt_sig(value, sig)
        else:
            body = value
            if str(body).strip() in ('', 'nan', 'None', 'NaN', '--'):
                raise ValueError(f'macro {name!r}: refusing the empty value {body!r}')
        check_raw(body, f'macro {name}')
        self.values[name] = str(body)
        self.sources[name] = source
        if note:
            self.notes[name] = note
        if provisional:
            self.provisional.add(name)
        return name

    def add_pm(self, base, mean, std, source='', sig=3, **kw):
        """``\\num<base>`` = the mean and ``\\num<base>Std`` = the seed std, as two macros.

        Two macros rather than one ``mean +/- std`` string, so the prose can quote either
        alone; the table cells use :func:`fmt_pm`.
        """
        self.add(base, mean, source=source, sig=sig, **kw)
        if finite(std):
            self.add(base + 'Std', std, source=source, sig=2, **kw)
        return base

    def update(self, other):
        for name, value in other.values.items():
            if name in self.values:
                raise ValueError(f'macro {name!r} defined in two modules')
            self.values[name] = value
            self.sources[name] = other.sources.get(name, '')
        self.notes.update(other.notes)
        self.provisional |= other.provisional
        return self

    def __len__(self):
        return len(self.values)

    def __contains__(self, name):
        return name in self.values

    def write(self, path=None, provenance=None):
        """Write ``\\newcommand`` lines, sorted, each with its source as a comment."""
        path = Path(path or NUM_DIR / f'numbers_v2_{self.module}.tex')
        return write_macros(path, self, provenance=provenance)


def write_macros(path, macros, provenance=None):
    """Emit ``\\newcommand{\\<name>}{<body>}`` lines for ``macros``.

    ``macros`` is a :class:`Macros` or a plain ``{name: value}`` dict (which is validated
    through :class:`Macros` on the way in, so the letters-only rule and the no-NaN rule
    hold either way).  The file opens with a provenance header naming the module, the
    selection hash and the timestamp; each macro carries the analysis function it came
    from.  Returns the path.
    """
    path = Path(path)
    if not isinstance(macros, Macros):
        book = Macros(module=path.stem)
        for name, value in dict(macros).items():
            book.add(name, value)
        macros = book
    lines = [
        f'% {path.name} -- GENERATED by analysis/paper_assets/{macros.module}.py.  '
        f'Do not edit by hand.',
        f'% Every number in the prose of the revision is one of these macros '
        f'(campaign/PAPER_CONTRACTS.md).',
        f'% generated: {_now()}',
        f'% selection: {SELECTION_PATH.relative_to(REPO)} '
        f'sha256:{selection_hash()} written {selection_generated_at()}',
        f'% macros: {len(macros)}',
    ]
    if macros.provisional:
        lines.append(f'% PROVISIONAL (an input was still refreshing): '
                     f'{", ".join(sorted(macros.provisional))}')
    lines.append('%')
    width = max((len(n) for n in macros.values), default=1)
    for name in sorted(macros.values):
        body = macros.values[name]
        src = macros.sources.get(name) or ''
        note = macros.notes.get(name) or ''
        tail = ' '.join(x for x in (src, note) if x)
        flag = ' [PROVISIONAL]' if name in macros.provisional else ''
        pad = ' ' * (width - len(name))
        comment = f'  % {tail}{flag}' if (tail or flag) else ''
        lines.append(f'\\newcommand{{\\{name}}}{{{body}}}{pad}{comment}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    if provenance is not None:
        write_provenance(path, provenance)
    return path


def write_numbers_index(path=None, modules=MODULES, present_only=True):
    """Write ``numbers_v2.tex``, which ``\\input``s the per-module macro files.

    The manuscript ``\\input{numbers_v2}`` once; this file is the only thing that knows
    how many modules there are.  ``present_only`` skips a module whose file has not been
    generated yet, so a partial build still compiles.
    """
    path = Path(path or NUM_DIR / 'numbers_v2.tex')
    lines = ['% numbers_v2.tex -- GENERATED by analysis/paper_assets.  Do not edit by hand.',
             '% Every campaign-derived number quoted in the revision is a macro defined in',
             '% one of the files below (campaign/PAPER_CONTRACTS.md: numbers are never typed).',
             f'% generated: {_now()}',
             '%']
    written, owners, clashes = [], {}, {}
    for module in modules:
        stem = f'numbers_v2_{module}'
        f = Path(path).parent / f'{stem}.tex'
        if present_only and not f.is_file():
            lines.append(f'% (no {stem}.tex yet -- run: cd analysis && '
                         f'python -m paper_assets --only {module})')
            continue
        lines.append(f'\\input{{{stem}}}')
        written.append(stem)
        for name in macro_names(f):
            if name in owners:
                clashes.setdefault(name, [owners[name]]).append(module)
            else:
                owners[name] = module
    # A macro defined in two modules makes LaTeX abort ("command already defined"), and
    # the module files are written by four agents in parallel, so the clash is detected
    # HERE -- once, where every file is visible -- rather than in the build log.
    if clashes:
        lines.insert(4, '% !! MACRO NAME CLASH between modules -- the build WILL fail:')
        for name, mods in sorted(clashes.items()):
            lines.insert(5, f'%    \\{name}: {sorted(set(mods))}')
        print(f'[paper_assets] MACRO CLASH: {len(clashes)} name(s) defined by two '
              f'modules: {sorted(clashes)[:8]}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    reg = LAB / 'macro_registry.json'
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({'generated_at': _now(), 'owners': owners,
                               'clashes': {k: sorted(set(v))
                                           for k, v in clashes.items()}},
                              indent=2, sort_keys=True) + '\n')
    return path, written


_NEWCOMMAND = re.compile(r'^\\newcommand\{\\([A-Za-z]+)\}')


def macro_names(path):
    """The macro names one ``numbers_v2_*.tex`` file defines."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    return [m.group(1) for m in (_NEWCOMMAND.match(line) for line in text.splitlines())
            if m]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def provenance(functions=(), scans=(), reads=(), note='', provisional=(), **extra):
    """The provenance record written beside every generated file.

    ``functions``: the analysis functions of record that produced the content (the paper
    plan names them per asset); ``scans``: the scan directories read; ``reads``: any other
    input (a profile root, a checkpoint-spectra directory).  The selection hash and the
    timestamp are added here so no caller can forget them.
    """
    rec = {
        'generated_at': _now(),
        'functions': sorted({str(f) for f in functions}),
        'scan_dirs': sorted({str(s) for s in scans}),
        'reads': sorted({str(r) for r in reads}),
        'results_root': results_root(),
        'selection_file': str(SELECTION_PATH.relative_to(REPO)),
        'selection_sha256_16': selection_hash(),
        'selection_generated_at': selection_generated_at(),
        'provisional': sorted({str(p) for p in provisional}),
    }
    if note:
        rec['note'] = note
    rec.update(extra)
    return rec


def write_provenance(target, record):
    """Write ``<target>.provenance.json`` next to a generated file.

    For a file inside the Overleaf-synced manuscript the sidecar goes to :data:`LAB`
    instead (mirroring the relative path), so nothing but the artefact itself lands in
    that repo.
    """
    target = Path(target)
    try:
        rel = target.relative_to(MANUSCRIPT)
        out = LAB / 'provenance' / rel.parent / f'{target.name}.provenance.json'
    except ValueError:
        out = target.with_name(f'{target.name}.provenance.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + '\n')
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def set_paper_style(fraction=0.32):
    """matplotlib rcParams for a panel drawn at ``fraction`` of ``\\linewidth``.

    The manuscript's visual language (``analysis/style.set_style``) sized for a 8x6 in
    single panel; a 1.76 in panel needs its own sizes or every label is illegible on the
    page.  Sven is black and colours come from :data:`style.METHOD_COLORS` regardless.
    """
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    base = _FONT_FOR_FRACTION.get(round(float(fraction), 2), 7.0)
    plt.rcParams.update({
        'font.size': base,
        'axes.labelsize': base,
        'axes.titlesize': base + 0.5,
        'legend.fontsize': base - 1.0,
        'xtick.labelsize': base - 1.0,
        'ytick.labelsize': base - 1.0,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': None,          # exact physical size: see the module docstring
        'savefig.pad_inches': 0.0,
        'lines.linewidth': 1.1,
        'lines.markersize': 3.0,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.45,
        'ytick.minor.width': 0.45,
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        'xtick.minor.size': 1.4,
        'ytick.minor.size': 1.4,
        'axes.grid': False,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 0.9,
        'legend.labelspacing': 0.25,
        'legend.borderaxespad': 0.3,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'cm',
        'pdf.fonttype': 42,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.012,
        'figure.constrained_layout.w_pad': 0.012,
        'figure.constrained_layout.hspace': 0.02,
        'figure.constrained_layout.wspace': 0.02,
    })
    return base


def figsize(ncols=1, nrows=1, fraction=0.32, aspect=DEFAULT_ASPECT, extra_h=0.0):
    """Figure size in inches for an ``nrows x ncols`` grid of ``fraction``-width panels.

    ``aspect`` is panel height / panel width; ``extra_h`` adds inches for a figure-level
    legend strip.  The returned width is what ``\\includegraphics[width=...]`` will show
    at 1:1, so the font sizes of :func:`set_paper_style` are the sizes on the page.
    """
    w = ncols * float(fraction) * LINEWIDTH_IN
    h = nrows * float(fraction) * LINEWIDTH_IN * float(aspect) + float(extra_h)
    return (w, h)


def fig_path(name, group):
    return FIG_DIR / group / f'{name}.pdf'


def save_fig(fig, name, group, provenance_record=None, png=True, close=True):
    """Save ``fig`` as the manuscript PDF and (for eyeballing) a PNG twin under
    ``agent_lab/paper_assets/``.

    Returns ``(pdf_path, png_path)``.  The PNG is rendered at the SAME physical size, so
    what it shows is what the page shows -- that is the whole point of looking at it.
    """
    pdf = fig_path(name, group)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    out_png = None
    if png:
        out_png = LAB / group / f'{name}.png'
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220)
    if provenance_record is not None:
        write_provenance(pdf, provenance_record)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return pdf, out_png


def legend_below(fig, handles, labels, ncol=4, y=0.0, **kw):
    """One figure-level legend under a panel row, so no panel wastes area on a key."""
    kw.setdefault('frameon', False)
    kw.setdefault('loc', 'lower center')
    kw.setdefault('bbox_to_anchor', (0.5, y))
    return fig.legend(handles, labels, ncol=ncol, **kw)


def method_handles(methods, lw=1.3):
    """``(handles, labels)`` for a method legend in :data:`style.METHOD_COLORS` order,
    with Sven's own thicker black line, using :func:`style.method_label` names."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for m in methods:
        is_sven = style.canonical_method(m) == 'Sven'
        handles.append(Line2D([], [], color=style.method_color(m),
                              lw=lw * (1.8 if is_sven else 1.0)))
        labels.append(style.method_label(m))
    return handles, labels


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
_ALIGN = {'l': 'l', 'r': 'r', 'c': 'c'}

#: A record key that turns a row into one full-width ``\multicolumn`` block heading,
#: instead of spending a column on a group name repeated down the table.
SPAN_KEY = '__span__'


def span_row(title):
    """A full-width block heading for :func:`booktabs` (see :data:`SPAN_KEY`)."""
    return {SPAN_KEY: title}


def _header(text):
    """A column / group header: escaped, unless it is already LaTeX.

    Headers carry math as often as cells do (``$k$``, ``$P/N$``, ``$2\\eta/\\kappa$``), so
    they go through the same :func:`is_raw` / :func:`check_raw` gate -- escaping them
    unconditionally printed ``\\$k\\$`` in the table head.
    """
    if is_raw(text):
        check_raw(text, 'column header')
        return str(text)
    return latex_escape(text)


def _ragged(cell):
    """Left-align one ``p{}`` cell without the ``array`` package (see :func:`booktabs`).

    ``{\\raggedright ...\\par}`` would also do it, but an explicit ``\\par`` inside a ``p``
    cell pushes the column template's closing ``\\strut`` onto a line of its own and costs
    a whole extra line per row (measured: T1 grows 141 pt -> 242 pt, a third of a page in
    a main text with ~0.4 pp of slack).  A ``\\parbox[t]{\\linewidth}`` breaks the
    paragraph inside itself, where ``\\raggedright`` is still in force, keeps the column's
    width, and is byte-for-byte the same height as the justified default once the
    closing ``\\strut`` the ``p`` template would have supplied is put back -- without it
    a two-line cell loses its depth and its second line crowds the next row.
    """
    cell = str(cell)
    return ('\\parbox[t]{\\linewidth}{\\raggedright ' + cell + '\\strut}'
            if cell.strip() else cell)


def booktabs(rows, columns=None, align=None, caption=None, label=None,
             groups=None, notes=(), env='tabular', column_format=None,
             midrules=(), fit=False):
    """A ``booktabs`` table body as a string, ready to ``\\input``.

    ``rows`` is a list of dicts or a DataFrame.  Cells are escaped
    (:func:`latex_escape`) unless they are :class:`Raw`, and every ``Raw`` cell is checked
    (:func:`check_raw`) so an un-escaped ``%`` or a text-mode ``_`` cannot reach the build.

    What this emits is the ``tabular`` only -- **no** ``table`` float, no caption, no
    ``\\centering``: the manuscript owns the float so the caption can be blue and the
    placement can be tuned (``PAPER_PLAN`` risk 9: ``\\color{blue}`` inside a float
    colours the rules).  ``caption`` / ``label``, if given, are emitted as comments so the
    integrator can see the intended wording.

    ``groups`` is an optional list of ``(span, title)`` for a ``\\cmidrule`` header row;
    ``midrules`` a set of row indices to draw a ``\\midrule`` BEFORE.

    ``align`` takes ``'l'``/``'r'``/``'c'`` and also a full column spec of more than one
    character, which is passed through verbatim -- ``'p{1.4in}'`` for a free-text column
    (a selected configuration, a swept grid) that must WRAP rather than set the width of
    the whole table.

    A record carrying the key :data:`SPAN_KEY` is emitted as one full-width
    ``\\multicolumn`` row instead of a data row, which is how a table groups its blocks
    (one per scan) without spending a column on repeating the group's name.

    ``fit=True`` wraps the ``tabular`` in ``\\resizebox{\\linewidth}{!}{...}`` (``graphicx``
    is already loaded by the manuscript), so a table slightly wider than the text block
    shrinks to fit instead of running into the margin.  It is a guard, not a licence: a
    table that needs more than ~0.8 shrink should lose columns instead, and the emitted
    comment says the guard is on so a reader of the file can see it.
    """
    if isinstance(rows, pd.DataFrame):
        columns = list(rows.columns) if columns is None else columns
        records = rows.to_dict('records')
    else:
        records = [dict(r) for r in rows]
        if columns is None:
            columns = list(dict.fromkeys(k for r in records
                                         for k in r if k != SPAN_KEY))
    ncol = len(columns)
    if column_format is None:
        if align is None:
            align = ['l'] + ['r'] * (ncol - 1)
        # a spec of more than one character ('p{1.4in}', '>{\\raggedright}p{1in}') is a
        # column specification in its own right and is passed through unchanged
        column_format = ''.join(a if len(str(a)) > 1 else _ALIGN.get(a, 'l')
                                for a in align)
    # A ``p{}`` column is JUSTIFIED by default, which on a 1-inch column of
    # "k=128, lr=0.5, rtol=0.3" stretches one inter-word space to half the column.
    # ``>{\raggedright\arraybackslash}p{}`` is the usual fix but needs the ``array``
    # package, which the manuscript does not load and which PAPER_PLAN section 6 forbids
    # adding.  So each ``p`` cell is wrapped in ``{\raggedright ...\par}`` instead: the
    # group confines \raggedright's redefinition of ``\\`` to the cell, so it is safe
    # even when the ``p`` column is the LAST one and the row's ``\\`` follows it.
    ragged = {i for i, a in enumerate(align or ())
              if len(str(a)) > 1 and 'p{' in str(a) and 'raggedright' not in str(a)}
    out = ['% GENERATED by analysis/paper_assets -- do not edit by hand.']
    if caption:
        out.append(f'% caption (draft): {caption}')
    if label:
        out.append(f'% label: {label}')
    for n in notes:
        out.append(f'% note: {n}')
    out.append(f'\\begin{{{env}}}{{{column_format}}}')
    out.append('\\toprule')
    if groups:
        cells, rules, at = [], [], 1
        for span, title in groups:
            span = int(span)
            if title:
                cells.append(f'\\multicolumn{{{span}}}{{c}}{{{_header(title)}}}')
                if span > 1:
                    rules.append(f'\\cmidrule(lr){{{at}-{at + span - 1}}}')
            else:
                cells.extend([''] * span)
            at += span
        out.append(' & '.join(cells) + r' \\')
        if rules:
            out.append(' '.join(rules))
    head = [f'\\textbf{{{_header(c)}}}' for c in columns]
    out.append(' & '.join(_ragged(h) if i in ragged else h
                          for i, h in enumerate(head)) + r' \\')
    out.append('\\midrule')
    for i, rec in enumerate(records):
        if i in set(midrules):
            out.append('\\midrule')
        if SPAN_KEY in rec:
            title = rec[SPAN_KEY]
            body = str(title) if is_raw(title) else latex_escape(title)
            if is_raw(title):
                check_raw(title, 'span row')
            out.append(f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{body}}}}}' + r' \\')
            continue
        cells = []
        for j, c in enumerate(columns):
            v = rec.get(c, '')
            if is_raw(v):
                check_raw(v, f'column {c!r}')
                cell = str(v)
            elif v is None or (is_number(v) and not finite(v)):
                cell = '--'
            elif is_number(v):
                cell = str(fmt_sig(v))
            else:
                cell = latex_escape(v)
            cells.append(_ragged(cell) if j in ragged else cell)
        out.append(' & '.join(cells) + r' \\')
    out.append('\\bottomrule')
    out.append(f'\\end{{{env}}}')
    if fit:
        at = out.index(f'\\begin{{{env}}}{{{column_format}}}')
        out[at:at] = ['% width guard: the tabular below is shrunk to \\linewidth if it '
                      'is wider (graphicx, already loaded)',
                      '\\resizebox{\\linewidth}{!}{%']
        out.append('}')
    return '\n'.join(out) + '\n'


def write_table(name, rows, provenance_record=None, **kw):
    """Write one generated table to ``iclr_manuscript/tables_v2/<name>.tex``."""
    path = TABLE_DIR / f'{name}.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(booktabs(rows, **kw))
    if provenance_record is not None:
        write_provenance(path, provenance_record)
    return path


# ---------------------------------------------------------------------------
# Freshness: three inputs are still refreshing while the paper is written
# ---------------------------------------------------------------------------
_fresh_memo = {}


def provisional_scans(scans=None):
    """The scans whose selection can still move (``headline.freshness_report``).

    Every module gates on this: an asset that depends on a provisional scan is stamped
    ``PROVISIONAL`` in its provenance and (for a caption-bearing asset) in its note, so a
    refresh is visible rather than assumed.
    """
    import headline as hl
    key = tuple(scans) if scans else None
    if key not in _fresh_memo:
        try:
            rep = hl.freshness_report(scans=list(scans) if scans else None)
            _fresh_memo[key] = list(rep.attrs.get('provisional_scans', []))
        except Exception as exc:                       # pragma: no cover - IO dependent
            print(f'[paper_assets] freshness_report failed ({exc}); '
                  f'treating every scan as fresh')
            _fresh_memo[key] = []
    return list(_fresh_memo[key])


def allow_provisional():
    """Whether the build may write assets for a still-moving scan.

    Default **yes**, with the stamp: three inputs (CIFAR-CE re-selection, the Fig-5 re-run
    and ``profile_results_v3``) are refreshing in a parallel workflow and the paper has to
    be assembled against what exists.  ``PAPER_ASSETS_STRICT=1`` makes a provisional scan
    a hard failure instead.
    """
    return os.environ.get('PAPER_ASSETS_STRICT', '') not in ('1', 'true', 'yes')


# ---------------------------------------------------------------------------
# "Did a refresh move the paper?"  --  the asset snapshot
# ---------------------------------------------------------------------------
SNAPSHOT_PATH = LAB / 'asset_snapshot.json'


def asset_digest():
    """``{relative path: sha256-16}`` over every generated asset in the manuscript.

    Tables, macro files and figure PDFs.  Three inputs are still refreshing while the
    paper is written, so the integrator has to be able to answer "what did re-running
    change" without reading 30 tables: :func:`check_assets` diffs this against the last
    snapshot.
    """
    out = {}
    for pattern in ('tables_v2/*.tex', 'numbers_v2*.tex', 'figures_iclr/*/*.pdf'):
        for p in sorted(MANUSCRIPT.glob(pattern)):
            try:
                out[str(p.relative_to(MANUSCRIPT))] = hashlib.sha256(
                    p.read_bytes()).hexdigest()[:16]
            except OSError:
                continue
    return out


def check_assets(update=True):
    """Diff the current assets against the last snapshot; return ``(added, changed,
    removed)`` and, by default, record the new snapshot.

    A figure PDF's bytes carry its creation date, so a figure counts as "changed" on
    every rebuild; the table and macro files do not, which is why they are the ones to
    read.  Nothing here writes into the manuscript.
    """
    now = asset_digest()
    before = {}
    if SNAPSHOT_PATH.is_file():
        try:
            before = json.loads(SNAPSHOT_PATH.read_text()).get('assets', {})
        except (OSError, ValueError):
            before = {}
    added = sorted(set(now) - set(before))
    removed = sorted(set(before) - set(now))
    changed = sorted(k for k in set(now) & set(before) if now[k] != before[k])
    if update:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(
            {'generated_at': _now(), 'selection_sha256_16': selection_hash(),
             'assets': now}, indent=2, sort_keys=True) + '\n')
    return added, changed, removed


def banner(module, extra=''):
    print('=' * 78)
    print(f'paper_assets.{module}  ->  {MANUSCRIPT.relative_to(REPO)}  ({_now()})')
    if extra:
        print(extra)
    print('=' * 78)


__all__ = [
    'ANALYSIS', 'REPO', 'MANUSCRIPT', 'FIG_DIR', 'TABLE_DIR', 'NUM_DIR', 'LAB',
    'SELECTION_PATH', 'MODULES', 'LINEWIDTH_IN', 'PANEL_FRACTIONS', 'DEFAULT_ASPECT',
    'Raw', 'is_raw', 'AUTO_RAW', 'latex_escape', 'check_raw',
    'fmt_sig', 'fmt_pm', 'fmt_int', 'fmt_counts', 'fmt_pct', 'fmt_pct_value', 'fmt_ci',
    'fmt_rank', 'finite', 'is_number',
    'Macros', 'write_macros', 'write_numbers_index', 'macro_names',
    'provenance', 'write_provenance', 'selection_hash', 'selection_generated_at',
    'results_root',
    'set_paper_style', 'figsize', 'fig_path', 'save_fig', 'legend_below',
    'method_handles',
    'booktabs', 'write_table', 'span_row', 'SPAN_KEY',
    'provisional_scans', 'allow_provisional', 'banner',
    'asset_digest', 'check_assets', 'SNAPSHOT_PATH',
]
