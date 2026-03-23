import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(name, results_root='../experiment_results', selection_fn=None):
    """Load experiment results, supporting both old and new storage formats.

    Old format: a single ``{name}.jsonl`` file with one JSON object per line.
    New format: a directory ``{name}/`` containing one ``.jsonl`` file per run.

    The function tries the directory format first, then falls back to the
    single-file format.  Returns a :class:`pd.DataFrame`.
    """
    root = Path(results_root)
    scan_dir = root / name

    records: list[dict] = []

    # New format: directory of per-run JSONL files
    if scan_dir.is_dir():
        for f in sorted(scan_dir.glob('*.jsonl')):
            if selection_fn is not None and not selection_fn(str(f).split("/")[-1]):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if records:
            print(f"Loaded {len(records)} runs from {scan_dir}/ (directory format)")
            return pd.DataFrame(records)

    # Old format: single JSONL file
    jsonl_path = root / f'{name}.jsonl'
    if jsonl_path.is_file():
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} runs from {jsonl_path} (single-file format)")
        return pd.DataFrame(records)

    raise FileNotFoundError(
        f"No results found for '{name}': tried {scan_dir}/ and {jsonl_path}"
    )

def load_results_jsonl(name, results_root='../experiment_results'):
    """Load experiment results from a single JSONL file.

    This is the old storage format, where all runs are stored in a single
    ``{name}.jsonl`` file.  Returns a :class:`pd.DataFrame`.
    """
    jsonl_path = Path(results_root) / f'{name}.jsonl'
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No results found for '{name}': {jsonl_path} does not exist")

    records: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} runs from {jsonl_path}")
    return pd.DataFrame(records)

def set_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.grid': False,
        'font.family': 'arial',
        'legend.frameon': False,
        'mathtext.fontset': 'cm',
    })
    
lr_labels = {10**k: f"10^{{{k}}}" for k in range(-10,10)}
lr_labels[0.5] = "0.5"
lr_labels[0.0003] = "3 \\times 10^{-4}"
lr_labels[0.003] = "3 \\times 10^{-3}"
lr_labels[0.03] = "3 \\times 10^{-2}"
lr_labels[0.05] = "0.05"
lr_labels[50.0] = "50"