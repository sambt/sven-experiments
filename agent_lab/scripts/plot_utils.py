"""
Shared plotting utilities for agent_lab experiments.

Results are expected to be JSONL files where each line is a JSON object with:
  - run_id: str
  - optimizer: str (e.g. "Sven", "Adam", "SGD")
  - loss_fn: str (e.g. "ce", "label_regression")
  - losses: dict with keys like:
      - train: list of per-epoch mean training loss
      - val: list of per-epoch mean validation loss
      - train_acc: list of per-epoch mean training accuracy (if classification)
      - val_acc: list of per-epoch mean validation accuracy (if classification)
      - train_batch: list of per-batch training losses
      - val_batch: list of per-batch validation losses
  - svd_info: dict (for Sven runs) with:
      - svs: list of arrays of singular values per batch
      - num_nonzero_svs: list of counts per batch
"""

import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_results(pattern: str = "*.jsonl") -> list[dict]:
    """Load all JSONL result files matching a glob pattern."""
    results = []
    for path in sorted(RESULTS_DIR.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def plot_training_curves(
    results: list[dict],
    metric: str = "val",
    title: str | None = None,
    save_name: str | None = None,
    log_scale: bool = False,
):
    """
    Plot a training metric for multiple runs.

    Args:
        results: list of result dicts (from load_results or run_* functions)
        metric: key into result['losses'], e.g. 'train', 'val', 'train_acc', 'val_acc'
        title: plot title
        save_name: filename to save in plots/ (without directory)
        log_scale: use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        label = r.get("run_id", f"{r.get('optimizer', '?')}_{r.get('loss_fn', '?')}")
        data = r.get("losses", {}).get(metric)
        if data is not None:
            ax.plot(data, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        path = PLOTS_DIR / save_name
        fig.savefig(path, dpi=150)
        print(f"Saved plot: {path}")
        plt.close(fig)
    return fig, ax


def plot_singular_values(
    results: list[dict],
    batch_indices: list[int] | None = None,
    title: str = "Singular Value Spectra",
    save_name: str | None = None,
):
    """
    Plot singular value spectra from Sven runs.

    Args:
        results: list of result dicts that have svd_info
        batch_indices: which batch steps to plot (default: first, middle, last)
        title: plot title
        save_name: filename to save
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[0, idx]
        svs_list = r.get("svd_info", {}).get("svs", [])
        if not svs_list:
            ax.text(0.5, 0.5, "No SVD info", transform=ax.transAxes, ha="center")
            continue

        if batch_indices is None:
            n = len(svs_list)
            batch_indices_use = [0, n // 2, n - 1]
        else:
            batch_indices_use = batch_indices

        for bi in batch_indices_use:
            if bi < len(svs_list):
                svs = np.array(svs_list[bi])
                ax.plot(svs, label=f"batch {bi}", marker=".")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.set_yscale("log")
        label = r.get("run_id", r.get("loss_fn", "?"))
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_name:
        path = PLOTS_DIR / save_name
        fig.savefig(path, dpi=150)
        print(f"Saved plot: {path}")
        plt.close(fig)
    return fig, axes
