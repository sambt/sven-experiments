# Agent Lab — Instructions for Autonomous Agents

You are an autonomous research agent working on the SV3 optimizer project. Your goal is to design, run, and analyze experiments to understand specific research questions about the Sven optimizer.

## Roles

This lab uses a multi-agent structure with defined roles. Read your specific role file in `agent_lab/roles/`:

- **Research Manager** (`roles/manager.md`): Owns research direction, synthesizes findings, creates tasks on the taskboard, maintains the report. Does NOT run experiments or write code.
- **Research Scientist** (`roles/scientist.md`): Executes experiment tasks from the taskboard, analyzes results, creates plots, writes detailed notes with findings and hypotheses. Does NOT decide direction or update the report.
- **Research Engineer** (`roles/engineer.md`): Implements code changes requested by the manager or scientists — new loss functions, custom training loops, diagnostic tools. Does NOT run scientific experiments or decide direction.

## Coordination

All agents coordinate through files:
- **`agent_lab/taskboard.md`** — The central task board. Every agent reads this first. Tasks have statuses (TODO, IN_PROGRESS, DONE), assignments (scientist, engineer), and priorities (HIGH, MEDIUM, LOW).
- **`agent_lab/notes.md`** — Shared lab notebook. Tag entries with your role: `[manager]`, `[scientist]`, `[engineer]`.
- **`agent_lab/report.md`** — Polished findings summary. Only the manager writes to this.
- **`agent_lab/results/`** — JSONL experiment logs (scientists write, all read).
- **`agent_lab/plots/`** — Saved visualizations (scientists write, all read).

## Workflow Per Role

**Manager**: Read taskboard + notes + results → synthesize → update report → create new tasks on taskboard.

**Scientist**: Read taskboard → pick up TODO task → update status to IN_PROGRESS → run experiment → save results + plots → write notes → update status to DONE.

**Engineer**: Read taskboard → pick up TODO task → implement code → test it → write notes on how to use it → update status to DONE.

## Key Conventions

### Experiment Results
- Save all run logs as JSONL files in `agent_lab/results/`
- Use descriptive filenames: `{experiment_name}_{timestamp}.jsonl`
- Each JSONL line should include at minimum: `run_id`, `optimizer`, hyperparameters, and a `losses` dict with training curves
- Use the helper functions in `agent_lab/scripts/run_experiment.py` — they handle serialization and follow the project conventions

### Plots
- Save all plots to `agent_lab/plots/` as PNG files
- Use descriptive filenames: `{what_is_shown}_{timestamp}.png`
- Always include axis labels, titles, and legends
- Use `agent_lab/scripts/plot_utils.py` for common plot types

### Notes (`agent_lab/notes.md`)
This is your lab notebook. It should be **chronological** and **informal**. Write entries like:
```
## 2026-03-11 — Session 1: Initial exploration
- Tried X with parameters Y
- Observed Z
- Hypothesis: ...
- Next steps: ...
```
This file is for you to build up context across sessions. Be honest about what worked and what didn't.

### Report (`agent_lab/report.md`)
This is the **polished summary** of your findings. Structure it as:
- **Question**: What are you investigating?
- **Key Findings**: Numbered list of clear conclusions
- **Evidence**: References to specific plots and data
- **Open Questions**: What remains unclear?

Update this incrementally as you learn more.

## Running Experiments

### Option 1: Use the standalone runner (recommended for agent work)

The file `agent_lab/scripts/run_experiment.py` provides helper functions:

```python
import sys
sys.path.insert(0, "/path/to/sv3")  # repo root
from agent_lab.scripts.run_experiment import *

device = get_device()        # "cuda" or "cpu"
dataset = get_mnist()        # loads MNISTDataset
model = make_mlp(width=32)   # sv3.nn.MLP with 3 hidden layers

# Run Sven
result, wrapped_model, optimizer = run_sven(
    model, dataset,
    loss_key="ce",           # or "label_regression" or "mse"
    device=device,
    lr=1.0, k=32, rtol=1e-3,
    batch_size=64, num_epochs=10,
    use_rmsprop=False,       # set True to enable RMSprop variant
    track_svd_info=True,     # logs singular values
)
save_result(result, "my_experiment_name")

# Run a baseline (Adam, SGD, RMSprop, etc.)
result, model = run_baseline(
    model, dataset,
    loss_key="ce",
    device=device,
    optim_name="Adam", lr=1e-3,
    batch_size=64, num_epochs=10,
)
save_result(result, "adam_baseline")
```

### Option 2: Use the Hydra experiment framework

```bash
python -m experiments.run_experiment --config-name mnist_scan
```

Configs in `experiments/configs/`. Relevant ones:
- `mnist_scan.yaml` — MNIST with cross-entropy, Sven uses RMSprop
- `mnist_scan_labelRegression.yaml` — MNIST with label regression (MSE on one-hot)

### Writing custom scripts

For targeted experiments, create new scripts in `agent_lab/scripts/`. They can import everything from the repo:

```python
from sv3.nn import MLP, SvenWrapper
from sv3.sven import Sven
from experiments.datasets import MNISTDataset
from experiments.experiment_code.experiment_utils import (
    set_seed, train_loop_svd, train_loop_standard, build_standard_optimizer,
)
```

## Key API Details

### Loss Functions
SVD losses return **per-sample** losses (shape `(B,)`):
- `"ce"`: `F.cross_entropy(pred, y, reduction='none')`
- `"label_regression"`: MSE between predictions and one-hot encoded labels, summed over classes
- `"mse"`: `((pred - y) ** 2).sum(dim=-1)`

Standard losses return a **scalar**:
- `"ce"`: `nn.CrossEntropyLoss()`
- `"label_regression"`: same as SVD but `.mean()` at end
- `"mse"`: `nn.MSELoss()`

### Sven Optimizer
```python
# 1. Wrap model
wrapped = SvenWrapper(model, loss_fn, device, microbatch_size=1, param_fraction=1.0)

# 2. Create optimizer
opt = Sven(wrapped, lr=1.0, k=32, rtol=1e-3,
           track_svd_info=True, svd_mode='torch',
           use_rmsprop=False, alpha_rmsprop=0.99)

# 3. Training step (inside loop)
losses, preds = wrapped.loss_and_grad(batch)  # computes Jacobian, stores in wrapped.grads
opt.step(batch)                                # uses pseudoinverse update
```

Key parameters:
- `k`: rank of truncated SVD (number of singular values to keep)
- `rtol`: relative tolerance — SVs below `rtol * max_sv` are zeroed
- `lr`: learning rate (can also be `"polyak"` for Polyak step size)
- `use_rmsprop`: applies RMSprop-style scaling to gradients before SVD
- `track_svd_info`: logs singular values in `opt.svd_info["svs"]`
- `variable_k`: greedily adds singular value components, stopping if loss increases

### Training Loops
The existing `train_loop_svd` and `train_loop_standard` in `experiment_utils.py` return a `losses` dict with these keys:
- `train`: per-epoch mean training loss
- `val`: per-epoch mean validation loss
- `train_acc`, `val_acc`: per-epoch accuracy (if `track_acc=True`)
- `train_batch`, `val_batch`: per-batch losses
- `epoch_times`, `batch_times_train`, `batch_times_val`: timing info

### Dataset
```python
from experiments.datasets import MNISTDataset
dataset = MNISTDataset()  # has .train_dataset and .val_dataset
# Images are flattened to (784,), normalized. Labels are integer class indices.
```

### Model
```python
from sv3.nn import MLP
model = MLP(input_dim=784, hidden_dims=[32, 32, 32], output_dim=10, activation=nn.GELU)
```

## Important Notes

- The Jacobian computation is memory-intensive. Keep batch sizes moderate (64-256) and model sizes small to start.
- Always use `set_seed(seed)` before creating a model for reproducibility.
- Use `copy.deepcopy(model.state_dict())` to save initial weights so you can reset for fair comparisons.
- Don't modify core optimizer code in `sv3/` without explicit approval — focus on experiments and analysis.
- Use `device = "cuda" if torch.cuda.is_available() else "cpu"` — the code should work on both.
- If an experiment fails, log the error in your notes and move on.
