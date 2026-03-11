# Task Board

This is the central coordination file for the agent lab. All agents read this at the start of each session.

## How to Use This File

- **Manager** creates tasks, sets priorities, and marks tasks DONE after reviewing results
- **Scientists** pick up `scientist` tasks, update status to IN_PROGRESS while working, DONE when finished
- **Engineer** picks up `engineer` tasks similarly
- Tasks should be worked in priority order (HIGH before MEDIUM before LOW)
- If a task depends on another, note it in the `Depends on` field

## Status Key
- `TODO` — Not started
- `IN_PROGRESS` — Being worked on
- `DONE` — Completed
- `BLOCKED` — Waiting on something (note what)
- `CANCELLED` — No longer needed

---

## Active Tasks

### Task 1: Baseline characterization — Sven CE vs label regression
- **Assigned to**: scientist
- **Status**: TODO
- **Priority**: HIGH
- **Date created**: 2026-03-11
- **Goal**: Establish the core performance gap between Sven on classification (CE) vs regression (label_regression) on MNIST, without RMSprop
- **Method**:
  - Use `make_mlp(width=32)`, `seed=42`, `batch_size=64`, `num_epochs=20`
  - Run Sven with `loss_key="ce"`, `lr=1.0`, `k=32`, `rtol=1e-3`, `use_rmsprop=False`, `track_svd_info=True`
  - Run Sven with `loss_key="label_regression"`, same params
  - Run Adam baseline with `loss_key="ce"`, `lr=1e-3`
  - Run Adam baseline with `loss_key="label_regression"`, `lr=1e-3`
- **Deliverables**:
  - 4 JSONL result files in `agent_lab/results/`
  - Plot: train loss curves (all 4 runs) — `agent_lab/plots/task1_train_loss.png`
  - Plot: val accuracy curves (all 4 runs) — `agent_lab/plots/task1_val_acc.png`
  - Notes entry with quantitative comparison

### Task 2: Singular value spectrum analysis
- **Assigned to**: scientist
- **Status**: TODO
- **Priority**: HIGH
- **Depends on**: Task 1
- **Date created**: 2026-03-11
- **Goal**: Compare how the Jacobian singular value spectrum differs between CE and label regression loss
- **Method**:
  - Use the Sven results from Task 1 (which have `track_svd_info=True`)
  - Extract singular values at early (epoch 1), mid (epoch 10), and late (epoch 20) training
  - Plot the full spectrum for both loss types at each stage
  - Compute and compare: condition number, effective rank (number of SVs > 1% of max), spectral decay rate
- **Deliverables**:
  - Plot: SV spectra comparison at 3 training stages — `agent_lab/plots/task2_sv_spectra.png`
  - Plot: condition number over training — `agent_lab/plots/task2_condition_number.png`
  - Notes entry with quantitative analysis and hypothesis about why spectra differ

### Task 3: Sven with RMSprop — what does it fix?
- **Assigned to**: scientist
- **Status**: TODO
- **Priority**: MEDIUM
- **Depends on**: Task 1
- **Date created**: 2026-03-11
- **Goal**: Understand what RMSprop-style scaling corrects when added to Sven for classification
- **Method**:
  - Run Sven with CE + `use_rmsprop=True` (same model params as Task 1)
  - Compare update norms and directions: with vs without RMSprop
  - If possible, log the RMSprop scaling factors to see which parameter dimensions are being rescaled most
- **Deliverables**:
  - JSONL results in `agent_lab/results/`
  - Plot comparing loss/accuracy curves: Sven CE (no RMSprop) vs Sven CE (RMSprop) vs Adam
  - Notes entry analyzing what RMSprop changes

---

## Completed Tasks

*None yet.*

---

## Backlog (Future Ideas)

- Try intermediate loss functions (focal loss, label-smoothed CE, MSE on softmax outputs)
- Investigate whether batch size affects the CE vs regression gap
- Test whether the issue is worse for deeper/wider models
- Analyze update direction cosine similarity with standard gradient
- Try different k values and rtol to see if truncation strategy matters more for CE
