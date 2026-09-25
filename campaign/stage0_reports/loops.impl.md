**Files changed** — `/n/home/anon/sven-experiments/experiments/experiment_code/experiment_utils.py` (loops + helpers only; the `_CUSTOM_OPTIMIZERS`…`build_standard_optimizer` region is byte-identical — verified by grep over the diff), `experiments/optimizers/hig.py` (`HIGWrapper.evaluate` + one import).
**Files created** — `experiments/nn/norm_utils.py`, `tests/test_evaluate.py`, `tests/test_loops_contract.py`, `tests/test_bn_policy_loops.py`.

**Acceptance tests** (observed summary lines: `8 passed in 7.93s` / `16 passed in 4.49s` / `9 passed in 4.35s`; combined `33 passed in 8.50s` via `campaign/run_cpu_tests.sh`)

| test | asserts |
|---|---|
| `test_evaluate_equals_mean_of_per_example_losses` | 10,000 val examples / eval batch 128 (79 batches, ragged 16): `abs(loss - mean(per-example)) < 1e-6`, `n == 10000` |
| `test_evaluate_accepts_a_mean_reduced_loss_too` | scalar `loss_fn` weighted by `n_b` == per-sample path (1e-9) |
| `test_ragged_last_batch_is_example_weighted…` | constructed case: exact `4*100/260`, mean-of-batch-means off by >1.0 (F10) |
| `test_evaluate_accuracy_is_correct_counts` | acc == exact correct fraction (1e-12) |
| `test_evaluate_is_token_weighted_for_lm` | per-sequence `lm_ce` and flattened token mean both == token mean |
| `test_evaluating_twice_changes_no_buffer` | SmallResNet: every buffer bit-identical, both calls equal |
| `test_evaluate_restores_every_previous_mode` | model stays train, a hand-frozen `bn1` stays frozen |
| `test_evaluate_uses_running_statistics…` | batch 1 == batch 16; differs from the batch-statistic value (F2 inverted) |
| `test_one_step_updates_running_mean_exactly_once[sgd, lbfgs_max_iter3]` | `running_mean == (1-m)*old + m*batch_mean` (rtol 1e-10), `num_batches_tracked == 1` for every norm module |
| `test_evaluation_leaves_every_buffer_bit_identical` | after training, 2 evals → `torch.equal` on all buffers |
| `test_frozen_mode_never_changes_a_buffer[sgd, lbfgs]` | 2 epochs, no buffer changes, norm layers still eval |
| `test_pretraining_validation_runs_in_eval_mode` | `val[0]` == `evaluate()` (1e-12), differs from the train-mode value |
| `test_no_norm_stat_updates_keeps_batch_statistics…` | output identical to a plain train-mode forward, **not** to `.eval()`, and writes nothing |
| `test_absurd_lr_raises_diverged_error…` | `DivergedError`, `len(val)==1`, `train_batch` non-finite tail, `len == step+1` |
| `test_lbfgs_train_loss_is_the_pre_update_loss` | `train_batch[0] == pre-update loss` (1e-12) with `max_iter=3, strong_wolfe`; post-step loss lower |
| `test_train_times_are_bounded_by_epoch_times` | `train_times[i] <= epoch_times[i]`, `train_times[0] == sum(batch_times_train[:2])` |
| `test_old_style_positional_call_still_works`, `test_svd_loop_old_style_positional_call`, plus positional calls in the hig/jd tests | legacy signatures unchanged |
| others | caller-owned dict identity, `val/test/train_eval` all length `n_epochs+1`, curves come from `evaluate`, `eval_step_idx == [2,4]`, checkpointer steps `[0,1,2,3]` + `epoch_end [(0,2),(1,4)]` with `module is model` (also for `SvenWrapper` → `wrapper.model`), `log_this_step` driven by `log_schedule`, `summarize_curves` incl. NaN guard, `bn_mode="eval"` rejected |

**Deviations from CONTRACTS.md**
1. Non-finite check is **after** the update in every loop (contract: after the step only for LBFGS). Reason: it reuses the `.item()` the online curve already needs — no extra sync, one code path — and the run aborts either way.
2. `batch_times_val` / `avg_batch_time_val` are gone (one `evaluate()` call replaces the timed per-batch val loop). Replaced by per-epoch `eval_times` and `avg_eval_time`. `analysis/style.py:44` `_DIAG_LOSS_KEYS` and `scan_analysis.py:735` `STANDALONE_QUANTITIES` still name the old keys (analysis track owns them); `_split_diagnostics` is safe (`if key in losses`).
3. `val_per_model` / `val_acc_per_model` dropped (`evaluate` returns `dict(loss, acc, n)` per contract). Train-side `train_batch_per_model` / `train_per_model` / `train_acc_per_model` unchanged. No live config builds a multi-model net.
4. `evaluate` accepts a per-sample **or** a mean-reduced loss fn (0-dim → weighted by `n_b`, exact); `train_loop_standard` likewise (`_scalar_loss`). This is what keeps the current scalar-`STANDARD_LOSS_FNS` call sites working; passing `SVD_LOSS_FNS[loss_key]` instead makes the online train loss exactly example-weighted.
5. `eval_every_steps` records **after** the step, so `eval_step_idx` = completed optimizer steps and step 0 is not re-evaluated (the pre-training point already is index 0 of `val`).
6. Loops leave the model in **train** mode at the end (legacy left it in eval). Mode is not in `state_dict`; use `evaluate()` for any post-loop measurement.

**Integrator notes**
- New public names in `experiment_utils.py`: `DivergedError(step, value=None)` (`.step`, `.value`), `evaluate(forward_fn, per_sample_loss_fn, loader, device, *, track_acc=False, is_lm=False) -> {"loss","acc","n"}`, `summarize_curves(losses) -> {val_final, val_best, val_best_epoch, val_last3_mean}` (also merged into `losses` by `_finish_losses`; call it yourself on the failure path). `experiments/nn/norm_utils.py`: `no_norm_stat_updates(module)`, `eval_mode(module)` (context managers), `freeze_norm_layers(module)`, `norm_stat_modules(module)`.
- All four loops gained the same keyword-only block: `losses=None, test_loader=None, train_eval_loader=None, eval_every_steps=None, checkpointer=None, log_schedule=None, stop_on_nonfinite=True, bn_mode="batch"` (`train_loop_jd` additionally `is_lm=False`, keyword-only).
- **Call sites you must change (files I do not own):** `generic_scan.py` — bind `losses = {}` before the `try` and pass `losses=losses` so a failed run keeps its curves; build `val/test/train_eval` loaders with the new `eval_batch_size` config key (`evaluate` only consumes the loader it is given) and `train_eval_loader` over the fixed `min(n_train, 10_000)` split-seed-derived training subset; pass `bn_mode` to the loop as well as to the wrapper; pass `checkpointer` / `log_schedule`; read `DivergedError.step` for `diverged_at_step`. Curve keys now include `test`, `test_acc`, `train_eval`, `train_eval_acc`, `train_times`, `eval_times`, `val_step`, `test_step`, `eval_step_idx`, `val_best*`.
- The `svd` loop freezes the norm layers when **either** its `bn_mode` or `getattr(wrapper, "bn_mode")` is `"frozen"`, and never calls `.train()` on the wrapped module (unchanged from legacy); the per-step running-stat write stays the wrapper's job (`loss_and_grad` → `update_norm_running_stats`).
- **Left to integration:** BatchNorm + Sven (the svd loop is unit-tested on an MLP only, as instructed, because `sven_wrapper.py`/`gram_wrapper.py` are being changed in parallel); `_ckpt_module` assumes wrappers expose `.model` (true for `SvenWrapper`, `GramSvenWrapper`, `HIGWrapper`).