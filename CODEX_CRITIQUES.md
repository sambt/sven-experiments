# Sven experiments: methodological critique

Reviewed September 18, 2026. Scope: experiment runners, datasets, models, optimizer implementations and wrappers, YAML configurations, launch scripts, analysis helpers, and all analysis/notebook code, including the exploratory SvenReg notebook. I also audited saved results and ran small CPU checks; I did not rerun GPU training or change experiment/analysis code.

The current analysis already improves on several issues described in `analysis/ANALYSIS_FIXES.md`: headline selection uses seed means, and sharded timings are identified. Those fixes should be retained. However, corrected source code does not repair historical results automatically. The findings below distinguish existing-result problems from blockers for planned experiments.

## Addressed on follow-up — AdamW defaults and replacement runs

Reviewed the newly pulled cluster-rerun setup (`135d36d`, merged in `6c57588`). **The AdamW duplication issue is addressed correctly in the current implementation and rerun procedure; it is no longer an active must-change item.**

- The [configuration/factory path](/Users/sambt/iaifi/sv3/experiments/experiment_code/experiment_utils.py:598) resolves unspecified AdamW decay to **0.01**, preserves explicit overrides, and passes it to PyTorch. A CPU update check confirmed actual parameter shrinkage at the default, no shrinkage with explicit zero, and correct handling of another nonzero value.
- The [scan runner](/Users/sambt/iaifi/sv3/experiments/experiment_code/generic_scan.py:612) includes decay in AdamW run IDs and result metadata, avoiding collisions with legacy runs. Analysis includes decay in configuration identity.
- The [new launcher](/Users/sambt/iaifi/sv3/submit_reruns_2026-09-17.sh:54) archives legacy duplicate AdamW files outside the loader's top-level glob and submits replacements for the four headline scans, nanoGPT, and critical-batch nanoGPT. It includes MuonW; [part 2](/Users/sambt/iaifi/sv3/submit_reruns_2026-09-17_part2.sh:25) adds MuonW to the explicit-decay sweeps. The older generic launcher's omission therefore does not describe these reruns.
- The [serial timing pass](/Users/sambt/iaifi/sv3/bench/timing_serial.sbatch:27) reselects configurations and forwards the selected decay into timing runs.

The [cluster launch log](/Users/sambt/iaifi/sv3/analysis/RERUNS_NEEDED.md:230) records submission and archiving. Local result files still contain the old zero-decay runs, so completion and regenerated figures are not independently verified here; their absence locally is not evidence that the cluster fix failed. Confirm replacement results after synchronization. Tuning decay beyond the valid default remains a **mid-priority improvement**, not the original correctness bug.

## Top priority — must change before relying on the affected conclusions

### 1. The “degree-4 polynomial” benchmark is actually an additive cubic

In [RandomPolynomialDataset](/Users/sambt/iaifi/sv3/experiments/datasets/all_datasets.py:93), `product(range(d), ...)` omits the intended constant, linear, and pure degree-`d` monomials; `term += x**power` then adds factors instead of multiplying them. The configured six-variable, degree-four target has no interactions and no fourth powers. A CPU check fitted it using just an intercept and each coordinate's first three powers—19 features—with relative RMSE approximately **8e-8**.

**Action:** correct both enumeration and evaluation, check against known monomials, and rerun every polynomial experiment, including batch-size, masking, and overparameterization studies. Alternatively, retain historical runs only under an accurate “additive cubic” description. They do not establish performance on the stated polynomial family.

### 2. CIFAR validation evaluates different models for Sven and the baselines

The [standard loop](/Users/sambt/iaifi/sv3/experiments/experiment_code/experiment_utils.py:173) switches to evaluation mode after training; the [Sven loop](/Users/sambt/iaifi/sv3/experiments/experiment_code/experiment_utils.py:298) never does. [SvenWrapper.evaluate](/Users/sambt/iaifi/sv3/sven/sven/nn/sven_wrapper.py:138) disables gradients but leaves BatchNorm in training mode and passes live buffers. Thus Sven uses validation-batch statistics and changes running statistics during validation. Even the standard loop's *initial* validation occurs in training mode. Freezing normalization during Gram capture does not fix evaluation.

A CPU probe confirmed mutated running means and changed predictions for an unchanged example when only its batch companions changed. **Action:** make evaluation side-effect-free and apply the same normalization policy across optimizers. Account for repeated forward/closure passes when updating training statistics; any recalibration must use training data only. Rerun affected CIFAR comparisons and fix this before the proposed fine-tuning study.

### 3. The streamed GPT-2 corpus preparation overlaps training and validation

In [prepare_tokens.py](/Users/sambt/iaifi/sv3/experiments/data_prep/prepare_tokens.py:32), both calls to `write_split` start `for ex in ds` over the same re-iterable streaming dataset. The second iteration restarts the stream: validation becomes a prefix of training, despite the disjointness comment. See the [Hugging Face streaming interface](https://huggingface.co/docs/datasets/stream). Also, an exhausted source leaves the preallocated file longer than the number of tokens actually written.

**Action:** split by disjoint documents or consume one shared iterator, truncate files to actual length, and verify overlap and token counts before training. This blocks claims from GPT-2/FineWeb runs generated this way. I found no such saved runs locally; the existing character-level Shakespeare split uses a different path and is not implicated by this bug.

### 4. Official test sets are being used for model selection

[MNIST and CIFAR loaders](/Users/sambt/iaifi/sv3/experiments/datasets/all_datasets.py:39) assign `train=False` datasets to validation. The notebooks select configurations by final validation loss and report performance on the same examples. Seed averaging does not make those estimates held-out test results. The toy dataset even constructs a test set that the experiment runner never evaluates.

**Action:** separate training, tuning, and final evaluation; save selected checkpoints and evaluate locked choices on held-out data. Label existing results as tuning-set comparisons. Since the official test sets have already informed development, acknowledge that reuse rather than claiming that a new split retroactively restores independence; use fresh confirmation tasks/data where possible.

### 5. Exceptions disappear, and missing seeds are treated more favorably than recorded failures

The [scan runner](/Users/sambt/iaifi/sv3/experiments/experiment_code/generic_scan.py:583) prints exceptions without writing failed-run records. [Configuration ranking](/Users/sambt/iaifi/sv3/analysis/analysis_helpers.py:86) infers expected seeds from observed files, accepts a majority of successful seeds, and penalizes recorded divergence but not missing runs. Consequently, three successes plus two exceptions can look more reliable than three successes plus two saved NaNs. Entirely failed configurations disappear. The selected polynomial LBFGS configuration currently has four successful seeds and one divergence, while its successful curves alone enter the aggregate.

**Action:** reconcile results against an explicit grid/seed manifest and record numerical failures, OOMs, timeouts, and unlaunched jobs separately. Predeclare eligibility and failure treatment; show success counts beside conditional-on-success curves. Complete matched seeds for headline comparisons or report the resulting uncertainty and failure rates explicitly. This requires both logging changes and reanalysis; missing historical runs need recovery/reruns.

### 6. Two timing summaries systematically distort optimizer costs

The training loops stop their batch timers **before** `loss.item()` synchronizes CUDA; Sven performs additional synchronization internally. These batch-time summaries compare different amounts of completed work. This criticism does **not** apply identically to epoch/total times, which encompass those synchronizations.

Separately, [profiler summarization](/Users/sambt/iaifi/sv3/experiments/optimizer_profile.py:89) removes timings beyond three MADs, and [profile plots](/Users/sambt/iaifi/sv3/analysis/profile_helpers.py:74) use that filtered mean. SOAP's legitimate preconditioner refreshes every ten steps are discarded as spikes. In the saved MNIST `SOAP/B64` profile, the mean over the last 40 measured steps is **6.14 ms**, versus the reported filtered **5.56 ms**.

**Action:** synchronize batch measurements consistently; report amortized cost over complete optimizer-update cycles, retaining periodic work. Reprocess saved profiles before collecting replacements. Use isolated, matched-hardware runs for speedup claims; retain sharded clocks only as clearly qualified diagnostics. Check loss/update finiteness too: a finite duration alone does not establish a valid training step.

### 7. Epoch metrics weight batches equally instead of examples

Both training loops aggregate batch means with `np.mean`. With 10,000 validation examples and batch size 128, the last 16 examples receive eight times the per-example weight of earlier examples. The distortion changes with batch size, affecting exactly the batch-size comparisons being studied. Accuracy aggregation has the same issue.

**Action:** aggregate summed losses/correct counts and divide by total examples—or valid tokens for language modeling. Stored per-batch validation losses can often repair historical loss curves using known batch lengths; accuracy may require reevaluation. Do not interpret online training loss as a checkpoint loss either: it spans changing parameters, and LBFGS records its last closure evaluation rather than the same observation point as SGD.

## Mid priority — improvements that would materially strengthen the comparison

### 1. Give baselines credible recipes and disclose the tuning budget

The main grids give Sven 72 configurations on toy/polynomial and 128 on MNIST, versus four learning rates for most ordinary baselines. SGD has no momentum or schedule; several selected baselines sit on learning-rate grid boundaries. Generic forwarding also leaves important damping/update-frequency parameters untuned and prevents nonzero decay for some optimizers that support it. Expand around boundary optima, tune momentum/decay and appropriate schedules, and include a budget-controlled tuning comparison. Equal epoch budgets alone do not equalize development effort.

Check baseline identity as well. The Muon factory sends **all 2-D parameters**, including embeddings/output heads, to Muon, but sends convolution kernels to AdamW. This differs from the intended hidden-weight usage in [PyTorch's Muon documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html); document the variant or implement and tune the intended grouping. The custom schedule-free classes also differ from the [reference AdamW](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py) and [SGD](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/sgd_schedulefree.py) algorithms, including extra momentum/averaging choices. They are not active saved comparisons, but should be replaced or explicitly named as variants before inclusion.

### 2. Make stochastic runs reproducible across sharding and resumption

[Seeding occurs once per model seed](/Users/sambt/iaifi/sv3/experiments/experiment_code/generic_scan.py:390), before the configuration loops. Initial weights are correctly copied and loader generators reset, but random parameter masks consume the global RNG. Skipping previous configurations or changing shards can therefore change a masked run with the same recorded seed/ID. Use separate, stable initialization, data-order, and optimizer RNG streams, and check isolated versus resumed execution.

Store the resolved configuration, dataset fingerprint, code revisions—including the editable Sven dependency—hardware/software versions, status, and actual resource use. Existing run IDs omit important changes such as epoch count and dataset seed, so existence-based skipping can retain stale results. Use explicit configuration identities in analysis rather than treating every scalar metadata field as a hyperparameter. Audit old spectra too: previously saved truncated spectra cannot establish the unseen spectral tail merely because current logging has been fixed.

### 3. Separate optimization efficiency from changes in problem and resource budget

Present updates, examples/tokens processed, and synchronized wall time as distinct axes. An epoch is not an update. Choose wall-time winners using the relevant budget, not automatically the configuration with best final-epoch loss; predeclare several meaningful target losses and account for runs that never reach them. [AlgoPerf's protocol](https://github.com/mlcommons/algorithmic-efficiency/blob/main/docs/DOCUMENTATION.md) is a useful reference for tuning budgets and time-to-target evaluation.

The MNIST `N_train` study changes steps per epoch; the full-batch synthetic studies change batch size and available Jacobian rank. Synthetic validation inputs and target normalization also change with dataset generation. These jointly confound a causal claim about `P/N`. Generate a fixed independent evaluation set, hold batch/resource budgets constant where possible, and complement varying `N` with varying width at fixed data. In critical-batch studies, vary retained rank separately from batch size and measure crossings more frequently than once per epoch.

### 4. Strengthen existing workloads before adding many more

The evidence is concentrated in small smooth synthetic problems, a small MNIST MLP, and a small character-level language model. CIFAR currently uses an ImageNet-style ResNet stem and no training augmentation, with a short 20-epoch horizon. All optimizers share this setup, but it is a weak test of competitiveness under a well-developed vision recipe.

Prioritize a CIFAR-style ResNet stem, standard augmentation, and credible training schedules on CIFAR-10, then CIFAR-100; and one properly split BPE language-model workload with equal token budgets, tuned AdamW/Muon, and intermediate validation. The existing small-data pretrained fine-tuning configuration is also relevant once normalization is corrected. These are more informative than multiplying further tiny-MNIST variants.

### 5. Report uncertainty and claims at the level the evidence supports

Use fresh confirmation seeds after tuning, pair comparisons by initialization/data order, and report individual runs or intervals on paired differences. The old two-seed critical-batch and three-seed language-model results cannot establish small advantages reliably; the pulled configurations and cluster rerun plan now expand these to five seeds, addressing the requested sample-size increase once completed. The clipped standard-deviation bands are descriptive spread, not confidence intervals; use accurate legends and visible completion counts.

The existing seed-mean selections are already task-dependent: HIG beats Sven on toy and polynomial final loss. Reassess nanoGPT rankings using the replacement AdamW runs rather than the superseded zero-decay comparison. Frame advantages by task, loss target, and resource budget. Likewise, qualify complexity claims by backend: exact Gram processing still operates on a batch-sized matrix when retained `k` is small, and full CIFAR capture materializes a Jacobian. A universal “cost scales with retained rank” interpretation is unsupported by those implementations.

## Low priority — focused experiments for understanding the optimizer

1. **Noise, non-interpolation, and loss scaling.** Add controlled observation/label noise and a small underparameterized case. Test multiplicative loss scaling and, as a diagnostic, an additive loss constant, which leaves the minimizer and ordinary gradients unchanged but can change a loss-based pseudoinverse step. For the `kappa` ablation, retune learning rate: in a full-row-rank, untruncated solve, changing the power also changes update scale. The existing fixed-rate sweep mixes this with geometric effects.

2. **Why truncation helps.** Compare hard truncation with a damped solve under matched tuning budgets; log full spectra, effective retained rank, update norm, and linearized predicted versus actual loss reduction. This would distinguish noise suppression, conditioning, and step-size control. Verify diagnostics against an explicit small Jacobian before using them mechanistically.

3. **What masking and microbatching trade away.** At matched measured time/memory, compare element, tensor, and row masks using the *achieved* parameter fraction, not just the requested fraction. Separately vary microbatch aggregation and retained rank: averaging loss rows changes the update geometry as well as computational cost. Include failure rates rather than only successful curves.

4. **Implicit regularization and parameterization.** Compare generalization, distance moved from initialization, margins where appropriate, and function-space changes under equivalent network rescalings. A minimum-norm linearized step is not evidence that training finds a minimum-norm final parameter vector; raw parameter norms alone do not establish the proposed implicit-bias mechanism. Treat the exploratory SvenReg notebook as a hypothesis generator until its settings, labels, and repeated-seed comparisons are made reproducible.

**Suggested order:** repair dataset/evaluation/status recording first; reanalyze failures, batch-weighted losses, and profile averages; then rerun corrected polynomial/CIFAR experiments and properly tuned baselines with locked confirmation protocols. Collect the additional mechanism experiments only after that foundation is sound.
