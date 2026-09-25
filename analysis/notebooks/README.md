# The analysis notebooks

25 notebooks, grouped by what they look at.  Read them in the order below: the
`headline/` group is the paper's main result, everything after it is a study that
qualifies it, and `paper/` is where the manuscript's own figures are made and edited.

Every notebook is self-contained in the sense that matters here — its first cell
chdir's to `analysis/` and puts `analysis/lib/` on `sys.path`, so it can be opened
and run from anywhere (Jupyter, `nbconvert`, an editor) and still finds
`../experiment_results`, `plots_v2/` and `tables/`.  Re-execute in place with
`./make_plots.sh <name>` (no group, no `.ipynb`) or `./make_plots.sh <group>` for a
whole directory.

## headline/ — the seven headline scans

| notebook | what it shows |
|---|---|
| `toy_1d_analysis` | toy 1D regression: the scan end to end (grid, selection, curves, diagnostics) |
| `polynomial_analysis` | random polynomial regression, same structure |
| `mnist_analysis` | MNIST cross-entropy, same structure |
| `mnist_analysis_labelRegression` | MNIST label regression (the paper's MNIST-LR), same structure |
| `baselines_analysis` | Sven against the full optimizer suite, the four MLP scans side by side |
| `headline_tables` | the tables the paper quotes: confirmation vs tuning seeds, paired differences, tuning budget, best-of-*n*, time-to-target |

## spectra/ — why it behaves that way

| notebook | what it shows |
|---|---|
| `spectra_analysis` | the singular-value spectra: what the `k`/`rtol` truncation keeps, what it discards, how both move along training |
| `comparisons` | the same quantities across scans, and against the baselines' trajectories |

## mlp_studies/ — the reviewer studies on the MLP tasks

| notebook | what it shows |
|---|---|
| `overparam_analysis` | dataset-level over-parameterisation, *P > N* (R1's crux) |
| `batchsize_analysis` | batch-size sensitivity (R2 Q1) |
| `kappa_analysis` | is the residual exponent κ more than a learning-rate rescaling? |
| `microbatch_analysis` | micro-batch scaling — the row-count knob (R1's memory ask) |
| `paramfrac_analysis` | parameter fraction on the MLPs (R1 Q2) |

## large_models/ — CIFAR, nanoGPT, GPT-2

| notebook | what it shows |
|---|---|
| `cifar_analysis` | CIFAR-10 / ResNet18: the two headline scans (LR and CE) and Fig. 5 (parameter fraction) |
| `nanogpt_analysis` | nanoGPT, character-level language modelling |
| `gpt2_analysis` | GPT-2-small: one seed, one epoch, and what that can and cannot say |
| `finetune_analysis` | fine-tuning a pretrained ResNet18 (*P ≫ N*) — the parked study; reads the pre-campaign results |

## profiling/ — memory and step time

All four read `profile_results_v3/` (the re-measurement; `profile_results_v2/` is the
frozen before-table), not the scans.

| notebook | what it shows |
|---|---|
| `profile_overview` | the profile at a glance: memory and step time per optimizer |
| `profile_scaling` | scaling with model size |
| `profile_sven_backends` | Sven's capture backends and the Gram trick |
| `profile_paramfrac_microbatch` | parameter fraction and micro-batching as memory knobs |

## paper/ — the manuscript's figures

The paper's own figures, drawn by the same builders `python -m paper_assets` uses, so an
edit here is an edit to the paper. `paper/README.md` has the loop (make → edit → save →
pin); pinned options go to `analysis/paper_assets/figure_overrides.yaml`, which the CLI
build replays.

| notebook | figures |
|---|---|
| `fig_main` | the main text: **Fig. 1 (`headline_curves`)**, `cost_memory`, `k_sweeps`, `hparam_landscape`, the all-seed versions |
| `fig_reviewer` | the reviewer studies: `overparam`, `batchsize`, `kappa`, `knobs`, `budget`, `divergence` |
| `fig_large` | CIFAR, Fig. 5, nanoGPT, GPT-2 and the six profile figures |
| `fig_spectra` | the singular-value figures: spectra along training, norms, used rank, probe sets |
