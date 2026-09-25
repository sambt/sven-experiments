| scan | title | where | n_records_leg | n_records_fresh | compared as | disposition |
|---|---|---|---|---|---|---|
| cifar10_resnet_ce_kappaScan | cifar10_resnet_ce_kappaScan | legacy only | 2 | 0 | -- | CUT: 1 seed at a tentative set point; replaced on MNIST only (mnist_kappaScan_labelRegression). NO fresh CIFAR kappa scan. |
| cifar10_resnet_ce_paramFrac_scan | cifar10_resnet_ce_paramFrac_scan | legacy only | 6 | 0 | -- | CUT: superseded by rebuttal_fig5_cifar_paramfrac_scan (label regression only), so the cross-entropy parameter-fraction curve has NO fresh counterpart. |
| cifar10_resnet_ce_scan | CIFAR-10 (CE) | both | 290 | 785 | headline |  |
| cifar10_resnet_ce_scan_confirm | cifar10_resnet_ce_scan_confirm | fresh only | 0 | 60 | -- | confirm pass of cifar10_resnet_ce_scan |
| cifar10_resnet_ce_scan_diag | cifar10_resnet_ce_scan_diag | fresh only | 0 | 60 | -- | diag pass of cifar10_resnet_ce_scan |
| cifar10_resnet_ce_scan_timing | cifar10_resnet_ce_scan_timing | fresh only | 0 | 60 | -- | timing pass of cifar10_resnet_ce_scan |
| cifar10_resnet_kappaScan_labelReg | cifar10_resnet_kappaScan_labelReg | legacy only | 5 | 0 | -- | CUT: 1 seed at a tentative set point. The kappa story now runs on MNIST at 5 seeds with a matched-effective-step grid (mnist_kappaScan_labelRegression); there is NO fresh CIFAR kappa scan. |
| cifar10_resnet_paramFrac_scan_labelReg | cifar10_resnet_paramFrac_scan_labelReg | legacy only | 5 | 0 | -- | CUT: superseded by rebuttal_fig5_cifar_paramfrac_scan at 3 seeds, which exists in both roots and IS diffed here (the Sven-only table). The legacy Fig-5 numbers come from that 1-seed scan, not from this one. |
| cifar10_resnet_scan_labelRegression | CIFAR-10 (label reg.) | both | 290 | 600 | headline |  |
| cifar10_resnet_scan_labelRegression_confirm | cifar10_resnet_scan_labelRegression_confirm | fresh only | 0 | 55 | -- | confirm pass of cifar10_resnet_scan_labelRegression |
| cifar10_resnet_scan_labelRegression_diag | cifar10_resnet_scan_labelRegression_diag | fresh only | 0 | 55 | -- | diag pass of cifar10_resnet_scan_labelRegression |
| cifar10_resnet_scan_labelRegression_timing | cifar10_resnet_scan_labelRegression_timing | fresh only | 0 | 55 | -- | timing pass of cifar10_resnet_scan_labelRegression |
| exp_critbatch_mnist | exp_critbatch_mnist | legacy only | 210 | 0 | -- | CUT by the user (09-18): the critical-batch study confounds retained rank with batch size (F23) and crossings were measured once per epoch. NO fresh data; the legacy critical-batch figure has no replacement and cannot be quoted. |
| exp_critbatch_nanogpt | exp_critbatch_nanogpt | legacy only | 210 | 0 | -- | CUT by the user (09-18), same reason. NO fresh data; the legacy critical-batch figure has no replacement and cannot be quoted. |
| exp_finetune_cifar_smallN | exp_finetune_cifar_smallN | legacy only | 240 | 0 | -- | PARKED to the extension phase (user, 09-18): its legacy runs trained BatchNorm on 250-2000 images, the defect C-E2 names. NO fresh data; the fine-tuning claim is unsupported until the parked 408 runs are launched. |
| exp_gpt2_small_comparison | GPT-2 small (FineWeb) | both | 6 | 29 | excluded |  |
| exp_nanogpt_speedrun | nanoGPT (tiny-shakespeare) | both | 100 | 140 | headline |  |
| exp_nanogpt_speedrun_confirm | exp_nanogpt_speedrun_confirm | fresh only | 0 | 25 | -- | confirm pass of exp_nanogpt_speedrun |
| exp_nanogpt_speedrun_diag | exp_nanogpt_speedrun_diag | fresh only | 0 | 25 | -- | diag pass of exp_nanogpt_speedrun |
| exp_nanogpt_speedrun_timing | exp_nanogpt_speedrun_timing | both | 12 | 25 | excluded |  |
| mnist_kappaScan_labelRegression | MNIST (label reg.), kappa | both | 15 | 210 | sven_only |  |
| mnist_microbatch_ce_scan | MNIST (CE), micro-batch | both | 140 | 140 | sven_only |  |
| mnist_microbatch_labelreg_scan | MNIST (label reg.), micro-batch | both | 140 | 140 | sven_only |  |
| mnist_paramfrac_ce_scan | MNIST (CE), param fraction | both | 92 | 100 | sven_only |  |
| mnist_paramfrac_labelreg_scan | MNIST (label reg.), param fraction | both | 87 | 100 | sven_only |  |
| mnist_scan_ce | MNIST (CE) | both | 1040 | 1610 | headline |  |
| mnist_scan_ce_confirm | mnist_scan_ce_confirm | fresh only | 0 | 70 | -- | confirm pass of mnist_scan_ce |
| mnist_scan_ce_diag | mnist_scan_ce_diag | fresh only | 0 | 70 | -- | diag pass of mnist_scan_ce |
| mnist_scan_ce_timing | mnist_scan_ce_timing | both | 60 | 70 | excluded |  |
| mnist_scan_labelRegression | MNIST (label reg.) | both | 999 | 1360 | headline |  |
| mnist_scan_labelRegression_confirm | mnist_scan_labelRegression_confirm | fresh only | 0 | 70 | -- | confirm pass of mnist_scan_labelRegression |
| mnist_scan_labelRegression_diag | mnist_scan_labelRegression_diag | fresh only | 0 | 70 | -- | diag pass of mnist_scan_labelRegression |
| mnist_scan_labelRegression_timing | mnist_scan_labelRegression_timing | both | 60 | 70 | excluded |  |
| polynomial_microbatch_scan | Polynomial, micro-batch | both | 120 | 120 | sven_only |  |
| polynomial_paramfrac_scan | Polynomial, param fraction | both | 86 | 100 | sven_only |  |
| polynomial_scan | Random Polynomial | both | 731 | 1630 | headline |  |
| polynomial_scan_confirm | polynomial_scan_confirm | fresh only | 0 | 225 | -- | confirm pass of polynomial_scan |
| polynomial_scan_diag | polynomial_scan_diag | fresh only | 0 | 75 | -- | diag pass of polynomial_scan |
| polynomial_scan_timing | polynomial_scan_timing | both | 65 | 75 | excluded |  |
| rebuttal_batchsize_polynomial_scan | Polynomial, batch-size sweep | both | 2496 | 3060 | grouped |  |
| rebuttal_fig5_cifar_paramfrac_scan | CIFAR-10 (label reg.), Fig-5 masks | both | 15 | 30 | sven_only |  |
| rebuttal_overparam_mnist_scan | MNIST (label reg.), P/N sweep | both | 2477 | 3600 | grouped |  |
| rebuttal_overparam_polynomial_scan | Polynomial, P/N sweep | both | 2160 | 4320 | grouped |  |
| rebuttal_overparam_toy_1d_scan | Toy 1D, P/N sweep | both | 2160 | 3240 | grouped |  |
| toy_1d_microbatch_scan | Toy 1D, micro-batch | both | 120 | 120 | sven_only |  |
| toy_1d_paramfrac_scan | Toy 1D, param fraction | both | 97 | 100 | sven_only |  |
| toy_1d_scan | Toy 1D | both | 734 | 1770 | headline |  |
| toy_1d_scan_confirm | toy_1d_scan_confirm | fresh only | 0 | 225 | -- | confirm pass of toy_1d_scan |
| toy_1d_scan_diag | toy_1d_scan_diag | fresh only | 0 | 75 | -- | diag pass of toy_1d_scan |
| toy_1d_scan_timing | toy_1d_scan_timing | both | 65 | 75 | excluded |  |
