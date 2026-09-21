| display | config | wall_s | wall_s_std | epoch_s | sync_train_s | ms_per_step | peak_gpu_mem_mb | steps | examples | wall_vs_sven | scan_wall_s | scan_inflation | gpu_timing | gpu_scan | gpu_scan_frac | fin_timing | att_timing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SGD | lr=3 | 54.08 | 0.1992 | 2.703 | 39.83 | 5.674 | 402.9 | 7020 | 8.986e+05 | 0.04282 | 162.2 | 2.999 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| SGD + momentum | lr=0.3 | 55.57 | 0.09502 | 2.778 | 41.3 | 5.883 | 447.6 | 7020 | 8.986e+05 | 0.04401 | 165.9 | 2.985 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| RMSprop | lr=0.001 | 56.56 | 0.2412 | 2.828 | 42.56 | 6.062 | 447.6 | 7020 | 8.986e+05 | 0.04479 | 166 | 2.935 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Adam | lr=0.001 | 59.1 | 0.3649 | 2.955 | 44.85 | 6.389 | 492.3 | 7020 | 8.986e+05 | 0.0468 | 166.6 | 2.82 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| AdamW | lr=0.01, weight_decay=0.01 | 59.38 | 0.2239 | 2.969 | 45.22 | 6.442 | 492.3 | 7020 | 8.986e+05 | 0.04703 | 206.4 | 3.476 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Polyak SGD | f_star=0, max_lr=1, eps=1e-08 | 69.04 | 0.3054 | 3.451 | 54.46 | 7.757 | 402.9 | 7020 | 8.986e+05 | 0.05467 | 177.3 | 2.569 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| MuonW | lr=0.01, weight_decay=0.1 | 116.8 | 4.027 | 5.84 | 101.4 | 14.45 | 447.7 | 7020 | 8.986e+05 | 0.0925 | 243.2 | 2.082 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Muon | lr=0.01 | 119.8 | 2.008 | 5.992 | 103.8 | 14.79 | 447.7 | 7020 | 8.986e+05 | 0.09491 | 237.3 | 1.98 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Stochastic L-BFGS | lr=2, max_iter=2, history_size=5 | 127 | 0.2342 | 6.347 | 112.1 | 15.97 | 1062 | 7020 | 8.986e+05 | 0.1005 | 377.1 | 2.97 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| SOAP | lr=0.01 | 247.6 | 0.7947 | 12.38 | 233.1 | 33.21 | 518.2 | 7020 | 8.986e+05 | 0.1961 | 484.7 | 1.958 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Sven | k=128, lr=0.5, rtol=0.3 | 1263 | 2.739 | 63.14 | 1247 | 177.6 | 2.296e+04 | 7020 | 8.986e+05 | 1 | 1328 | 1.051 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
