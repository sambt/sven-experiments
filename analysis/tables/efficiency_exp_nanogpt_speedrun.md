| display | config | wall_s | wall_s_std | epoch_s | sync_train_s | ms_per_step | peak_gpu_mem_mb | steps | examples | wall_vs_sven | scan_wall_s | scan_inflation | gpu_timing | gpu_scan | gpu_scan_frac | fin_timing | att_timing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdamW | lr=0.0003, weight_decay=0.01 | 45.7 | 0.1398 | 0.9139 | 38.99 | 7.22 | 329.7 | 5400 | 3.456e+05 | 0.3802 | 166.3 | 3.638 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| MuonW | lr=0.0001, weight_decay=0.1 | 87.69 | 0.272 | 1.754 | 81.04 | 15.01 | 326.5 | 5400 | 3.456e+05 | 0.7295 | 225.2 | 2.568 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Muon | lr=0.0001 | 88.24 | 0.2544 | 1.765 | 81.55 | 15.1 | 326.5 | 5400 | 3.456e+05 | 0.7341 | 223.9 | 2.537 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| Sven | k=64, lr=0.1, rtol=0.001 | 120.2 | 0.2529 | 2.404 | 112.7 | 20.86 | 615.4 | 5400 | 3.456e+05 | 1 | 347.6 | 2.891 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
| SOAP | lr=0.0001 | 136.5 | 0.3368 | 2.73 | 129.6 | 23.99 | 352.3 | 5400 | 3.456e+05 | 1.136 | 303 | 2.219 | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 1 | 5 | 5 |
