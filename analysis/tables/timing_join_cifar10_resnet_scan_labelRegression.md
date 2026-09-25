| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 0.08251 | 0.01085 | False | 0 | 0 | 0.08251 | svd_bs128_k128_lr0.5_rtol0.001_svdtorch_mseed4000_lseed4000_gram_bnbatch |
| Adam | Adam | 5 | 5 | 0.04914 | 0.02874 | False | 0 | 0 | 0.04914 | std_bs128_lr0.01_optimAdam_mseed4002_lseed4000 |
| AdamW | AdamW | 5 | 5 | 0.05862 | 0.02591 | False | 0 | 0 | 0.05862 | std_bs128_lr0.01_optimAdamW_wd0.01_mseed4001_lseed4000 |
| LBFGS | Stochastic L-BFGS | 5 | 5 | 0.2275 | 0.03738 | False | 0 | 0 | 0.2275 | std_bs128_lr2.0_optimLBFGS_mi3_hs2_lsstrong_wolfe_mseed4001_lseed4000 |
| Muon | Muon | 5 | 5 | 0.04151 | 0.02529 | False | 0 | 0 | 0.04151 | std_bs128_lr0.01_optimMuon_mseed4002_lseed4000 |
| MuonW | MuonW | 5 | 5 | 0.059 | 0.04429 | False | 0 | 0 | 0.059 | std_bs128_lr0.01_optimMuonW_wd0.1_mseed4001_lseed4000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 0.06402 | 0.04333 | False | 0 | 0 | 0.06402 | std_bs128_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed4001_lseed4000 |
| RMSprop | RMSprop | 5 | 5 | 0.1284 | 0.06048 | False | 0 | 0 | 0.1284 | std_bs128_lr0.001_optimRMSprop_mseed4000_lseed4000 |
| SGD | SGD | 5 | 5 | 0.02174 | 0.01274 | False | 0 | 0 | 0.02174 | std_bs128_lr0.001_optimSGD_mseed4001_lseed4000 |
| SGDm | SGD + momentum | 5 | 5 | 0.02251 | 0.01812 | False | 0 | 0 | 0.02251 | std_bs128_lr0.001_optimSGDm_mseed4003_lseed4000 |
| SOAP | SOAP | 5 | 5 | 0.05124 | 0.02033 | False | 0 | 0 | 0.05124 | std_bs128_lr0.01_optimSOAP_mseed4002_lseed4000 |
