| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 0.01793 | 0.01141 | False | 0 | 0 | 0.01793 | svd_bs128_k128_lr0.1_rtol0.01_svdtorch_mseed4000_lseed4000_gram_bnbatch |
| Adam | Adam | 5 | 5 | 0.09787 | 0.05606 | False | 0 | 0 | 0.09787 | std_bs128_lr0.001_optimAdam_mseed4001_lseed4000 |
| AdamW | AdamW | 5 | 5 | 0.1241 | 0.07331 | False | 0 | 0 | 0.1241 | std_bs128_lr0.01_optimAdamW_wd0.01_mseed4003_lseed4000 |
| LBFGS | Stochastic L-BFGS | 5 | 5 | 0.1572 | 0.0861 | False | 0 | 0 | 0.1572 | std_bs128_lr2.0_optimLBFGS_mi2_hs5_lsstrong_wolfe_mseed4000_lseed4000 |
| Muon | Muon | 5 | 5 | 0.1216 | 0.05746 | False | 0 | 0 | 0.1216 | std_bs128_lr0.01_optimMuon_mseed4004_lseed4000 |
| MuonW | MuonW | 5 | 5 | 0.09216 | 0.02801 | False | 0 | 0 | 0.09216 | std_bs128_lr0.01_optimMuonW_wd0.1_mseed4004_lseed4000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 0.02632 | 0.01329 | False | 0 | 0 | 0.02632 | std_bs128_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed4000_lseed4000 |
| RMSprop | RMSprop | 5 | 5 | 0.05678 | 0.0421 | False | 0 | 0 | 0.05678 | std_bs128_lr0.001_optimRMSprop_mseed4002_lseed4000 |
| SGD | SGD | 5 | 5 | 0.08894 | 0.01573 | False | 0 | 0 | 0.08894 | std_bs128_lr3.0_optimSGD_mseed4003_lseed4000 |
| SGDm | SGD + momentum | 5 | 5 | 0.2077 | 0.1433 | False | 0 | 0 | 0.2077 | std_bs128_lr0.3_optimSGDm_mseed4002_lseed4000 |
| SOAP | SOAP | 5 | 5 | 0.06642 | 0.03683 | False | 0 | 0 | 0.06642 | std_bs128_lr0.01_optimSOAP_mseed4004_lseed4000 |
