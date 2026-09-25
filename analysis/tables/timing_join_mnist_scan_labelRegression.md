| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 0.001263 | 0.0001418 | False | 0 | 0 | 0.001263 | svd_bs64_mlp_width32_k64_lr0.5_rtol0.001_svdtorch_mseed3003_lseed3000_gram |
| Adam | Adam | 5 | 5 | 0.01851 | 0.01012 | False | 0 | 0 | 0.01851 | std_bs64_mlp_width32_lr0.001_optimAdam_mseed3002_lseed3000 |
| AdamW | AdamW | 5 | 5 | 0.01015 | 0.005051 | False | 0 | 0 | 0.01015 | std_bs64_mlp_width32_lr0.001_optimAdamW_wd0.01_mseed3003_lseed3000 |
| HIG | HIG | 5 | 5 | 0.003892 | 0.001528 | False | 0 | 0 | 0.003892 | hig_bs64_mlp_width32_lr0.05_tau0.03_mseed3000_lseed3000 |
| JD | JD (UPGrad) | 5 | 5 | 0.04348 | 0.01926 | False | 0 | 0 | 0.04348 | jd_bs64_mlp_width32_lr0.001_aggUPGrad_innerAdam_mseed3000_lseed3000 |
| LBFGS | Stochastic L-BFGS | 5 | 5 | 0.1978 | 0.0253 | False | 0 | 0 | 0.1978 | std_bs64_mlp_width32_lr0.1_optimLBFGS_mi3_hs2_lsstrong_wolfe_mseed3000_lseed3000 |
| Muon | Muon | 5 | 5 | 0.03761 | 0.01766 | False | 0 | 0 | 0.03761 | std_bs64_mlp_width32_lr0.001_optimMuon_mseed3001_lseed3000 |
| MuonW | MuonW | 5 | 5 | 0.01626 | 0.00706 | False | 0 | 0 | 0.01626 | std_bs64_mlp_width32_lr0.001_optimMuonW_wd0.1_mseed3000_lseed3000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 0.05488 | 0.02902 | False | 0 | 0 | 0.05488 | std_bs64_mlp_width32_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed3001_lseed3000 |
| RMSprop | RMSprop | 5 | 5 | 0.05202 | 0.02377 | False | 0 | 0 | 0.05202 | std_bs64_mlp_width32_lr0.001_optimRMSprop_mseed3002_lseed3000 |
| SGD | SGD | 5 | 5 | 2.452e-07 | 1.641e-07 | True | 0 | 0 | 2.452e-07 | std_bs64_mlp_width32_lr0.1_optimSGD_mseed3003_lseed3000 |
| SGDm | SGD + momentum | 5 | 5 | 2.379e-07 | 1.978e-08 | True | 0 | 0 | 2.379e-07 | std_bs64_mlp_width32_lr0.01_optimSGDm_mseed3003_lseed3000 |
| SOAP | SOAP | 5 | 4 | 0.2058 | 0.02498 | False | 1 | 0 | 204.2 | std_bs64_mlp_width32_lr0.01_optimSOAP_mseed3001_lseed3000 |
| Shampoo | Shampoo | 5 | 5 | 0.04531 | 0.02083 | False | 0 | 0 | 0.04531 | std_bs64_mlp_width32_lr0.1_optimShampoo_mseed3002_lseed3000 |
