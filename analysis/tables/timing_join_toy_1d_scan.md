| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 2.63e-05 | 2.936e-07 | True | 0 | 0 | 2.63e-05 | svd_bs32_mlp_width16_k32_lr0.01_rtol0.0001_svdtorch_mseed1002_lseed1000_gram |
| Adam | Adam | 5 | 5 | 3.547e-06 | 3.8e-07 | True | 0 | 0 | 3.547e-06 | std_bs32_mlp_width16_lr0.001_optimAdam_mseed1000_lseed1000 |
| AdamW | AdamW | 5 | 5 | 1.681e-06 | 1.031e-06 | True | 0 | 0 | 1.681e-06 | std_bs32_mlp_width16_lr0.001_optimAdamW_wd0.01_mseed1002_lseed1000 |
| HIG | HIG | 5 | 5 | 0.2093 | 0.01375 | False | 0 | 0 | 0.2093 | hig_bs32_mlp_width16_lr0.005_tau1e-06_mseed1002_lseed1000 |
| JD | JD (UPGrad) | 5 | 5 | 9.177e-07 | 6.325e-07 | True | 0 | 0 | 9.177e-07 | jd_bs32_mlp_width16_lr0.01_aggUPGrad_innerAdam_mseed1000_lseed1000 |
| KFAC | KFAC | 5 | 5 | 2.305e-06 | 5.691e-07 | True | 0 | 0 | 2.305e-06 | std_bs32_mlp_width16_lr0.001_optimKFAC_mseed1004_lseed1000 |
| LBFGS | Stochastic L-BFGS | 5 | 5 | 4.747e-07 | 6.872e-08 | True | 0 | 0 | 4.747e-07 | std_bs32_mlp_width16_lr0.03_optimLBFGS_mi3_hs10_lsstrong_wolfe_mseed1000_lseed1000 |
| Muon | Muon | 5 | 5 | 1.597e-06 | 1.068e-06 | True | 0 | 0 | 1.597e-06 | std_bs32_mlp_width16_lr0.001_optimMuon_mseed1003_lseed1000 |
| MuonW | MuonW | 5 | 5 | 4.057e-07 | 2.097e-07 | True | 0 | 0 | 4.057e-07 | std_bs32_mlp_width16_lr0.001_optimMuonW_wd0.1_mseed1002_lseed1000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 1.088e-06 | 2.121e-07 | True | 0 | 0 | 1.088e-06 | std_bs32_mlp_width16_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed1000_lseed1000 |
| RMSprop | RMSprop | 5 | 5 | 2.29e-06 | 1.935e-07 | True | 0 | 0 | 2.29e-06 | std_bs32_mlp_width16_lr0.001_optimRMSprop_mseed1000_lseed1000 |
| SGD | SGD | 5 | 5 | 7.585e-07 | 4.404e-07 | True | 0 | 0 | 7.585e-07 | std_bs32_mlp_width16_lr0.3_optimSGD_mseed1001_lseed1000 |
| SGDm | SGD + momentum | 5 | 5 | 1.593e-06 | 8.655e-07 | True | 0 | 0 | 1.593e-06 | std_bs32_mlp_width16_lr0.1_optimSGDm_mseed1002_lseed1000 |
| SOAP | SOAP | 5 | 5 | 4.207e-06 | 1.193e-06 | True | 0 | 0 | 4.207e-06 | std_bs32_mlp_width16_lr0.001_optimSOAP_mseed1001_lseed1000 |
| Shampoo | Shampoo | 5 | 5 | 2.136e-06 | 6.128e-07 | True | 0 | 0 | 2.136e-06 | std_bs32_mlp_width16_lr0.1_optimShampoo_mseed1000_lseed1000 |
