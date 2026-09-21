| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 1.717e-08 | 3.041e-09 | True | 0 | 0 | 1.717e-08 | svd_bs32_mlp_width16_k16_lr0.5_rtol0.03_svdtorch_mseed2000_lseed2000_gram |
| Adam | Adam | 5 | 5 | 1.17e-08 | 9.481e-09 | True | 0 | 0 | 1.17e-08 | std_bs32_mlp_width16_lr0.01_optimAdam_mseed2004_lseed2000 |
| AdamW | AdamW | 5 | 5 | 9.882e-09 | 5.984e-09 | True | 0 | 0 | 9.882e-09 | std_bs32_mlp_width16_lr0.01_optimAdamW_wd0.01_mseed2004_lseed2000 |
| HIG | HIG | 5 | 5 | 4.845e-05 | 2.568e-06 | True | 0 | 0 | 4.845e-05 | hig_bs32_mlp_width16_lr0.1_tau1e-08_mseed2003_lseed2000 |
| JD | JD (UPGrad) | 5 | 5 | 2.194e-08 | 8.88e-09 | True | 0 | 0 | 2.194e-08 | jd_bs32_mlp_width16_lr0.001_aggUPGrad_innerAdam_mseed2000_lseed2000 |
| KFAC | KFAC | 5 | 5 | 1.586e-08 | 8.29e-09 | True | 0 | 0 | 1.586e-08 | std_bs32_mlp_width16_lr1e-05_optimKFAC_mseed2001_lseed2000 |
| LBFGS | Stochastic L-BFGS | 5 | 3 | 2.538e-08 | 8.681e-09 | True | 0 | 0 | 2.538e-08 | std_bs32_mlp_width16_lr1.0_optimLBFGS_mi1_hs2_lsstrong_wolfe_mseed2003_lseed2000 |
| Muon | Muon | 5 | 5 | 0.2526 | 0.07043 | False | 0 | 0 | 0.2526 | std_bs32_mlp_width16_lr0.01_optimMuon_mseed2000_lseed2000 |
| MuonW | MuonW | 5 | 5 | 0.2137 | 0.04433 | False | 0 | 0 | 0.2137 | std_bs32_mlp_width16_lr0.01_optimMuonW_wd0.1_mseed2000_lseed2000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 1.339e-08 | 1.06e-08 | True | 0 | 0 | 1.339e-08 | std_bs32_mlp_width16_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed2002_lseed2000 |
| RMSprop | RMSprop | 5 | 5 | 1.27e-08 | 3.17e-09 | True | 0 | 0 | 1.27e-08 | std_bs32_mlp_width16_lr0.01_optimRMSprop_mseed2003_lseed2000 |
| SGD | SGD | 5 | 5 | 1.493e-08 | 8.922e-09 | True | 0 | 0 | 1.493e-08 | std_bs32_mlp_width16_lr0.1_optimSGD_mseed2004_lseed2000 |
| SGDm | SGD + momentum | 5 | 5 | 8.287e-09 | 3.507e-09 | True | 0 | 0 | 8.287e-09 | std_bs32_mlp_width16_lr0.01_optimSGDm_mseed2004_lseed2000 |
| SOAP | SOAP | 5 | 5 | 1.034e-08 | 5.796e-09 | True | 0 | 0 | 1.034e-08 | std_bs32_mlp_width16_lr0.01_optimSOAP_mseed2002_lseed2000 |
| Shampoo | Shampoo | 5 | 5 | 1.531e-08 | 3.868e-09 | True | 0 | 0 | 1.531e-08 | std_bs32_mlp_width16_lr0.1_optimShampoo_mseed2002_lseed2000 |
