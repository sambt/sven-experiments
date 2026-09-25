| method | display | n_timing | n_joined | max_rel_dev | median_rel_dev | bit_reproduced | n_diverged_only_in_timing | n_diverged_only_in_scan | max_rel_dev_raw | worst_run_raw |
|---|---|---|---|---|---|---|---|---|---|---|
| Sven | Sven | 5 | 5 | 0.038 | 0.01066 | False | 0 | 0 | 0.038 | svd_bs64_mlp_width32_k32_lr0.5_rtol0.3_svdtorch_mseed3002_lseed3000_gram |
| Adam | Adam | 5 | 5 | 8.074e-08 | 3.407e-08 | True | 0 | 0 | 8.074e-08 | std_bs64_mlp_width32_lr0.0001_optimAdam_mseed3003_lseed3000 |
| AdamW | AdamW | 5 | 5 | 0.004556 | 7.879e-05 | False | 0 | 0 | 0.004556 | std_bs64_mlp_width32_lr0.001_optimAdamW_wd0.01_mseed3004_lseed3000 |
| HIG | HIG | 5 | 5 | 0.0001424 | 8.614e-05 | True | 0 | 0 | 0.0001424 | hig_bs64_mlp_width32_lr0.05_tau0.03_mseed3004_lseed3000 |
| JD | JD (UPGrad) | 5 | 5 | 8.699e-08 | 4.133e-08 | True | 0 | 0 | 8.699e-08 | jd_bs64_mlp_width32_lr0.0001_aggUPGrad_innerAdam_mseed3004_lseed3000 |
| LBFGS | Stochastic L-BFGS | 5 | 5 | 0.3082 | 0.1044 | False | 0 | 0 | 0.3082 | std_bs64_mlp_width32_lr0.5_optimLBFGS_mi1_hs2_lsstrong_wolfe_mseed3000_lseed3000 |
| Muon | Muon | 5 | 5 | 0.0002538 | 0.0001599 | True | 0 | 0 | 0.0002538 | std_bs64_mlp_width32_lr0.0001_optimMuon_mseed3004_lseed3000 |
| MuonW | MuonW | 5 | 5 | 0.0001916 | 8.367e-05 | True | 0 | 0 | 0.0001916 | std_bs64_mlp_width32_lr0.0001_optimMuonW_wd0.1_mseed3000_lseed3000 |
| PolyakSGD | Polyak SGD | 5 | 5 | 0.02163 | 0.0005764 | False | 0 | 0 | 0.02163 | std_bs64_mlp_width32_optimPolyakSGD_fstar0.0_maxlr1.0_eps1e-08_mseed3004_lseed3000 |
| RMSprop | RMSprop | 5 | 5 | 4.507e-08 | 3.986e-08 | True | 0 | 0 | 4.507e-08 | std_bs64_mlp_width32_lr0.0001_optimRMSprop_mseed3000_lseed3000 |
| SGD | SGD | 5 | 5 | 1.108e-07 | 7.03e-08 | True | 0 | 0 | 1.108e-07 | std_bs64_mlp_width32_lr0.01_optimSGD_mseed3003_lseed3000 |
| SGDm | SGD + momentum | 5 | 5 | 1.085e-07 | 4.142e-08 | True | 0 | 0 | 1.085e-07 | std_bs64_mlp_width32_lr0.001_optimSGDm_mseed3004_lseed3000 |
| SOAP | SOAP | 5 | 5 | 0.1334 | 0.004163 | False | 0 | 0 | 0.1334 | std_bs64_mlp_width32_lr0.0001_optimSOAP_mseed3000_lseed3000 |
| Shampoo | Shampoo | 5 | 5 | 0.01476 | 0.00363 | False | 0 | 0 | 0.01476 | std_bs64_mlp_width32_lr0.1_optimShampoo_mseed3004_lseed3000 |
