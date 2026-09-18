**TRACK sven-opt — review fixes applied.** Changed `/n/home11/sambt/iaifi/sv3/sven/sven/opt/sven.py`; `/n/home11/sambt/iaifi/sv3/sven/tests/test_torch_logging.py` (18 → 36 tests). Nothing else touched (`pinv.py` untouched).

### Findings
1. **H, `_record_rank` 0-d CUDA tensor — real, fixed.** Now unconditionally `int(torch.count_nonzero(kept))`. In `SvenGram`/`SvenGramReg` the count instead rides the sync the step already makes: `any_kept, n_kept = torch.stack((keep.any().to(int64), count_nonzero(s_inv_sq))).tolist()`, so the guard and the rank cost **one** transfer (was 2). `finalize_svd_info()` kept, no longer load-bearing. Measured sites/step (`sven/opt` frames only): SvenGram 1 (log) / 1, SvenGramReg 1 / 1, classic 7 / **2** (`pinv.py:63` + the rank int).
2. **H, `_split_diagnostics` seam — real, not my files.** See integrator notes; verified `svs_step = np.arange(0,len(svs),every)` is positional and `si['step']` unused.
3. **M, no sync test — real, added** (`test_non_logged_step_transfer_budget`, exact budget + strictly < logged).
4. **M, `update_norm` pre-lr — real, fixed.** Now the **applied** change `‖lr·update‖ = ‖θ_new−θ_old‖`; asserted against `(p0−p1).norm()` directly; stated in the docstring.
5. **M, fp32 Gram floor — real, fixed.** New key `sv_noise_floor = sqrt(eps(params.dtype))·σ_max`; docstring gives the error law. Replacing finding 7's tautology **exposed this in float64**: the 5-decade `rtol_trunc` case deviates from `svdvals(J)` by 1.3e-5 rel at σ₈, matching `eps·(σ_max/σ_i)²/2` — so `assert_spectrum_matches()` now uses that budget instead of a flat rtol.
6. **L, per-component `(P,)` allocs — fixed** (`clone()`/`add_`, zero sentinel; new test for `k_used==0`).
7. **L, tautology — fixed** (compares `svdvals(jac)`).
8. **L, `SvenGramReg` not in empty_cache parametrize — fixed** (4th site covered); `utr` sign caveat in docstring.
None rejected.

### Acceptance tests — `34 passed, 2 skipped in 3.37s`
`svs == svdvals(J)` full-length at k=4/rtol=0.1 (Gram hooks+chunked, classic); update `== JᵀU[:, :n](utr/s²)`; `log_this_step=False` → only `num_nonzero_svs`, `torch.equal` params; `empty_cache=False` → 0 calls, identical params, and **`empty_cache` default is now False** (`test_empty_cache_defaults_to_false`); `test_svd_info_is_host_side[Sven|SvenGram|SvenGramReg × log]` runs the runner's `np.asarray` calls and rejects any `torch.Tensor`; `test_record_rank_is_host_side_off_cpu` fakes `device=cuda:0` via a `torch.Tensor` subclass (would have caught finding 1 on CPU); 2 CUDA tests skip here. Regression: `56 passed in 8.48s` (`test_torch_gram/gram_reg/residual`); whole suite via `campaign/run_cpu_tests.sh` → `197 passed, 7 skipped in 6.67s` (7 skips = 4 jax, 1 OpenWebText, 2 no-GPU).

### Deviations from CONTRACTS.md
`empty_cache` default **False** (Scope update; the interface line at CONTRACTS.md:85 still says `= True` — needs editing, not my file). `update_norm` now includes `lr`. Extra key `sv_noise_floor`. `SvenGramReg` logs the full record (contract allowed skipping).

### Integrator
- **No call-site change for C-T3**: `generic_scan.py:426/435`, `optimizer_profile.py:229/241` get the fast path from the default; pass `empty_cache=True` only to reproduce legacy runs.
- `generic_scan.py:516` (`result["svd_info"] = getattr(optimizer,"svd_info",{})`) is now GPU-safe as-is; `finalize_svd_info() -> dict[str,Any]` optional/idempotent.
- **C-L2 must still fix** `_split_diagnostics` (generic_scan.py:178-233): index spectra by `si['step']`, drop the positional `svs_step`, compute `sv_max/sv_min` epoch means on step indices, store `utr/update_norm/resid_norm/sv_min_kept/sv_noise_floor`, rename `sv_min`→σ_M (F20). `variable_k_substep_losses` still holds tensors (runner's `float(t)` handles it).
- `SvenGramReg._rows_jvp` depends on `SvenWrapper._pass_norm_stats()` (sven-nn track).

*Relayed user question ("only two review agents at once — can't this be parallelized?") is orchestrator-level; nothing in this track limits fan-out.*