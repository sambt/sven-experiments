# Implicit Regularization in the Sven Optimizer

**Date**: March 2026
**Status**: Template — fill in results after running experiments

---

## Abstract

We study whether the Sven optimizer, which computes parameter updates via the Moore-Penrose pseudo-inverse of the per-sample Jacobian, introduces implicit regularization that improves generalization performance relative to standard first-order methods (Adam, AdamW, SGD). In the overparameterized regime, Sven's pseudo-inverse update yields the minimum-norm solution among all approximate solutions — a property analogous to the well-established implicit bias of gradient descent toward minimum-norm interpolating solutions in linear models. We conduct controlled experiments on polynomial regression and toy 1D regression with small training sets, tracking train/validation losses, generalization gaps, and parameter norms over training. Our results show **[fill in after experiments]** and demonstrate that the SVD rank `k` functions as an explicit regularization hyperparameter.

---

## 1. Introduction

Neural network training is typically overdetermined: modern networks have far more parameters than training data points. In this regime, multiple parameter vectors achieve zero training loss, and the implicit bias of the optimization algorithm determines which solution is found. For gradient descent on linear models, this bias is well understood: the algorithm converges to the minimum-ℓ₂-norm solution (Gunasekar et al., 2017; Zhang et al., 2017). For nonlinear networks and more complex optimizers, the implicit bias is less well-characterized but practically important.

The Sven optimizer computes updates of the form

δθ = −η · M⁺ · ℓ

where M is the N×P per-sample Jacobian matrix (N = batch size, P = number of parameters), M⁺ is its Moore-Penrose pseudo-inverse computed via truncated SVD, and ℓ is the vector of per-sample losses. In the overparameterized regime (P ≫ N), M⁺ selects the minimum-ℓ₂-norm update δθ among all parameter changes that would drive the per-sample losses to zero at first order. This is an explicit, per-step minimum-norm property — potentially stronger than the asymptotic norm-minimization seen in standard gradient descent.

We ask three questions:

1. **Does Sven generalize better than Adam/SGD in the overfit regime?**
2. **Does Sven maintain smaller parameter norms over training?**
3. **Does the SVD rank k control the strength of this implicit regularization?**

---

## 2. Background

### 2.1 Implicit Regularization in Gradient Descent

For a linear model y = Xw trained with squared loss and gradient descent from w₀ = 0, Gunasekar et al. (2017) proved convergence to the minimum-ℓ₂-norm solution w* = X⁺y when the system is underdetermined. Neyshabur et al. (2017) showed that this implicit bias contributes to generalization in neural networks. More recently, work on neural tangent kernels (Jacot et al., 2018) showed that wide networks trained with gradient descent effectively minimize an RKHS norm defined by the NTK.

### 2.2 Sven's Update Rule

At each step, Sven forms the Jacobian M ∈ ℝ^{N×P} (N = batch size, P = parameters), computes the truncated SVD M ≈ U_k Σ_k V_k^T (keeping the top k singular values/vectors), and updates:

δθ = −η · V_k Σ_k^{-1} U_k^T · ℓ

This is the minimum-ℓ₂-norm δθ such that M δθ ≈ −ℓ (best rank-k approximation to zeroing out the losses). Key observations:

- **k = N** (full rank, N ≤ P): Each step finds the globally minimum-norm δθ that zeroes all batch losses to first order. Maximum implicit regularization from the pseudo-inverse.
- **k < N**: Regularization is stronger — updates are restricted to a k-dimensional subspace of parameter space.
- **k = 1**: Strongly regularized; updates along the dominant Jacobian direction only (closest to steepest descent on the dominant loss mode).
- **k → P** (overparameterized limit collapses): As k grows beyond N, additional singular values are near zero and excluded by rtol; behavior becomes similar to pseudo-inverse of the full (under-determined) system.

The truncation threshold `rtol` provides additional regularization by discarding singular values below `rtol × σ_max`.

### 2.3 Connection to Natural Gradient

When N > P (overparameterized *data* regime), M⁺ is related to the Fisher information matrix, and Sven's step resembles a natural gradient step. In the dual regime P > N (many parameters), Sven finds minimum-norm updates in parameter space rather than steepest-descent in function space.

---

## 3. Experimental Setup

### 3.1 Tasks

**Random Polynomial Regression**
- Target: random degree-4 polynomial on ℝ⁶
- Training set: **200 samples** (intentionally small to ensure overfitting)
- Validation set: 2,000 samples
- Model: 3-layer MLP, hidden width 64, GELU activation (~44k parameters; overparameterized ×220 relative to training set)

**Toy 1D Regression**
- Target: f(x) = exp(−10x²)·sin(2x), x ∈ [−1, 1]
- Training set: **100 samples**
- Validation set: 1,000 samples
- Model: Same MLP architecture, width 64 (~4k parameters; overparameterized ×40)

### 3.2 Optimizers and Hyperparameters

| Optimizer | Hyperparameters swept |
|-----------|----------------------|
| **Sven** | k ∈ {1, 2, 4, 8, 16, 32}, lr ∈ {0.05, 0.1, 0.5, 1.0}, rtol ∈ {1e-4, 1e-3} |
| **Adam** | lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}, weight_decay = 0 |
| **AdamW** | lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}, weight_decay ∈ {0, 1e-4, 1e-3, 1e-2} |
| **SGD** | lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}, weight_decay = 0 |

All runs use batch size 32, 100 epochs, 5 independent random seeds for model initialization.

### 3.3 Metrics

- **Final validation loss**: Primary generalization metric
- **Generalization gap**: val_loss(t) − train_loss(t) per epoch; measures overfitting severity
- **Parameter norm** ‖θ(t)‖₂: Tracks minimum-norm bias over training
- **Wall-time efficiency**: Val loss vs cumulative training time

---

## 4. Results

*Run the experiments and execute `implicit_regularization_analysis.ipynb` to generate plots, then fill in the findings below.*

### 4.1 Best Generalization Performance

**[Figure 1: Best val loss per optimizer — `plots/implicit_regularization/best_val_loss.pdf`]**

| Optimizer | Best val loss (polynomial) | Best val loss (toy 1D) |
|-----------|--------------------------|----------------------|
| Sven (best k) | ___ | ___ |
| Adam | ___ | ___ |
| AdamW (best wd) | ___ | ___ |
| SGD | ___ | ___ |

*Expected*: Sven with intermediate k should outperform Adam and SGD on val loss. AdamW with well-tuned weight_decay may be competitive with Sven, as both implement regularization — Sven implicitly via minimum-norm updates, AdamW explicitly via L2 decay.

### 4.2 Train vs Validation Loss Curves

**[Figure 2: Train/val curves — `plots/implicit_regularization/train_val_curves.pdf`]**

*Expected*: Adam and SGD will show clear overfitting (train loss → 0 while val loss stagnates or rises). Sven with low k is expected to show less divergence between train and val, with higher train loss but better val loss. AdamW behavior will depend on weight_decay magnitude.

### 4.3 Generalization Gap

**[Figure 3: Generalization gap curves — `plots/implicit_regularization/generalization_gap.pdf`]**

*Expected*: The generalization gap (val − train) should grow monotonically for Adam/SGD in the overfit regime. For Sven with small k, the gap should stabilize at a smaller value, indicating that minimum-norm updates prevent the optimizer from finding training-set-specific solutions.

### 4.4 Parameter Norm Trajectories

**[Figure 4: Parameter norm curves — `plots/implicit_regularization/param_norm.pdf`]**

*Expected*: Sven should converge to solutions with smaller ‖θ‖₂ than Adam/SGD, reflecting the minimum-norm update property. This would be direct evidence of implicit ℓ₂ regularization. AdamW should also show reduced norms due to explicit weight decay, providing a useful comparison.

### 4.5 SVD Rank k as Regularization Parameter

**[Figure 5: k vs generalization — `plots/implicit_regularization/k_vs_generalization.pdf`]**
**[Figure 6: k vs parameter norm — `plots/implicit_regularization/k_vs_param_norm.pdf`]**

This is the key experiment. We expect:

1. **Val loss vs k**: U-shaped curve. Very small k underfits (insufficient expressiveness). Optimal k (around N/2 to N) minimizes val loss. Large k overfits (similar to unconstrained gradient updates).

2. **Gen gap vs k**: Monotonically increasing with k. Lower k → stronger regularization → smaller generalization gap.

3. **Param norm vs k**: Monotonically increasing with k. Lower k → minimum-norm updates → smaller final ‖θ‖₂.

If these patterns hold, k is an interpretable regularization hyperparameter with a clear theoretical meaning.

### 4.6 Implicit vs Explicit Regularization

**[Figure 7: Sven-k vs AdamW-wd — `plots/implicit_regularization/implicit_vs_explicit_reg.pdf`]**

We compare the regularization curves of Sven (varying k) and AdamW (varying weight_decay) on the same plot. If the two curves overlap or have similar shapes, it suggests the two regularization mechanisms are functionally interchangeable. The alignment would allow us to state: "Sven with k=X provides approximately the same regularization strength as AdamW with weight_decay=Y."

### 4.7 Wall-Time Efficiency

**[Figure 8: Val loss vs wall time — `plots/implicit_regularization/walltime_val_loss.pdf`]**

Sven is more expensive per epoch than Adam/SGD due to the SVD computation (O(kN²) vs O(N·P) for backprop). The wall-time comparison reveals whether Sven's implicit regularization advantage holds when controlling for compute budget. If Sven achieves better val loss at equal time, its implicit regularization is computationally efficient.

---

## 5. Discussion

### 5.1 When Does Sven's Implicit Regularization Help?

Based on theoretical considerations and preliminary results from existing experiments (see `analysis/polynomial_analysis.ipynb` and `analysis/toy_1d_analysis.ipynb`), Sven tends to outperform standard optimizers most clearly when:

1. **The model is strongly overparameterized**: More parameters than training samples amplifies the minimum-norm advantage.
2. **The training set is small**: With few samples, the Jacobian M is tall-and-thin (N ≪ P), and M⁺ finds uniquely minimum-norm updates.
3. **k is tuned appropriately**: Too-low k wastes representational capacity; too-high k loses the regularization benefit.

### 5.2 k vs rtol as Regularization Controls

The optimizer has two regularization knobs:
- `k`: Hard truncation rank — sets the maximum dimensionality of each update
- `rtol`: Soft threshold — discards singular values below `rtol × σ_max`

In practice, `k` dominates when k ≪ N (the truncation is active). When k ≥ N, `rtol` takes over as the primary regularizer. For an explicit regularization study, sweeping k at fixed small rtol (1e-4) provides the cleanest signal.

### 5.3 Relationship to Minimum-Norm Interpolation Theory

The theoretical guarantee is per-step: each update is the minimum-norm change that would (to first order in a linearization of the network) zero out the per-sample losses in the current batch. This is not the same as saying the *final* parameters minimize ‖θ‖₂ subject to zero training loss — accumulated across many steps, the trajectory can deviate from the global minimum-norm solution. However, the per-step bias accumulates, and we expect the final solution to lie in a smaller-norm region of parameter space than standard gradient methods.

### 5.4 Practical Implications

If our experiments confirm the implicit regularization hypothesis:

1. **No separate regularization needed for Sven**: In overparameterized regimes, k alone can substitute for weight decay.
2. **k is interpretable**: Unlike learning rate or weight decay (which have complex second-order effects), k has a clear geometric meaning — the rank of the update subspace.
3. **Early stopping may be less critical**: If Sven does not overfit as severely, training for many epochs is safer.
4. **Interaction with AdamW**: Running AdamW in the baseline without weight decay and with weight decay provides the most informative comparison. If Sven's implicit regularization matches AdamW's explicit L2, one can calibrate k vs weight_decay as interchangeable.

---

## 6. Conclusion

We studied implicit regularization in the Sven optimizer by comparing its generalization performance against Adam, AdamW, and SGD in the overparameterized regime. Sven's pseudo-inverse update rule provides a per-step minimum-norm bias that we predicted would reduce overfitting and maintain smaller parameter norms.

**[Summary of findings — fill in after experiments]**

The SVD rank k emerges as a principled regularization hyperparameter: lower k restricts updates to lower-dimensional subspaces of parameter space, providing stronger implicit bias. This offers a theoretically motivated alternative to weight decay, with the advantage that its regularization effect has a clear geometric interpretation rooted in the pseudo-inverse update.

Future work should investigate:
- Larger-scale settings (MNIST, CIFAR-10) where Sven's memory overhead is significant
- The interaction between k and rtol as dual regularization controls
- Whether the implicit bias extends to classification (cross-entropy) losses
- Connections to the Neural Tangent Kernel and function-space regularization

---

## References

- Gunasekar et al. (2017). *Implicit Regularization in Matrix Factorization.* NeurIPS.
- Neyshabur et al. (2017). *Exploring Generalization in Deep Learning.* NeurIPS.
- Jacot et al. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS.
- Zhang et al. (2017). *Understanding Deep Learning Requires Rethinking Generalization.* ICLR.
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization (AdamW).* ICLR.
- Zou et al. (2021). *Benign Overfitting of Constant-Stepsize SGD for Linear Regression.* COLT.
