# Problem: Why Does Sven Struggle on Classification vs Regression?

## Status: Active

## Background

The Sven optimizer (SVD-based pseudo-inverse optimizer) shows markedly different behavior on classification vs regression tasks:

- **Regression** (e.g., MNIST label regression with MSE on one-hot targets): Sven performs well, matching or approaching Adam/SGD performance without much hyperparameter tuning.
- **Classification** (e.g., MNIST with cross-entropy): Sven struggles to outperform Adam. Getting competitive results seems to require adding RMSprop-style adaptive learning rates (`use_rmsprop: true`), which undermines the theoretical elegance of the SVD approach.

## The Core Question

**Why does the choice of loss function (MSE regression vs cross-entropy classification) so dramatically affect Sven's performance, and what does this tell us about when SVD-based optimization is appropriate?**

## Specific Sub-Questions

1. **Loss landscape geometry**: How does the Jacobian structure differ between cross-entropy and MSE? Does cross-entropy create ill-conditioned Jacobians that make the pseudo-inverse unstable?

2. **Singular value spectrum**: How does the distribution of singular values of the Jacobian differ between the two loss types? Does cross-entropy lead to a more spread spectrum that makes truncation at rank k more lossy?

3. **Update magnitude/direction**: Are the SVD updates for cross-entropy pointing in fundamentally different directions than gradient descent updates? Are the magnitudes pathological?

4. **Role of RMSprop**: Why does adding RMSprop-style scaling help for classification? What is it correcting for? Is it normalizing away some pathology in the raw SVD updates?

5. **Softmax/log-softmax effects**: Is the issue specifically about the log-softmax in cross-entropy, or would any "sharp" loss function cause problems? What about using MSE loss on softmax outputs (soft classification)?

6. **Interpolation**: What happens with loss functions that interpolate between MSE and CE? (e.g., label smoothing, focal loss, squared hinge loss)

## Suggested Experiments

### Phase 1: Characterize the Problem
- Run Sven (no RMSprop) on MNIST with CE loss and MSE label regression, same model/hyperparameters. Compare loss curves, accuracy curves.
- Log singular value spectra of the Jacobian during training for both loss types.
- Log the norm and direction (cosine similarity with gradient) of the SVD update vs standard gradient.

### Phase 2: Diagnose the Cause
- Try intermediate loss functions: MSE on softmax outputs, label-smoothed CE, focal loss.
- Analyze how the condition number of the Jacobian evolves during training for CE vs MSE.
- Compare the effective rank (number of significant singular values) across loss types.

### Phase 3: Potential Fixes
- Test whether simple rescaling of the SVD update (without full RMSprop) helps.
- Try different truncation strategies (adaptive rtol, different k schedules).
- Investigate whether the issue is worse at certain training phases (early vs late).

## Existing Data

Check `experiments/configs/mnist_scan.yaml` and `mnist_scan_labelRegression.yaml` for existing experiment configs. Results (if any) would be in `experiment_results/`.

## Constraints

- Use MNIST as the primary testbed (fast to iterate)
- MLP architecture (start with width 32 as in existing configs, can vary)
- Keep batch sizes manageable (64-256) given memory constraints of the Jacobian computation
- Always compare against Adam as a baseline
