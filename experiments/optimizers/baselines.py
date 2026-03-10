"""
Baseline optimizers for Sven comparison experiments.
Implements: Lion, Muon, ScheduleFreeAdamW, ScheduleFreeSGD
"""

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Lion (Chen et al., NeurIPS 2023)
# Evolved Sign Momentum: update = sign(beta1 * m + (1-beta1) * g)
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    """
    Lion optimizer (Evolved Sign Momentum).
    Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Update: sign(beta1 * m + (1-beta1) * g)
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # Momentum update (different from the update above)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# ---------------------------------------------------------------------------
# Muon (Jordan et al., 2024)
# SGD with Nesterov momentum + Newton-Schulz orthogonalization for 2D params
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the polar factor of G (orthogonalization).
    Coefficients from Jordan et al. 2024 (Muon paper).
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype = G.dtype
    X = G.float() / (G.norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(dtype)


class Muon(Optimizer):
    """
    Muon optimizer (MomentUm Orthogonalized by Newton-schulz).
    Applies Newton-Schulz orthogonalization to 2D weight gradients.
    For 1D parameters (biases), falls back to SGD with momentum.
    Reference: Jordan et al., 2024.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # Effective gradient (Nesterov or classical heavy ball)
                g = grad.add(buf, alpha=momentum) if nesterov else buf

                # Orthogonalize if 2D weight matrix
                if g.ndim == 2:
                    g = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                    # Scale to preserve RMS of updates
                    scale = max(g.shape[0], g.shape[1]) ** 0.5
                    g = g.mul_(scale)

                p.add_(g, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Schedule-Free AdamW (Defazio et al., NeurIPS 2024 Oral)
# No learning rate schedule required; maintains cumulative average x of iterates z.
# Parameters store y = lerp(z, x, beta1) during training, x during eval.
# ---------------------------------------------------------------------------

class ScheduleFreeAdamW(Optimizer):
    """
    Schedule-Free AdamW.
    Reference: Defazio et al., "The Road Less Scheduled", NeurIPS 2024 Oral.

    Usage:
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        # Training: params store y = (1-beta1)*z + beta1*x
        # Before eval: call optimizer.eval() → params store x
        # After eval:  call optimizer.train() → params store y
    """

    schedule_free = True  # marker for train_loop_standard

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        self._mode = 'train'  # 'train' (params = y) or 'eval' (params = x)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            warmup = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    # p.data is y = z = x at initialization
                    state['step'] = 0
                    state['z'] = p.data.clone()
                    state['x'] = p.data.clone()
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                z = state['z']
                x = state['x']

                # Learning rate warmup
                effective_lr = lr * min(1.0, t / max(warmup, 1)) if warmup > 0 else lr

                # Update moments (computed using grad at y = p.data)
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected estimates
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m_hat = state['exp_avg'] / bc1
                v_hat = state['exp_avg_sq'] / bc2

                # Decoupled weight decay on z
                if wd != 0:
                    z.mul_(1 - effective_lr * wd)

                # Adam update on z
                z.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-effective_lr)

                # Update x as cumulative average of z: x_{t} = ((t-1)*x + z) / t
                x.lerp_(z, 1.0 / t)

                # Restore p.data to y = (1-beta1)*z + beta1*x
                p.data.copy_(z).lerp_(x, beta1)

        self._mode = 'train'
        return loss

    @torch.no_grad()
    def eval(self):
        """Switch parameters from y to x (for evaluation)."""
        if self._mode == 'eval':
            return
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'x' not in state:
                    continue  # not yet initialized
                p.data.copy_(state['x'])
        self._mode = 'eval'

    @torch.no_grad()
    def train(self):
        """Switch parameters from x back to y (for training)."""
        if self._mode == 'train':
            return
        for group in self.param_groups:
            beta1 = group['betas'][0]
            for p in group['params']:
                state = self.state[p]
                if 'z' not in state:
                    continue  # not yet initialized
                # Recompute y = (1-beta1)*z + beta1*x
                p.data.copy_(state['z']).lerp_(state['x'], beta1)
        self._mode = 'train'


# ---------------------------------------------------------------------------
# Schedule-Free SGD (Defazio et al., NeurIPS 2024 Oral)
# ---------------------------------------------------------------------------

class ScheduleFreeSGD(Optimizer):
    """
    Schedule-Free SGD with momentum.
    Reference: Defazio et al., "The Road Less Scheduled", NeurIPS 2024 Oral.
    """

    schedule_free = True  # marker for train_loop_standard

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, warmup_steps=0):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        self._mode = 'train'

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            warmup = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['z'] = p.data.clone()
                    state['x'] = p.data.clone()
                    state['momentum_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                z = state['z']
                x = state['x']

                effective_lr = lr * min(1.0, t / max(warmup, 1)) if warmup > 0 else lr

                # Weight decay (L2 regularization on z)
                if wd != 0:
                    grad = grad.add(z, alpha=wd)

                # Momentum update
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                # Nesterov
                g_effective = grad.add(buf, alpha=momentum)

                # SGD update on z
                z.add_(g_effective, alpha=-effective_lr)

                # Update x as cumulative average of z
                x.lerp_(z, 1.0 / t)

                # Restore p.data to y = (1-momentum)*z + momentum*x
                p.data.copy_(z).lerp_(x, momentum)

        self._mode = 'train'
        return loss

    @torch.no_grad()
    def eval(self):
        """Switch parameters from y to x (for evaluation)."""
        if self._mode == 'eval':
            return
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'x' not in state:
                    continue
                p.data.copy_(state['x'])
        self._mode = 'eval'

    @torch.no_grad()
    def train(self):
        """Switch parameters from x back to y (for training)."""
        if self._mode == 'train':
            return
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                state = self.state[p]
                if 'z' not in state:
                    continue
                p.data.copy_(state['z']).lerp_(state['x'], momentum)
        self._mode = 'train'
