import torch
from torch.optim import Optimizer

class PolyakSGD(Optimizer):
    def __init__(self, params, f_star=0.0, max_lr=1.0, eps=1e-8):
        """
        params:  model parameters
        f_star:  estimated minimum loss (0.0 for interpolating models)
        max_lr:  clamp to prevent huge steps
        eps:     numerical stability
        """
        defaults = dict(f_star=f_star, max_lr=max_lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        # Closure must return the loss
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            f_star = group['f_star']
            max_lr = group['max_lr']
            eps     = group['eps']

            # Compute squared gradient norm across all params in group
            grad_sq_norm = sum(
                p.grad.detach().norm() ** 2
                for p in group['params']
                if p.grad is not None
            )

            # Polyak step size
            numerator = loss.item() - f_star
            lr = numerator / (grad_sq_norm.item() + eps)
            lr = min(lr, max_lr)  # clamp

            # Apply update
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad.detach(), alpha=-lr)

        return loss