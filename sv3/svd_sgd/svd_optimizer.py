import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd.functional import jacobian
from torch.func import vmap, grad, functional_call
import inspect
from collections import defaultdict

from sv3.utils.perf_tracking import get_gpu_memory_mb
from .pinv import pinv
from sv3.nn.torch_func import FunctionalModelJac

class SVDOptimizer:
    def __init__(self, model:FunctionalModelJac, lr, k, rtol, track_svd_info=False):
        self.model = model 
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self._compute_delta_compiled = torch.compile(self._compute_delta, mode='max-autotune')
        self.svd_info = {
            "svs":[],
            "num_nonzero_svs":[]
        }
        self.track_svd_info = track_svd_info

    @staticmethod
    def _compute_delta(U_T, S_inv, VhT, losses, lr):
        """Compiled helper for computing parameter update. Fused matmul operations."""
        delta_p = U_T @ losses  # (k x B) @ (B,) -> (k,)
        delta_p = S_inv * delta_p  # element-wise multiply instead of diag
        delta_p = VhT @ delta_p  # (P x k) @ (k,) -> (P,)
        return -lr * delta_p

    @torch.no_grad()
    def step(self):
        """
        Compute parameter update using SVD pseudo-inverse.
        Memory-optimized: never materializes full pseudo-inverse matrix.
        """
        jacobian = self.model.grads
        losses = self.model.losses
        # Get SVD components (memory efficient - returns views/slices)
        VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, randomized=True)
        # Vh.T is (P x k), S_inv is (k,), U.T is (k x B)
        
        # clean up
        del jacobian
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use compiled version for matmul operations (fused kernel)
        update = self._compute_delta_compiled(U_T, S_inv, VhT, losses, self.lr)

        # apply the update
        if self.model.param_mask is not None:
            self.model.params[self.model.param_mask] += update
        else:
            self.model.params += update
        
        # log svd info
        if self.track_svd_info:
            self.svd_info["svs"].append(1.0 / S_inv[S_inv > 0].cpu().numpy())
            self.svd_info["num_nonzero_svs"].append(torch.count_nonzero(S_inv).item())
        
        # Aggressive memory cleanup
        del VhT, S_inv, U_T
        del self.model.losses, self.model.grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SVDOptimizerRMSprop:
    def __init__(self, lr, k, rtol, alpha=0.99, eps=1e-8):
        """
        SVD optimizer with RMSprop-style adaptive learning rates.
        
        Args:
            lr: Base learning rate
            k: Number of singular values to keep in truncated SVD
            rtol: Relative tolerance for singular value truncation
            alpha: Smoothing constant for squared gradient averaging (default: 0.99)
            eps: Small constant for numerical stability (default: 1e-8)
        """
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self.alpha = alpha
        self.eps = eps
        self.v = None  # Running average of squared updates
        self._compute_delta_compiled = torch.compile(self._compute_delta, mode='max-autotune')

    @staticmethod
    def _compute_delta(U_T, S_inv, VhT, losses, lr):
        """Compiled helper for computing parameter update. Fused matmul operations."""
        delta_p = U_T @ losses  # (k x B) @ (B,) -> (k,)
        delta_p = S_inv * delta_p  # element-wise multiply instead of diag
        delta_p = VhT @ delta_p  # (P x k) @ (k,) -> (P,)
        return -lr * delta_p

    @torch.no_grad()
    def compute_update(self, jacobian, losses):
        """
        Compute parameter update with RMSprop preconditioning.
        Scales Jacobian rows by adaptive learning rates before SVD.
        """
        # Compute mean gradient across batch
        mean_grad = torch.mean(jacobian, dim=0)
        
        # Initialize v on first call
        if self.v is None:
            self.v = torch.zeros_like(mean_grad)
        
        # Update running average of squared gradients (in-place)
        self.v.mul_(self.alpha).addcmul_(mean_grad, mean_grad, value=1 - self.alpha)

        # Scale jacobian by RMSprop preconditioner (broadcast over batch dimension)
        jacobian = jacobian / (torch.sqrt(self.v) + self.eps)

        # Compute SVD of preconditioned Jacobian
        VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, randomized=True)
        # Vh.T is (P x k), S_inv is (k,), U.T is (k x B)
        
        # Use compiled version for matmul operations (fused kernel)
        update = self._compute_delta_compiled(U_T, S_inv, VhT, losses, self.lr)

        extras = {
            "singular_values": 1.0 / S_inv[S_inv > 0].cpu().numpy(),
            "num_svs": torch.count_nonzero(S_inv).item()
        }
        
        # Aggressive memory cleanup
        del jacobian, VhT, S_inv, U_T, mean_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return update, extras