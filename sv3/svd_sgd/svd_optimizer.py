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
    def __init__(self, model:FunctionalModelJac, lr, k, rtol, track_svd_info=False, svd_mode='randomized',
                 power_iterations=1,use_rmsprop=False,alpha_rmsprop=0.99,eps_rmsprop=1e-8,compile=True):
        self.model = model 
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self.power_iterations = power_iterations # number of power iterations for randomized SVD
        self._compute_delta_compiled = torch.compile(self._compute_delta) if compile else self._compute_delta
        self.svd_info = {
            "svs":[],
            "num_nonzero_svs":[]
        }
        self.track_svd_info = track_svd_info
        self.svd_mode = svd_mode
        self.use_rmsprop = use_rmsprop
        if use_rmsprop:
            self.alpha_rmsprop = alpha_rmsprop
            self.eps_rmsprop = eps_rmsprop
            self.v = None  # Running average of squared updates

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

        # If doing rmsprop
        if self.use_rmsprop:
            mean_grad = torch.mean(jacobian, dim=0)
            if self.v is None:
                self.v = torch.zeros_like(mean_grad)
            self.v.mul_(self.alpha_rmsprop).addcmul_(mean_grad, mean_grad, value=1 - self.alpha_rmsprop)
            jacobian = jacobian / (torch.sqrt(self.v) + self.eps_rmsprop)

        # Get SVD components (memory efficient - returns views/slices)
        if self.svd_mode == 'randomized':
            VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, full=False, randomized=True, scipy=False, power_iter=self.power_iterations)
        elif self.svd_mode == 'scipy':
            VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, full=False, randomized=False, scipy=True)
        elif self.svd_mode == 'full':
            VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, full=True, randomized=False, scipy=False)
        elif self.svd_mode == 'lobpcg':
            VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, full=False, randomized=False, scipy=False, power_iter=self.power_iterations)
        else:
            raise ValueError(f"Unknown svd_mode: {self.svd_mode}")
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
