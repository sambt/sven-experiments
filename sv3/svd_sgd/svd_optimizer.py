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

class FunctionalModel:
    def __init__(self, model, loss_lambda):
        """
        model: nn.Module
        loss_lambda: function taking (pred, *args) and returning a scalar loss
        """
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.loss_lambda = loss_lambda
        
        self.params = parameters_to_vector(model.parameters()).detach()
        self.param_shapes = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.num_loss_args = len(inspect.signature(loss_lambda).parameters) - 1  # subtract 'pred'
        self.create_batch_gradient()

    def func_call(self, params, x):
        param_dict = {}
        start_idx = 0
        for name,shape,size in self.param_shapes:
            param_dict[name] = params[start_idx:start_idx+size].view(shape)
            start_idx += size
        # Fetch fresh buffers on every call (includes updated BatchNorm stats)
        for name, buffer in self.model.named_buffers():
            param_dict[name] = buffer
            
        return functional_call(self.model, param_dict, x)

    @torch.compile
    @torch.no_grad()
    def evaluate(self, x):
        return self.func_call(self.params, x)
    
    def single_loss(self, params, x, *args):
        pred = self.func_call(params, x)
        loss = self.loss_lambda(pred,*args)
        return loss, loss

    def create_batch_gradient(self):
        grad_fn = grad(self.single_loss,argnums=0,has_aux=True)
        self.batched_grad_fn = torch.compile(
            vmap(grad_fn, in_dims=(None, 0, *(0 for _ in range(self.num_loss_args))), out_dims=(0,0))
        )

    def batch_gradient(self,params,batch):
        x, *args = batch
        grads, losses = self.batched_grad_fn(params, x, *args)
        return grads, losses

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
            self.svd_info["num_nonzero_svd"].append(torch.count_nonzero(S_inv).item())
        
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