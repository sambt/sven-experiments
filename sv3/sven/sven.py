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
from sv3.nn.sven_wrapper import SvenWrapper

class Sven:
    def __init__(self, model:SvenWrapper, lr, k, rtol, track_svd_info=False, svd_mode='torch',
                 power_iterations=1,use_rmsprop=False,alpha_rmsprop=0.99,eps_rmsprop=1e-8,mu_rmsprop=0,rmsprop_post=False,variable_k=False):
        self.model = model 
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self.power_iterations = power_iterations # number of power iterations for randomized SVD (if using)
        self.svd_info = {
            "svs":[],
            "num_nonzero_svs":[],
            "k_used":[],
            "variable_k_substep_losses":[]
        }
        self.track_svd_info = track_svd_info
        self.svd_mode = svd_mode
        self.variable_k = variable_k
        self.use_rmsprop = use_rmsprop
        self.rmsprop_post = rmsprop_post
        if use_rmsprop:
            self.alpha_rmsprop = alpha_rmsprop
            self.eps_rmsprop = eps_rmsprop
            self.mu_rmsprop = mu_rmsprop
            self.v = None  # Running average of squared updates
            if self.mu_rmsprop > 0:
                self.b = None

    @staticmethod
    def _compute_delta(U_T, S_inv, VhT, losses):
        """Compiled helper for computing parameter update. Fused matmul operations."""
        delta_p = U_T @ losses  # (k x B) @ (B,) -> (k,)
        delta_p = S_inv * delta_p  # element-wise multiply instead of diag
        delta_p = VhT @ delta_p  # (P x k) @ (k,) -> (P,)
        return delta_p
    
    @staticmethod
    def _compute_delta_k(k, U_T, S_inv, VhT, losses):
        delta_p = U_T[k:k+1, :] @ losses  # (1 x B) @ (B,) -> (1,)
        delta_p = S_inv[k] * delta_p  # multiply in 1/s_i
        delta_p = VhT[:, k:k+1] @ delta_p  # (P x 1) @ (1,) -> (P,)
        return delta_p.squeeze()

    
    def _get_pinv(self, jacobian):
        # Get SVD components (memory efficient - returns views/slices)
        VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, mode=self.svd_mode, power_iter=self.power_iterations)
        return VhT, S_inv, U_T
    
    def _apply_update(self, update):
        # apply direction and learning rate
        update = -self.lr * update
        # apply the update
        if self.model.param_mask is not None:
            self.model.params[self.model.param_mask] += update
        else:
            self.model.params += update
    
    @torch.no_grad()
    def _update_params(self, U_T, S_inv, VhT, losses):
        # Use compiled version for matmul operations (fused kernel)
        update = self._compute_delta(U_T, S_inv, VhT, losses)

        if self.use_rmsprop and self.rmsprop_post:
            # update rms estimate
            if self.v is None:
                self.v = torch.zeros_like(update)
            self.v.mul_(self.alpha_rmsprop).addcmul_(update, update, value=1 - self.alpha_rmsprop)

            # scale update by rms
            update = update / (torch.sqrt(self.v) + self.eps_rmsprop)
            
            # if using momentum, add momentum term
            if self.mu_rmsprop > 0:
                if self.b is None:
                    self.b = torch.zeros_like(update)
                self.b.mul_(self.mu_rmsprop).add_(update)
                update = self.b

        # apply the update
        self._apply_update(update)

    @torch.no_grad()
    def _update_params_variable_k(self, batch, U_T, S_inv, VhT, losses):
        if batch is None:
            raise ValueError("Batch must be provided when using variable_k=True")
        
        original_loss = losses.mean()
        kmax = len(S_inv)
        kcurr = 0
        x, *args = batch
        
        substep_losses = [original_loss]
        while kcurr < kmax:
            update = self._compute_delta_k(kcurr, U_T, S_inv, VhT, losses)
            self._apply_update(update)

            # evaluate new train loss after update
            new_loss = self.model.evaluate_and_loss(x, *args).mean()

            if new_loss > original_loss:
                # if loss increased, revert update and stop
                self._apply_update(-update)
                break
                
            substep_losses.append(new_loss)
            kcurr += 1
        return kcurr, substep_losses

    @torch.no_grad()
    def step(self, batch=None):
        """
        Compute parameter update using Moore-Penrose pseudoinverse
        """
        jacobian = self.model.grads
        losses = self.model.losses

        # If doing rmsprop and applying it to the *gradients*, rather than the *updates*, then do it before the pinv
        if self.use_rmsprop and not self.rmsprop_post:
            mean_grad = torch.mean(jacobian, dim=0)
            if self.v is None:
                self.v = torch.zeros_like(mean_grad)
            self.v.mul_(self.alpha_rmsprop).addcmul_(mean_grad, mean_grad, value=1 - self.alpha_rmsprop)
            jacobian = jacobian / (torch.sqrt(self.v) + self.eps_rmsprop)

        VhT, S_inv, U_T = self._get_pinv(jacobian)
        
        # clean up
        del jacobian
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # update parameters
        if self.variable_k:
            k_used, substep_losses = self._update_params_variable_k(batch, U_T, S_inv, VhT, losses)
        else:
            self._update_params(U_T, S_inv, VhT, losses)
        
        # log svd info
        if self.track_svd_info:
            self.svd_info["svs"].append(1.0 / S_inv[S_inv > 0].cpu().numpy())
            self.svd_info["num_nonzero_svs"].append(torch.count_nonzero(S_inv).item())
            if self.variable_k:
                self.svd_info["k_used"].append(k_used)
                self.svd_info["variable_k_substep_losses"].append(substep_losses)
        # Aggressive memory cleanup
        del VhT, S_inv, U_T
        del self.model.losses, self.model.grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
