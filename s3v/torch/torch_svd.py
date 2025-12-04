import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd.functional import jacobian
from torch.func import vmap, grad, functional_call
import inspect

# Memory tracking utilities
def get_gpu_memory_mb():
    """Returns current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def reset_peak_memory():
    """Reset peak memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation()]
        for i in range(1,len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def make_functional_model(model):
    param_shapes = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
    
    def functional_model(params,x):
        param_dict = {}
        start_idx = 0
        for name,shape,size in param_shapes:
            param_dict[name] = params[start_idx:start_idx+size].view(shape)
            start_idx += size
        return torch.func.functional_call(model, param_dict, x)
    
    return functional_model

@torch.no_grad()
def pinv(M: torch.Tensor, k: int = 2, tol: float = 1e-10, rtol:float = 1e-3, full=False, randomized=False) -> torch.Tensor:
    """
    Compute pseudo-inverse via truncated SVD. Memory-optimized version.
    Returns VhT, S_inv, U_T to avoid storing full pseudo-inverse matrix.
    """
    with torch.no_grad():
        M = M.detach()
        if full:
            U, S, Vh = truncated_svd_full(M,k=k,rtol=rtol)
        elif randomized:
            U, S, Vh = randomized_SVD(M, k=k, rtol=rtol)
        else:
            U, S, Vh = truncated_svd(M, k=k, rtol=rtol)
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    return Vh.T.detach(), S_inv.detach(), U.T.detach()

@torch.no_grad()
def truncated_svd(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        A = A.detach()
        m, n = A.shape
        if n <= m:
            C = A.T @ A       # (n x n), symmetric PSD
            # Use LOBPCG or plain eigen; eigsh-style not yet standard in torch
            # torch.lobpcg expects symmetric; give random init
            #X = torch.randn(n, k, device=A.device, dtype=A.dtype)
            evals, evecs = torch.lobpcg(C, k=k, largest=True)  # top-k
            s = evals.clamp_min(0).sqrt()
            V = evecs
            # Left sing vecs: U = A V / s
            U = (A @ V) / s.clamp_min(torch.finfo(A.dtype).eps)
            Vh = V.T
            s = torch.where(s > rtol * s[0], s, torch.zeros_like(s))
            return U.detach(), s.detach(), Vh.detach()
            #Av = A @ evecs
            #u,s,vh = torch.linalg.svd(Av, full_matrices=False)
        else:
            C = A @ A.T       # (m x m)
            #X = torch.randn(m, k, device=A.device, dtype=A.dtype)
            evals, evecs = torch.lobpcg(C, k=k, largest=True)
            s = evals.clamp_min(0).sqrt()
            U = evecs
            Vh = ((U.T @ A) / s.clamp_min(torch.finfo(A.dtype).eps).reshape(-1,1)).conj()
            s = torch.where(s > rtol * s[0], s, torch.zeros_like(s))
            return U.detach(), s.detach(), Vh.detach()

@torch.compile
@torch.no_grad()     
def truncated_svd_full(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        A = A.detach()
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
        U = U[:,:k]
        S = S[:k]
        Vh = Vh[:k,:]
        S = torch.where(S > rtol * S[0], S, torch.zeros_like(S))
        return U.detach(), S.detach(), Vh.detach()

@torch.compile
@torch.no_grad()
def randomized_SVD(A, k, p=5, q=1, rtol:float=1e-3):
    """
    Randomized SVD algorithm. Memory-efficient for rank-k approximations.
    p: oversampling parameter (default 5)
    q: power iteration parameter (default 1)
    """
    m,n = A.shape
    r = min(k+p, min(m, n))  # Don't exceed matrix dimensions
    device = A.device
    dtype = A.dtype

    # Random projection
    Omega = torch.randn(n, r, device=device, dtype=dtype)
    Y = A @ Omega
    
    # Power iterations for better accuracy
    for _ in range(q):
        Y = A @ (A.T @ Y)
    
    # QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # Project and compute SVD of smaller matrix
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    # Truncate to rank k
    U = U[:,:k].contiguous()
    S = S[:k].contiguous()
    Vh = Vh[:k,:].contiguous()
    
    # Threshold small singular values
    S = torch.where(S > rtol * S[0], S, torch.zeros_like(S))
    
    return U, S, Vh

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
    
class FunctionalModelJac:
    def __init__(self, model, loss_lambda, optimizer, param_fraction=None, sub_batch_size=None):
        """
        model: nn.Module
        loss_lambda: function taking (pred, *args) and returning a scalar loss
        param_fraction: fraction of parameters to compute Jacobian w.r.t.
        sub_batch_size: if not None, aggregate loss over sub-batches to reduce size of Jacobian matrix
        """
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.loss_lambda = loss_lambda
        self.optimizer = optimizer

        self.param_fraction = param_fraction # fraction of parameters to compute Jacobian w.r.t.
        self.param_mask = None # will be randomized each training step if param_fraction is not None
        self.sub_batch_size = sub_batch_size
        self.params = parameters_to_vector(model.parameters()).detach()
        self.n_params = self.params.shape[0]
        self.param_shapes = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.num_loss_args = len(inspect.signature(loss_lambda).parameters) - 1  # subtract 'pred'
        self.compiled_batch_gradient = self.get_compiled_batch_gradient()

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

    @torch.no_grad()
    def evaluate(self, x):
        return self.func_call(self.params, x)
    
    def loss(self, params, x, *args):
        if self.param_mask is not None:
            input_params = self.params.clone()
            input_params[self.param_mask] = params
        else:
            input_params = params

        pred = self.func_call(input_params, x)
        loss = self.loss_lambda(pred,*args)
        if self.sub_batch_size is not None:
            # assuming loss has shape (B,)
            loss = loss.view(-1, self.sub_batch_size).mean(dim=1)
        return loss, loss

    def batch_gradient(self,batch):
        x, *args = batch
        params = self.params[self.param_mask] if self.param_mask is not None else self.params
        grads, losses = torch.func.jacrev(self.loss, argnums=0, has_aux=True)(params, x, *args)
        return grads, losses
    
    def get_compiled_batch_gradient(self):
        """Returns a compiled version of batch_gradient for faster execution."""
        return torch.compile(self.batch_gradient)

    @torch.no_grad()
    def train_step(self, batch, track_memory=False):
        """
        Perform one training step.
        
        Args:
            batch: Input batch (x, y, ...)
            track_memory: If True, return memory stats in extras dict
        """
        mem_stats = {}
        if track_memory:
            mem_stats['start'] = get_gpu_memory_mb()
        
        if self.param_fraction is not None:
            self.param_mask = (torch.rand(self.n_params) < self.param_fraction).to(self.params.device)
        
        if track_memory:
            mem_stats['after_mask'] = get_gpu_memory_mb()
        
        # Enable grad temporarily for jacrev
        with torch.enable_grad():
            #grads, losses = self.batch_gradient(batch)
            grads, losses = self.compiled_batch_gradient(batch)
        
        if track_memory:
            mem_stats['after_jacobian'] = get_gpu_memory_mb()
        
        # Detach immediately to free graph
        grads = grads.detach()
        losses = losses.detach()
        
        update, extras = self.optimizer.compute_update(grads, losses)
        
        if track_memory:
            mem_stats['after_svd'] = get_gpu_memory_mb()
        
        if self.param_mask is not None:
            self.params[self.param_mask] += update
        else:
            self.params += update
        
        if track_memory:
            mem_stats['after_update'] = get_gpu_memory_mb()
        
        # Clear cache to free memory
        del grads, update
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if track_memory:
            mem_stats['after_cleanup'] = get_gpu_memory_mb()
            extras['memory'] = mem_stats
            
        return losses, extras

class SVDOptimizer:
    def __init__(self, lr, k, rtol):
        self.lr = lr
        self.k = k
        self.rtol = rtol
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
        Compute parameter update using SVD pseudo-inverse.
        Memory-optimized: never materializes full pseudo-inverse matrix.
        """
        # Get SVD components (memory efficient - returns views/slices)
        VhT, S_inv, U_T = pinv(jacobian, k=self.k, rtol=self.rtol, randomized=True)
        # Vh.T is (P x k), S_inv is (k,), U.T is (k x B)
        del jacobian
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use compiled version for matmul operations (fused kernel)
        update = self._compute_delta_compiled(U_T, S_inv, VhT, losses, self.lr)

        extras = {
            "singular_values": 1.0 / S_inv[S_inv > 0].cpu().numpy(),
            "num_svs": torch.count_nonzero(S_inv).item()
        }
        
        # Aggressive memory cleanup
        del VhT, S_inv, U_T
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return update, extras

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

"""
Thomas's RMSprop implementation
@jax.jit
def update(grads_batched, loss_batched, state, params=None):
    step = state.step + 1

    if with_rms:
        # Compute mean gradient across batch
        mean_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_batched)

        # Update biased second raw moment estimate
        v = jax.tree_map(lambda v, g: beta * v + (1 - beta) * (g ** 2), state.v, mean_grads)

        # Compute bias-corrected second raw moment estimate (for preconditioning)
        v_hat = jax.tree_map(lambda v: v / (1 - beta ** step), v)

        # Scale each gradient in the batch by the Adam preconditioner
        # This applies element-wise adaptive learning rate scaling to each sample's gradient
        grads_batched = jax.tree_map(
            lambda g, v_h: g / (jnp.sqrt(v_h) + eps),
            grads_batched,
            v_hat
        )

    mat = __get_matrix(grads_batched, loss_batched.shape[0])

    p_inv = pseudo_inverse_jax(mat, k=num_svd, rtol=rtol, key=state.svd_key)

    delta_params = p_inv @ loss_batched

    flat_updates =  - learning_rate * delta_params.flatten()
    unravel_fn = jax.flatten_util.ravel_pytree(params)[1]
    updates = unravel_fn(flat_updates)
    if with_rms:
        new_state = svd_sgd_state(
            step=step,
            svd_key=jax.random.split(state.svd_key)[0],
            v=v,
            beta=state.beta,
            eps=state.eps,
        )
    else:
        new_state = svd_sgd_state(
            step=step,
            svd_key=jax.random.split(state.svd_key)[0],
        )

    return updates, new_state
    """