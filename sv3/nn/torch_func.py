import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import vmap, grad, functional_call
from sv3.utils.perf_tracking import get_gpu_memory_mb
import inspect

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
    def __init__(self, model, loss_fn, param_fraction=None, sub_batch_size=None):
        """
        model: nn.Module
        loss_fn: function taking (pred, *args) and returning a scalar loss
        param_fraction: fraction of parameters to compute Jacobian w.r.t.
        sub_batch_size: if not None, aggregate loss over sub-batches to reduce size of Jacobian matrix
        """
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.loss_fn = loss_fn

        self.param_fraction = param_fraction # fraction of parameters to compute Jacobian w.r.t.
        self.param_mask = None # will be randomized each training step if param_fraction is not None
        self.sub_batch_size = sub_batch_size
        self.params = parameters_to_vector(model.parameters()).detach()
        self.n_params = self.params.shape[0]
        self.param_shapes = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.num_loss_args = len(inspect.signature(loss_fn).parameters) - 1  # subtract 'pred'
        self.compiled_batch_gradient = self.get_compiled_batch_gradient()

        # variables to track gradients/losses for optimizer
        self.grads = torch.empty(0)
        self.losses = torch.empty(0)

    @torch.compile
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
    
    def loss(self, params, x, *args) -> tuple[torch.Tensor,torch.Tensor]:
        if self.param_mask is not None:
            input_params = self.params.clone()
            input_params[self.param_mask] = params
        else:
            input_params = params

        pred = self.func_call(input_params, x)
        loss = self.loss_fn(pred,*args)
        if self.sub_batch_size is not None:
            # assuming loss has shape (B,)
            loss = loss.view(-1, self.sub_batch_size).mean(dim=1)
        return loss, loss

    def batch_gradient(self,batch) -> tuple[torch.Tensor,torch.Tensor]:
        x, *args = batch
        params = self.params[self.param_mask] if self.param_mask is not None else self.params
        grads, losses = torch.func.jacrev(self.loss, argnums=0, has_aux=True)(params, x, *args)
        return grads, losses
    
    def get_compiled_batch_gradient(self):
        """Returns a compiled version of batch_gradient for faster execution."""
        return torch.compile(self.batch_gradient)

    def loss_and_grad(self, batch):
        """
        Perform one training step.
        
        Args:
            batch: Input batch (x, y, ...)
            track_memory: If True, return memory stats in extras dict
        """
        if self.param_fraction is not None:
            self.param_mask = (torch.rand(self.n_params) < self.param_fraction).to(self.params.device)
        
        #grads, losses = self.batch_gradient(batch)
        grads, losses = self.compiled_batch_gradient(batch)
        
        # Detach immediately to free graph
        grads = grads.detach()
        losses = losses.detach()

        self.grads = grads
        self.losses = losses

        return losses