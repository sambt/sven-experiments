import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import vmap, grad, functional_call
from sv3.utils.perf_tracking import get_gpu_memory_mb
import inspect
    
class FunctionalModelJac:
    def __init__(self, model, loss_fn, device, param_fraction=1.0, mask_by_block=False, microbatch_size=1, compile=True):
        """
        model: nn.Module
        loss_fn: function taking (pred, *args) and returning a scalar loss
        param_fraction: fraction of parameters to compute Jacobian w.r.t.
        microbatch_size: if not None, aggregate loss over sub-batches to reduce size of Jacobian matrix
        """
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.param_names_counts_startIdx = [] # to be filled in tie_parameters_to_flat
        self.params = self.tie_parameters_to_flat(requires_grad=False) # disable grads for the model
        self.params.requires_grad_(True) # enable grad for functional calls
        self.loss_fn = loss_fn

        self.param_fraction = param_fraction # fraction of parameters to compute Jacobian w.r.t.
        self.mask_by_block = mask_by_block # if True, mask entire parameter blocks (i.e. layers) instead of individual params when param_fraction is set
        self.param_mask = None # will be randomized each training step if param_fraction is not None
        self.microbatch_size = microbatch_size
        self.n_params = self.params.shape[0]
        self.param_shapes = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.num_loss_args = len(inspect.signature(loss_fn).parameters) - 1  # subtract 'pred'
        self.compiled_batch_gradient = self.get_compiled_batch_gradient()
        #if (compile and not self.param_fraction) else self.batch_gradient

        # variables to track gradients/losses for optimizer
        self.grads = torch.empty(0).to(device)
        self.losses = torch.empty(0).to(device)

    #@torch.compile
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
    
    def loss(self, params, x, *args) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        if self.param_mask is not None:
            input_params = self.params.clone()
            input_params[self.param_mask] = params
        else:
            input_params = params

        pred = self.func_call(input_params, x)
        loss = self.loss_fn(pred,*args)
        if self.microbatch_size > 1:
            # assuming loss has shape (B,)
            loss = loss.view(-1, self.microbatch_size).mean(dim=1)
        return loss, (loss, pred)

    def batch_gradient(self,batch) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        x, *args = batch
        params = self.params[self.param_mask] if self.param_mask is not None else self.params
        grads, (losses, preds) = torch.func.jacrev(self.loss, argnums=0, has_aux=True)(params, x, *args)
        return grads, losses, preds
    
    def get_compiled_batch_gradient(self):
        """Returns a compiled version of batch_gradient for faster execution."""
        return torch.compile(self.batch_gradient)

    def loss_and_grad(self, batch):
        """
        Perform one training step.
        
        Args:
            batch: Input batch (x, y, ...)
        """
        if self.param_fraction < 1.0:
            if not self.mask_by_block:
                self.param_mask = self.make_param_mask().to(self.params.device)
            else:
                self.param_mask = self.make_param_mask_byBlock(self.param_fraction).to(self.params.device)
        
        #grads, losses = self.batch_gradient(batch)
        grads, losses, preds = self.compiled_batch_gradient(batch)
        
        # Detach immediately to free graph
        grads = grads.detach()
        losses = losses.detach()

        self.grads = grads
        self.losses = losses

        return losses, preds
    
    def tie_parameters_to_flat(self, requires_grad=False):
        flat = parameters_to_vector(self.model.parameters()).detach()
        flat = flat.requires_grad_(requires_grad)

        # 2) rebind each parameter to a view into `flat`
        start = 0
        for name, p in self.model.named_parameters():
            n = p.numel()
            view = flat[start:start+n].view_as(p)
            self.param_names_counts_startIdx.append((name, n, start))
            start += n

            # walk to owning module and replace the parameter storage
            mod = self.model
            *prefix, leaf = name.split(".")
            for part in prefix:
                mod = getattr(mod, part)
            # assign the view as the new parameter (shares storage with flat)
            mod._parameters[leaf] = torch.nn.Parameter(view, requires_grad=requires_grad)

        return flat

    def make_param_mask(self) -> torch.Tensor:
        """Create a random parameter mask selecting a fraction of parameters."""
        n_active = int(self.param_fraction * self.n_params)
        param_mask = torch.zeros(self.n_params)
        param_mask[torch.randperm(self.n_params)[:n_active]] = 1
        return param_mask.to(torch.bool)
    
    def make_param_mask_byBlock(self, fraction: float) -> torch.Tensor:
        """Create a parameter mask that selects entire parameter blocks (layers) randomly until we hit the desired fraction."""
        n_blocks = len(self.param_names_counts_startIdx)
        random_order = torch.randperm(n_blocks)
        
        running_param_count = 0
        param_mask = torch.zeros(self.n_params)
        for i in random_order:
            name, nparam, start_idx = self.param_names_counts_startIdx[i]
            if running_param_count + nparam >= fraction * self.n_params:
                n_to_use = int(fraction * self.n_params) - running_param_count
                param_mask[start_idx:start_idx + n_to_use] = 1
                running_param_count += n_to_use
                break
            else:
                param_mask[start_idx:start_idx + nparam] = 1
                running_param_count += nparam
        
        return param_mask.to(torch.bool)
