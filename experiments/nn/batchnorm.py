from torch.nn.modules.batchnorm import _NormBase
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _in_functorch_transform() -> bool:
    """Return True when called from within a torch.func transform (jacrev, grad, vmap…).

    During such transforms, in-place mutations to captured tensors (e.g. BatchNorm's
    running_mean / running_var buffers) raise a hard error.  We use this flag to skip
    those updates and keep the forward pass purely functional.
    """
    try:
        _functorch = getattr(torch._C, "_functorch", None)
        if _functorch is None:
            return False
        peek = getattr(_functorch, "peek_interpreter_stack", None)
        return peek is not None and peek() is not None
    except Exception:
        return False


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                # Out-of-place increment: avoids aten::add_.Tensor on a captured buffer.
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # When computing with batch statistics (bn_training=True), F.batch_norm would
        # normally update running_mean / running_var in-place.  Inside a torch.func
        # transform (e.g. jacrev used by Sven) that in-place mutation on a captured
        # buffer tensor raises:
        #   "in-place operation … would mutate a captured Tensor"
        # Fix: pass None so F.batch_norm uses batch stats without writing back to the
        # running-stat buffers.  Outside of any transform, behaviour is unchanged.
        if bn_training and _in_functorch_transform():
            running_mean = None
            running_var = None
        else:
            running_mean = (
                self.running_mean if not self.training or self.track_running_stats else None
            )
            running_var = (
                self.running_var if not self.training or self.track_running_stats else None
            )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


def replace_batchnorm(model: nn.Module) -> nn.Module:
    """Recursively replace all standard nn.BatchNorm2d modules in *model* with the
    torch.func-compatible BatchNorm2d defined in this module.

    This is required for models like torchvision ResNet18 that use PyTorch's built-in
    BatchNorm2d, whose forward pass does in-place updates to running_mean / running_var
    — mutations that are illegal inside torch.func transforms (jacrev, grad, vmap, …)
    used by the Sven optimiser.

    The replacement copies all learned parameters (weight, bias) and running statistics
    so the model's behaviour is unchanged outside of torch.func contexts.

    Returns *model* in-place (also returned for convenience).
    """
    for name, child in list(model.named_children()):
        if type(child) is nn.BatchNorm2d:
            # Determine device from whichever tensor is available
            ref = next(child.parameters(), None)
            if ref is None:
                ref = child.running_mean
            device = ref.device if ref is not None else torch.device("cpu")
            new_bn = BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            ).to(device)
            if child.affine:
                new_bn.weight = child.weight
                new_bn.bias = child.bias
            if child.track_running_stats:
                # running_mean/var/num_batches_tracked are always Tensors when
                # track_running_stats=True; the Optional typing is a PyTorch
                # annotation artefact.
                src_rm = child.running_mean
                src_rv = child.running_var
                src_nbt = child.num_batches_tracked
                dst_rm = new_bn.running_mean
                dst_rv = new_bn.running_var
                dst_nbt = new_bn.num_batches_tracked
                if (src_rm is not None and src_rv is not None and src_nbt is not None
                        and dst_rm is not None and dst_rv is not None and dst_nbt is not None):
                    dst_rm.copy_(src_rm)
                    dst_rv.copy_(src_rv)
                    dst_nbt.copy_(src_nbt)
            setattr(model, name, new_bn)
        else:
            replace_batchnorm(child)
    return model
