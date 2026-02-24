from experiments import register_optimizer
from sv3.nn import SvenWrapper
from sv3.sven import Sven


def mse_loss(pred, target):
    loss = (pred - target) ** 2
    return loss.sum(dim=-1)


@register_optimizer("svd")
def build_svd_optimizer(
    model,
    lr: float = 0.1,
    k: int | None = None,
    rtol: float = 1e-3,
    loss_fn=None,
    param_fraction=None,
    sub_batch_size=None,
    track_svd_info: bool = True,
    **_,
):
    """
    Wrap a base model with SvenWrapper and return the Sven optimizer.
    """
    loss_fn = loss_fn or mse_loss
    fn_model = SvenWrapper(
        model, loss_fn=loss_fn, param_fraction=param_fraction, sub_batch_size=sub_batch_size
    )
    if k is None:
        raise ValueError("k must be specified for SVD optimizer (e.g., set k equal to batch_size // 2)")
    optimizer = Sven(fn_model, lr=lr, k=k, rtol=rtol, track_svd_info=track_svd_info)
    return fn_model, optimizer, {"loss_fn": loss_fn}
