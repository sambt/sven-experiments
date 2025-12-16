import torch
import torch.nn as nn

from experiments import register_optimizer


@register_optimizer("adam")
def build_adam_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.0, loss_fn=None, **_):
    """
    Standard Adam optimizer for baseline comparisons.
    Returns the model unchanged plus the optimizer and an optional loss function (default MSE).
    """
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer, {"loss_fn": loss_fn}
