import torch.nn as nn

from experiments import register_model
from experiments.nn import MLP


@register_model("mlp")
def build_mlp(meta: dict, device="cpu", hidden_dims=None, activation="gelu", **kwargs):
    """
    Build an MLP matching dataset meta info.
    """
    hidden_dims = hidden_dims or [16, 16, 16]
    act_cls = getattr(nn, activation.upper(), None) if isinstance(activation, str) else activation
    if act_cls is None:
        act_cls = nn.GELU

    input_dim = meta.get("input_dim")
    output_dim = meta.get("output_dim")
    model = MLP(input_dim, hidden_dims, output_dim, activation=act_cls)
    return model.to(device)
