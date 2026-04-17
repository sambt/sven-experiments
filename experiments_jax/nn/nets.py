"""Flax Linen models used by the JAX experiment framework.

The ``build_*`` helpers take plain Python / YAML-friendly args (lists, strings)
and return a Flax module suitable for instantiation via ``hydra.utils.instantiate``.

Notes on normalisation
----------------------
In the PyTorch version the ResNet uses a custom BatchNorm that plays nicely
with ``torch.func``. In JAX, BatchNorm's mutable ``batch_stats`` collide with
``jacrev`` in awkward ways. We use **GroupNorm** instead, which is stateless
and fully differentiable — no special care required.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Activation dispatch
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "tanh": jnp.tanh,
    "silu": nn.silu,
    "sigmoid": nn.sigmoid,
}


def _resolve_activation(name: str):
    if not isinstance(name, str):
        return name
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation {name!r}") from exc


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    input_dim: int
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        act = _resolve_activation(self.activation)
        for h in self.hidden_dims:
            x = act(nn.Dense(h)(x))
        return nn.Dense(self.output_dim)(x)


def build_mlp(input_dim, hidden_dims, output_dim, activation="gelu"):
    return MLP(
        input_dim=int(input_dim),
        hidden_dims=tuple(hidden_dims),
        output_dim=int(output_dim),
        activation=activation,
    )


# ---------------------------------------------------------------------------
# MultiMLP — ensemble of independent MLPs batched along an extra leading axis
# ---------------------------------------------------------------------------

class _MultiDense(nn.Module):
    num_models: int
    features: int

    @nn.compact
    def __call__(self, x):
        # x: (num_models, batch, in_features)
        in_features = x.shape[-1]
        W = self.param(
            "kernel",
            nn.initializers.he_uniform(),
            (self.num_models, in_features, self.features),
        )
        b = self.param(
            "bias",
            nn.initializers.zeros,
            (self.num_models, self.features),
        )
        # (M, B, in) @ (M, in, out) -> (M, B, out)
        return jnp.einsum("mbi,mio->mbo", x, W) + b[:, None, :]


class MultiMLP(nn.Module):
    num_models: int
    input_dim: int
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        # x: (batch, input_dim) -> (M, batch, input_dim)
        act = _resolve_activation(self.activation)
        x = jnp.broadcast_to(x[None], (self.num_models, *x.shape))
        for h in self.hidden_dims:
            x = act(_MultiDense(self.num_models, h)(x))
        return _MultiDense(self.num_models, self.output_dim)(x)


def build_multi_mlp(num_models, input_dim, hidden_dims, output_dim, activation="gelu"):
    return MultiMLP(
        num_models=int(num_models),
        input_dim=int(input_dim),
        hidden_dims=tuple(hidden_dims),
        output_dim=int(output_dim),
        activation=activation,
    )


# ---------------------------------------------------------------------------
# SmallCNN — CIFAR-10 baseline
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """CIFAR-sized CNN. Uses GroupNorm rather than BatchNorm (see module docstring)."""

    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        # x: (B, 32, 32, 3) NHWC
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        x = nn.Conv(64, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        x = nn.Conv(64, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


def build_small_cnn(num_classes=10):
    return SmallCNN(num_classes=int(num_classes))


# ---------------------------------------------------------------------------
# SmallResNet — CIFAR-style basic block ResNet (GroupNorm variant)
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        residual = x

        y = nn.Conv(self.out_channels, (3, 3), strides=self.stride, padding="SAME", use_bias=False)(x)
        y = nn.GroupNorm(num_groups=min(8, self.out_channels))(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, (3, 3), strides=1, padding="SAME", use_bias=False)(y)
        y = nn.GroupNorm(num_groups=min(8, self.out_channels))(y)

        if self.stride != 1 or in_channels != self.out_channels:
            residual = nn.Conv(self.out_channels, (1, 1), strides=self.stride, use_bias=False)(x)
            residual = nn.GroupNorm(num_groups=min(8, self.out_channels))(residual)

        return nn.relu(y + residual)


class SmallResNet(nn.Module):
    num_classes: int = 10
    width: int = 16
    num_blocks: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) NHWC
        x = nn.Conv(self.width, (3, 3), strides=1, padding="SAME", use_bias=False)(x)
        x = nn.GroupNorm(num_groups=min(8, self.width))(x)
        x = nn.relu(x)

        for i in range(self.num_blocks):
            x = _BasicBlock(self.width, stride=1)(x)
        for i in range(self.num_blocks):
            x = _BasicBlock(self.width * 2, stride=2 if i == 0 else 1)(x)
        for i in range(self.num_blocks):
            x = _BasicBlock(self.width * 4, stride=2 if i == 0 else 1)(x)

        x = jnp.mean(x, axis=(1, 2))
        return nn.Dense(self.num_classes)(x)


def build_small_resnet(num_classes=10, width=16, num_blocks=2):
    return SmallResNet(
        num_classes=int(num_classes),
        width=int(width),
        num_blocks=int(num_blocks),
    )
