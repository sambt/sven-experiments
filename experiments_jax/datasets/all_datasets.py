"""JAX-native datasets for the experiment framework.

Each dataset class holds ``train`` and ``val`` (and optionally ``test``)
attributes, each of which is a tuple ``(x, y)`` of ``jnp.ndarray``. Batching
is performed by :func:`experiments_jax.experiment_code.experiment_utils.iter_batches`
rather than a DataLoader — the data is small enough to live on-device
throughout training.
"""

from __future__ import annotations

import os
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


class _JaxDataset:
    """Bag of ``(x, y)`` arrays for train / val / (optional) test splits."""

    train: tuple[jnp.ndarray, jnp.ndarray]
    val: tuple[jnp.ndarray, jnp.ndarray]
    test: tuple[jnp.ndarray, jnp.ndarray] | None = None


class Toy1DRegressionDataset(_JaxDataset):
    """1D regression task: ``y = exp(-10 x^2) sin(2 x)`` on ``[-1, 1]``."""

    def __init__(
        self,
        n_train: int = 10_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed

        rng = np.random.default_rng(seed)

        def sample(n: int):
            x = (2 * rng.uniform(size=(n, 1)) - 1).astype(np.float32)
            y = (np.exp(-10 * x ** 2) * np.sin(2 * x)).astype(np.float32)
            return x, y

        xtrain, ytrain = sample(n_train)
        xval, yval = sample(n_val)
        xtest, ytest = sample(n_test)

        mean, std = ytrain.mean(), ytrain.std()
        ytrain = (ytrain - mean) / std
        yval = (yval - mean) / std
        ytest = (ytest - mean) / std

        self.train = (jnp.asarray(xtrain), jnp.asarray(ytrain))
        self.val = (jnp.asarray(xval), jnp.asarray(yval))
        self.test = (jnp.asarray(xtest), jnp.asarray(ytest))


class MNISTDataset(_JaxDataset):
    """Flat MNIST. Labels are integer class ids. Images are standardised with
    the canonical ``(0.1307, 0.3081)`` normalisation used elsewhere in this
    repo, then flattened to ``(784,)``.
    """

    def __init__(
        self,
        ROOT: str = "/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/mnist/",
        digits: list[int] | None = None,
    ) -> None:
        from torchvision.datasets import MNIST  # lazy — torchvision only to grab the raw bytes
        from torchvision import transforms
        import torch

        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train = MNIST(root=ROOT, train=True, download=True, transform=transform)
        val = MNIST(root=ROOT, train=False, download=True, transform=transform)

        def collect(ds):
            xs, ys = [], []
            for x, y in torch.utils.data.DataLoader(ds, batch_size=1024):
                xs.append(x.numpy()); ys.append(y.numpy())
            return np.concatenate(xs, 0).astype(np.float32), np.concatenate(ys, 0).astype(np.int32)

        xtr, ytr = collect(train)
        xv, yv = collect(val)

        if digits is not None:
            def filter_digits(x, y):
                mask = np.zeros_like(y, dtype=bool)
                for d in digits:
                    mask |= (y == d)
                x, y = x[mask], y[mask]
                for i, d in enumerate(digits):
                    y = np.where(y == d, i, y)
                return x, y
            xtr, ytr = filter_digits(xtr, ytr)
            xv, yv = filter_digits(xv, yv)

        self.train = (jnp.asarray(xtr), jnp.asarray(ytr))
        self.val = (jnp.asarray(xv), jnp.asarray(yv))


class CIFAR10Dataset(_JaxDataset):
    """CIFAR-10 as ``(N, 32, 32, 3)`` NHWC arrays (Flax convention)."""

    def __init__(
        self,
        ROOT: str = "/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/cifar10/",
        train_fraction: float = 1.0,
        val_fraction: float = 1.0,
    ) -> None:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        import torch

        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])
        train = CIFAR10(root=ROOT, train=True, download=True, transform=transform)
        val = CIFAR10(root=ROOT, train=False, download=True, transform=transform)

        def collect(ds):
            xs, ys = [], []
            for x, y in torch.utils.data.DataLoader(ds, batch_size=512):
                xs.append(x.numpy()); ys.append(y.numpy())
            # (N, C, H, W) -> (N, H, W, C) for Flax
            x = np.concatenate(xs, 0).transpose(0, 2, 3, 1).astype(np.float32)
            return x, np.concatenate(ys, 0).astype(np.int32)

        xtr, ytr = collect(train)
        xv, yv = collect(val)

        if train_fraction < 1.0:
            n = int(len(xtr) * train_fraction)
            xtr, ytr = xtr[:n], ytr[:n]
        if val_fraction < 1.0:
            n = int(len(xv) * val_fraction)
            xv, yv = xv[:n], yv[:n]

        self.train = (jnp.asarray(xtr), jnp.asarray(ytr))
        self.val = (jnp.asarray(xv), jnp.asarray(yv))


class RandomPolynomialDataset(_JaxDataset):
    """Random polynomial of ``degree`` in ``num_vars`` inputs."""

    def __init__(
        self,
        degree: int,
        num_vars: int,
        seed: int,
        n_train: int = 10_000,
        n_val: int = 10_000,
    ) -> None:
        rng = np.random.default_rng(seed)

        power_combinations = []
        for d in range(degree + 1):
            for powers in product(range(d), repeat=num_vars):
                if sum(powers) == d:
                    power_combinations.append(powers)

        self.power_combinations = power_combinations
        self.num_terms = len(power_combinations)
        self.coeffs = rng.normal(size=(self.num_terms,))

        x_train = rng.normal(size=(n_train, num_vars)).astype(np.float32)
        x_val = rng.normal(size=(n_val, num_vars)).astype(np.float32)

        def eval_poly(x):
            y = np.zeros(x.shape[0], dtype=np.float32)
            for coeff, power in zip(self.coeffs, self.power_combinations):
                term = np.ones(x.shape[0], dtype=np.float32)
                for j in range(num_vars):
                    term += x[:, j] ** power[j]
                y += coeff * term
            return y

        y_train = eval_poly(x_train)
        y_val = eval_poly(x_val)

        mean, std = y_train.mean(), y_train.std()
        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std

        self.train = (jnp.asarray(x_train), jnp.asarray(y_train)[:, None])
        self.val = (jnp.asarray(x_val), jnp.asarray(y_val)[:, None])
