import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import numpy as np
from itertools import combinations_with_replacement, product
import json
import os
import zlib

# ---------------------------------------------------------------------------
# Split / subsample helpers (pure functions, no global RNG, no data files)
#
# Every dataset class exposes ``train_dataset`` / ``val_dataset`` / ``test_dataset``,
# records the ``split_seed`` that produced the split, and exposes the three sizes as
# ``n_train`` / ``n_val`` / ``n_test`` (C-E1).  The split seed is a dataset-config value:
# it is independent of the model seed, the loader seed and the subsample seed, so
# changing ``n_train`` never moves the held-out data.
# ---------------------------------------------------------------------------

def derive_seeds(seed, *names):
    """One independent generator seed per name, derived from a single dataset seed.

    ``zlib.crc32`` (not Python's salted ``hash``) so the seeds are reproducible across
    processes.  Used to give the synthetic pool / val / test draws separate generators.
    """
    return tuple(int((int(seed) ^ zlib.crc32(name.encode())) & 0x7FFF_FFFF) for name in names)


def holdout_split_indices(n, n_holdout, split_seed):
    """Split ``range(n)`` into (train_idx, holdout_idx) by a permutation of ``split_seed``.

    Pure: draws from its own generator, so the split does not depend on the position of
    the call in the process, on the model seed or on the subsample seed.  ``n_holdout``
    is clipped to ``n - 1``.
    """
    n = int(n)
    n_holdout = max(0, min(int(n_holdout), n - 1))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(split_seed)))
    return perm[n_holdout:], perm[:n_holdout]


def subsample_indices(n_pool, n_train, subsample_seed):
    """Indices of ``n_train`` examples drawn from a pool of ``n_pool`` (train part only).

    A prefix of one permutation, so the draws are nested: the n_train = 150 set is a
    subset of the n_train = 1200 set for the same ``subsample_seed``.  ``n_train`` of
    ``None`` (or exactly the pool size) returns the whole pool in its natural order.

    An ``n_train`` **larger than the pool RAISES** (scope update 2026-09-18): silently
    clamping it mislabels the run (the record would claim n_data = 60,000 while training
    on the 50,000-example MNIST train part) and so corrupts every P/N figure.
    """
    n_pool = int(n_pool)
    if n_train is None:
        return torch.arange(n_pool)
    n_train = int(n_train)
    if not 1 <= n_train <= n_pool:
        raise ValueError(
            f"n_train={n_train} is outside the available training pool of {n_pool} "
            "examples (val/test are held out and are never trained on). Lower n_train "
            "-- e.g. the MNIST train part is 50,000, so the overparam sweep's top point "
            "is N=50000 -- or raise pool_size for the synthetic datasets.")
    if n_train == n_pool:
        return torch.arange(n_pool)
    g = torch.Generator().manual_seed(int(subsample_seed))
    return torch.randperm(n_pool, generator=g)[:n_train]


def contiguous_split_bounds(n, val_fraction=0.1, test_fraction=0.1):
    """Contiguous train / val / test ``(start, stop)`` bounds by position (80/10/10).

    Used for the character corpus: text split by position, train first, so no window of
    one split can overlap another.
    """
    n = int(n)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_train = n - n_val - n_test
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)


def _stack_dataset(dataset, batch_size=512):
    """Materialise a torchvision dataset into (data, labels) tensors."""
    data, labels = [], []
    for d, l in DataLoader(dataset, batch_size=batch_size):
        data.append(d)
        labels.append(l)
    return torch.cat(data, dim=0), torch.cat(labels, dim=0)


def _subsample(tensors, n_train, subsample_seed):
    """:func:`subsample_indices` applied to aligned tensors; a no-op copies nothing.

    Inherits the raise on ``n_train`` > the available pool.
    """
    n = tensors[0].shape[0]
    if n_train is None or int(n_train) == n:
        return tensors
    idx = subsample_indices(n, n_train, subsample_seed)
    return tuple(t[idx] for t in tensors)


class Toy1DRegressionDataset:
    """1-D regression target with a fixed training pool (C-D3).

    The training pool (``pool_size`` = 10,000), the validation set and the test set are
    drawn from three *separate* generators whose seeds derive from ``split_seed``
    (defaults to ``seed``), and the targets are normalised by the **pool** mean/std.
    Validation, test and the target scale are therefore identical at every ``n_train``,
    which subsamples the pool.
    """
    def __init__(self, n_train=10_000, n_val=10_000, n_test=10_000, seed=0,
                 pool_size=10_000, subsample_seed=0, split_seed=None):
        self.seed = seed
        # split_seed is accepted by every dataset class (CONTRACTS.md "Datasets") so the
        # same config key works everywhere; here it seeds the pool / val / test draws.
        self.split_seed = int(seed if split_seed is None else split_seed)
        self.pool_size = int(pool_size)
        pool_seed, val_seed, test_seed = derive_seeds(self.split_seed, "pool", "val", "test")

        def func(x):
            return torch.exp(-10 * x**2) * torch.sin(2 * x)

        def sample(n, s):
            g = torch.Generator().manual_seed(s)
            x = 2 * torch.rand((n, 1), generator=g) - 1
            return x, func(x)

        x_pool, y_pool = sample(self.pool_size, pool_seed)
        xval, yval = sample(n_val, val_seed)
        xtest, ytest = sample(n_test, test_seed)

        mean, std = y_pool.mean(), y_pool.std()
        y_pool = (y_pool - mean) / std
        yval = (yval - mean) / std
        ytest = (ytest - mean) / std

        idx = subsample_indices(self.pool_size, n_train, subsample_seed)
        xtrain, ytrain = x_pool[idx], y_pool[idx]

        self.train_dataset = TensorDataset(xtrain, ytrain)
        self.val_dataset = TensorDataset(xval, yval)
        self.test_dataset = TensorDataset(xtest, ytest)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)

class MNISTDataset:
    """MNIST with a real validation split: official train -> 50,000 / 10,000 by
    ``split_seed``; ``test_dataset`` is the official test set (C-E1).  ``n_train``
    subsamples the **train** part only.  Normalisation constants are the published
    full-train statistics and are deliberately not recomputed for the 50k split."""
    def __init__(self, ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/mnist/",digits=None,n_train=None,subsample_seed=0,
                 n_val=10_000, split_seed=1234):
        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = MNIST(root=ROOT, train=True, download=True, transform=transform)
        test_dataset = MNIST(root=ROOT, train=False, download=True, transform=transform)

        train_data, train_labels = _stack_dataset(train_dataset)
        test_data, test_labels = _stack_dataset(test_dataset)

        if digits is not None:
            def select(data, labels):
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for d in digits:
                    mask |= (labels == d)
                data, labels = data[mask], labels[mask]
                # relabel to 0..len(digits)-1 from the ORIGINAL labels, so digit lists
                # whose remapping overlaps (e.g. [1, 0]) are not relabelled twice
                remapped = torch.empty_like(labels)
                for i, d in enumerate(digits):
                    remapped[labels == d] = i
                return data, remapped
            train_data, train_labels = select(train_data, train_labels)
            test_data, test_labels = select(test_data, test_labels)

        # Held-out validation split of the official training set, by a fixed split_seed
        # independent of the model / loader / subsample seeds.
        self.split_seed = int(split_seed)
        tr_idx, va_idx = holdout_split_indices(train_data.shape[0], n_val, self.split_seed)
        val_data, val_labels = train_data[va_idx], train_labels[va_idx]
        train_data, train_labels = train_data[tr_idx], train_labels[tr_idx]

        # Optional training-set subsampling (for dataset-overparametrized experiments,
        # where P > N_train). Validation and test sets are left full.
        train_data, train_labels = _subsample((train_data, train_labels), n_train, subsample_seed)

        self.train_dataset = TensorDataset(train_data, train_labels)
        self.val_dataset = TensorDataset(val_data, val_labels)
        self.test_dataset = TensorDataset(test_data, test_labels)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)


def monomial_powers(degree, num_vars):
    """Exponent tuples of every monomial of total degree <= ``degree`` in ``num_vars``
    variables, including the constant and the linear terms.

    ``C(num_vars + degree, degree)`` of them, so 210 for degree 4 in 6 variables.
    """
    powers = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(num_vars), d):
            p = [0] * num_vars
            for j in combo:
                p[j] += 1
            powers.append(tuple(p))
    return powers


def _double_factorial_odd(p):
    """(2p - 1)!! = E[x^(2p)] for x ~ N(0, 1); 1 for p = 0 by the (-1)!! = 1 convention."""
    out = 1
    for i in range(1, int(p) + 1):
        out *= 2 * i - 1
    return out


def monomial_rms(powers):
    """sqrt(E[m(x)^2]) for the monomial with exponents ``powers`` and x ~ N(0, I).

    E[m^2] = prod_j E[x_j^(2 p_j)] = prod_j (2 p_j - 1)!!.
    """
    m2 = 1.0
    for p in powers:
        m2 *= _double_factorial_odd(p)
    return float(np.sqrt(m2))


def eval_polynomial(x, coeffs, powers):
    """Evaluate sum_i coeffs[i] * prod_j x[:, j] ** powers[i][j].  Factors multiplied."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros(x.shape[0], dtype=np.float64)
    for coeff, p in zip(coeffs, powers):
        term = np.ones(x.shape[0], dtype=np.float64)
        for j, pj in enumerate(p):
            if pj:
                term = term * x[:, j] ** pj
        y += coeff * term
    return y


class RandomPolynomialDataset:
    """Random polynomial of total degree <= ``degree`` in ``num_vars`` variables (C-D1).

    All monomials of total degree <= degree (constant and linear included, 210 for
    degree 4 in 6 variables), factors **multiplied**, x ~ N(0, I).  The coefficient of
    monomial m is ``N(0,1) / sqrt(E[m(x)^2])`` with ``E[m^2] = prod_j (2 p_j - 1)!!``
    ("variance-normalised monomials"), so no single degree dominates the target.

    Fixed training pool + separate val / test generators as in C-D3: the validation and
    test tensors and the target scale do not move with ``n_train``.  The legacy additive
    generator is kept as :class:`AdditiveCubicDataset`.
    """
    def __init__(self, degree, num_vars, seed, n_train=10_000, n_val=10_000,
                 n_test=10_000, pool_size=10_000, subsample_seed=0, split_seed=None):
        self.seed = seed
        # ``seed`` fixes the target function (the coefficients); ``split_seed`` fixes
        # which x's land in the pool / val / test and defaults to it (CONTRACTS.md).
        self.split_seed = int(seed if split_seed is None else split_seed)
        self.degree = int(degree)
        self.num_vars = int(num_vars)
        self.pool_size = int(pool_size)

        coeff_seed, = derive_seeds(seed, "coeffs")
        pool_seed, val_seed, test_seed = derive_seeds(self.split_seed, "pool", "val", "test")

        self.power_combinations = monomial_powers(degree, num_vars)
        self.num_terms = len(self.power_combinations)
        rms = np.array([monomial_rms(p) for p in self.power_combinations])
        self.coeffs = np.random.default_rng(coeff_seed).normal(size=(self.num_terms,)) / rms

        def sample(n, s):
            x = np.random.default_rng(s).normal(size=(n, num_vars))
            return x, eval_polynomial(x, self.coeffs, self.power_combinations)

        x_pool, y_pool = sample(self.pool_size, pool_seed)
        x_val, y_val = sample(n_val, val_seed)
        x_test, y_test = sample(n_test, test_seed)

        # Normalise by the POOL statistics: the target scale is the same at every n_train.
        mean, std = np.mean(y_pool), np.std(y_pool)
        y_pool = (y_pool - mean) / std
        y_val = (y_val - mean) / std
        y_test = (y_test - mean) / std
        self.target_mean, self.target_std = float(mean), float(std)

        idx = subsample_indices(self.pool_size, n_train, subsample_seed).numpy()
        x_train, y_train = x_pool[idx], y_pool[idx]

        def as_dataset(x, y):
            return TensorDataset(torch.tensor(x, dtype=torch.float32),
                                 torch.tensor(y, dtype=torch.float32).unsqueeze(1))

        self.train_dataset = as_dataset(x_train, y_train)
        self.val_dataset = as_dataset(x_val, y_val)
        self.test_dataset = as_dataset(x_test, y_test)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)


def legacy_additive_powers(degree, num_vars):
    """The exponent tuples the pre-2026-09 polynomial generator enumerated.

    ``product(range(d), ...)`` excludes any exponent >= d, so the constant and all
    linear monomials are missing (185 tuples for degree 4 in 6 variables).
    """
    powers = []
    for d in range(degree + 1):
        for p in product(range(d), repeat=num_vars):
            if sum(p) == d:
                powers.append(p)
    return powers


class AdditiveCubicDataset:
    """The pre-2026-09 "random polynomial" generator, kept so historical results stay
    reproducible and correctly named (C-D1 / F1).

    Despite the name it is an **additive cubic**: the per-variable factors were *added*
    (``term += x[:, j] ** p[j]`` starting from ones) rather than multiplied, and the
    exponent enumeration missed the constant and linear monomials, so the target
    collapses to ``c0 + sum_j (a1 x_j + a2 x_j^2 + a3 x_j^3)`` -- 19 effective features.
    Train and validation targets are bit-for-bit identical to the legacy class for the
    same ``seed`` / ``n_train`` / ``n_val``: the RNG draw order (coeffs, x_train, x_val)
    and the y_train-based normalisation are unchanged.  The test split is drawn *after*
    x_val, so adding it does not perturb train or val.
    """
    def __init__(self, degree, num_vars, seed, n_train=10_000, n_val=10_000, n_test=10_000,
                 split_seed=None):
        rng = np.random.default_rng(seed)
        self.seed = seed
        # ``split_seed`` is accepted (so the shared config key never breaks) and recorded,
        # but the legacy generator draws coeffs / train / val / test from ONE RNG seeded
        # by ``seed``: reseeding the split would break the bit-for-bit reproduction.
        self.split_seed = int(seed if split_seed is None else split_seed)

        self.power_combinations = legacy_additive_powers(degree, num_vars)
        self.num_terms = len(self.power_combinations)
        self.coeffs = rng.normal(size=(self.num_terms,))

        x_train = rng.normal(size=(n_train, num_vars))
        x_val = rng.normal(size=(n_val, num_vars))
        x_test = rng.normal(size=(n_test, num_vars))

        def legacy_eval(x):
            y = np.zeros(x.shape[0])
            for coeff, power in zip(self.coeffs, self.power_combinations):
                term = np.ones(x.shape[0])
                for j in range(num_vars):
                    term += x[:, j]**power[j]
                y += coeff * term
            return y

        y_train = legacy_eval(x_train)
        y_val = legacy_eval(x_val)
        y_test = legacy_eval(x_test)

        mean = np.mean(y_train)
        std = np.std(y_train)
        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std
        y_test = (y_test - mean) / std

        def as_dataset(x, y):
            return TensorDataset(torch.tensor(x, dtype=torch.float32),
                                 torch.tensor(y, dtype=torch.float32).unsqueeze(1))

        self.train_dataset = as_dataset(x_train, y_train)
        self.val_dataset = as_dataset(x_val, y_val)
        self.test_dataset = as_dataset(x_test, y_test)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)


class CIFAR10Dataset:
    """CIFAR-10 with a real validation split: official train -> 45,000 / 5,000 by
    ``split_seed``; ``test_dataset`` is the official test set (C-E1).  ``n_train``
    subsamples the **train** part only.  Channel statistics are the published
    full-train values and are deliberately not recomputed for the 45k split."""
    def __init__(self, for_mlp=False, ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/cifar10/",
                 n_train=None, subsample_seed=0, n_val=5_000, split_seed=1234):
        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"
        transformations = [transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
        if for_mlp:
            transformations.append(transforms.Lambda(lambda x: x.view(-1)))
        transform = transforms.Compose(transformations)
        train_dataset = CIFAR10(root=ROOT, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=ROOT, train=False, download=True, transform=transform)

        train_data, train_labels = _stack_dataset(train_dataset)
        test_data, test_labels = _stack_dataset(test_dataset)

        self.split_seed = int(split_seed)
        tr_idx, va_idx = holdout_split_indices(train_data.shape[0], n_val, self.split_seed)
        val_data, val_labels = train_data[va_idx], train_labels[va_idx]
        train_data, train_labels = train_data[tr_idx], train_labels[tr_idx]

        # Optional training-set subsampling (dataset-overparametrized experiments).
        train_data, train_labels = _subsample((train_data, train_labels), n_train, subsample_seed)

        self.train_dataset = TensorDataset(train_data, train_labels)
        self.val_dataset = TensorDataset(val_data, val_labels)
        self.test_dataset = TensorDataset(test_data, test_labels)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)


class CharTextDataset:
    """Char-level language-modeling dataset (tiny-shakespeare by default).

    Produces fixed (input, target) sequence pairs for next-token prediction:
    the corpus is encoded to char ids, split train/val/test **contiguously by
    position** (80/10/10, train first), and each part is chunked separately into
    non-overlapping blocks of length ``block_size`` (target = input shifted by one), so
    no block spans a split boundary.  Exposes ``vocab_size`` and ``block_size`` for
    model construction.  ``n_train`` optionally subsamples the number of training
    sequences (for the critical-batch / data-scaling studies).
    """
    def __init__(self,
                 ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/shakespeare/",
                 block_size=128, val_fraction=0.1, test_fraction=0.1,
                 n_train=None, subsample_seed=0, split_seed=None):
        path = os.path.join(ROOT, "input.txt")
        if not os.path.isfile(path):
            path = "./torch_datasets/shakespeare/input.txt"
        with open(path, "r") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        # Contiguous split by position: no randomness, so there is no split seed of its
        # own.  The argument is accepted (shared config key) and recorded if given.
        self.bounds = contiguous_split_bounds(len(data), val_fraction, test_fraction)
        self.split_seed = None if split_seed is None else int(split_seed)

        def chunk(ids):
            # non-overlapping (block_size + 1) windows -> (x, y) each (N, block_size)
            n_seq = (len(ids) - 1) // block_size
            usable = ids[: n_seq * block_size + 1]
            x = torch.stack([usable[i * block_size: (i + 1) * block_size] for i in range(n_seq)])
            y = torch.stack([usable[i * block_size + 1: (i + 1) * block_size + 1] for i in range(n_seq)])
            return x, y

        (tr0, tr1), (va0, va1), (te0, te1) = self.bounds
        xtr, ytr = chunk(data[tr0:tr1])
        xva, yva = chunk(data[va0:va1])
        xte, yte = chunk(data[te0:te1])

        xtr, ytr = _subsample((xtr, ytr), n_train, subsample_seed)

        self.train_dataset = TensorDataset(xtr, ytr)
        self.val_dataset = TensorDataset(xva, yva)
        self.test_dataset = TensorDataset(xte, yte)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)


class _BlockDataset(torch.utils.data.Dataset):
    """Lazy fixed-block view over a memmapped uint16 token array (next-token LM)."""
    def __init__(self, data, block_size, n_blocks=None, name="tokens"):
        self.data = data
        self.bs = block_size
        max_blocks = (len(data) - 1) // block_size
        # A truncated / empty .bin used to give a NEGATIVE length instead of failing.
        if max_blocks < 1:
            raise ValueError(f"{name}: {len(data)} tokens is too short for one block of "
                             f"{block_size} (+1 for the shifted target)")
        self.n = min(n_blocks, max_blocks) if n_blocks else max_blocks

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.bs
        x = torch.from_numpy(self.data[s:s + self.bs].astype("int64"))
        y = torch.from_numpy(self.data[s + 1:s + self.bs + 1].astype("int64"))
        return x, y


class TokenBinDataset:
    """Token-level LM dataset over nanoGPT-style uint16 .bin shards.

    Reads ``{ROOT}/train.bin``, ``{ROOT}/val.bin`` and ``{ROOT}/test.bin`` (uint16 GPT-2
    BPE token ids, as written by experiments/data_prep/prepare_tokens.py from **disjoint
    documents** of one shared stream), memmaps them, and serves non-overlapping
    ``block_size`` (input, target) pairs.  One pass over train.bin = the full token
    budget.  ``n_train_blocks`` optionally caps it.  The split is a property of the
    files, so there is no split seed; ``token_counts.json`` (if present) is recorded.
    """
    def __init__(self,
                 ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/openwebtext_gpt2",
                 block_size=1024, n_train_blocks=None, val_blocks=200, test_blocks=None,
                 vocab_size=50304, split_seed=None):
        self.block_size = block_size
        self.vocab_size = vocab_size
        # No split seed of its own (the split is a property of the files); the argument
        # is accepted (shared config key) and recorded if given.
        self.split_seed = None if split_seed is None else int(split_seed)
        if test_blocks is None:
            test_blocks = val_blocks

        def blocks(name, n_blocks):
            path = os.path.join(ROOT, name + ".bin")
            data = np.memmap(path, dtype=np.uint16, mode="r")
            return _BlockDataset(data, block_size, n_blocks, name=path)

        self.train_dataset = blocks("train", n_train_blocks)
        self.val_dataset = blocks("val", val_blocks)
        self.test_dataset = blocks("test", test_blocks)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)

        counts_path = os.path.join(ROOT, "token_counts.json")
        self.token_counts = None
        if os.path.isfile(counts_path):
            with open(counts_path) as f:
                self.token_counts = json.load(f).get("tokens")
