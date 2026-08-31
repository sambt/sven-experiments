import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import numpy as np
from itertools import combinations_with_replacement, product
import os

class Toy1DRegressionDataset:
    def __init__(self, n_train=10_000, n_val=10_000, n_test=10_000, seed=0):
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed

        rng = torch.Generator().manual_seed(seed)
        def func(x):
            return torch.exp(-10 * x**2) * torch.sin(2 * x)

        def sample(n):
            x = 2 * torch.rand((n, 1), generator=rng) - 1
            y = func(x)
            return x, y

        xtrain, ytrain = sample(n_train)
        xval, yval = sample(n_val)
        xtest, ytest = sample(n_test)

        mean, std = ytrain.mean(), ytrain.std()
        ytrain = (ytrain - mean) / std
        yval = (yval - mean) / std
        ytest = (ytest - mean) / std

        self.train_dataset = TensorDataset(xtrain, ytrain)
        self.val_dataset = TensorDataset(xval, yval)
        self.test_dataset = TensorDataset(xtest, ytest)

class MNISTDataset:
    def __init__(self, ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/mnist/",digits=None,n_train=None,subsample_seed=0):
        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = MNIST(root=ROOT, train=True, download=True, transform=transform)
        val_dataset = MNIST(root=ROOT, train=False, download=True, transform=transform)

        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        for data,label in torch.utils.data.DataLoader(train_dataset, batch_size=512):
            train_data.append(data)
            train_labels.append(label)
        for data,label in torch.utils.data.DataLoader(val_dataset, batch_size=512):
            val_data.append(data)
            val_labels.append(label)
        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        val_data = torch.cat(val_data, dim=0)
        val_labels = torch.cat(val_labels, dim=0)

        if digits is not None:
            mask_train = torch.zeros_like(train_labels, dtype=torch.bool)
            for d in digits:
                mask_train |= (train_labels == d)
            train_data = train_data[mask_train]
            train_labels = train_labels[mask_train]

            mask_val = torch.zeros_like(val_labels, dtype=torch.bool)
            for d in digits:
                mask_val |= (val_labels == d)
            val_data = val_data[mask_val]
            val_labels = val_labels[mask_val]

            for d,i in zip(digits, range(len(digits))):
                train_labels[train_labels == d] = i
                val_labels[val_labels == d] = i

        # Optional training-set subsampling (for dataset-overparametrized experiments,
        # where P > N_train). Validation set is left full for a stable estimate.
        if n_train is not None and n_train < train_data.shape[0]:
            g = torch.Generator().manual_seed(subsample_seed)
            idx = torch.randperm(train_data.shape[0], generator=g)[:n_train]
            train_data = train_data[idx]
            train_labels = train_labels[idx]

        self.train_dataset = TensorDataset(train_data, train_labels)
        self.val_dataset = TensorDataset(val_data, val_labels)

class RandomPolynomialDataset:
    def __init__(self,degree,num_vars,seed,n_train=10_000,n_val=10_000):
        rng = np.random.default_rng(seed)

        power_combinations = []
        for d in range(degree+1):
            for powers in product(range(d), repeat=num_vars):
                if sum(powers) == d:
                    power_combinations.append(powers)
        
        self.power_combinations = power_combinations
        self.num_terms = len(power_combinations)
        self.coeffs = rng.normal(size=(self.num_terms,))

        x_train = rng.normal(size=(n_train,num_vars))
        x_val = rng.normal(size=(n_val,num_vars))

        y_train = np.zeros(n_train)
        y_val = np.zeros(n_val)
        for i, (coeff, power) in enumerate(zip(self.coeffs,self.power_combinations)):
            term_train = np.ones(n_train)
            term_val = np.ones(n_val)
            for j in range(num_vars):
                term_train += x_train[:,j]**power[j]
                term_val += x_val[:,j]**power[j]
            y_train += coeff * term_train
            y_val += coeff * term_val

        mean = np.mean(y_train)
        std = np.std(y_train)
        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std
        
        self.train_dataset = TensorDataset(torch.tensor(x_train,dtype=torch.float32),
                                           torch.tensor(y_train,dtype=torch.float32).unsqueeze(1))
        self.val_dataset = TensorDataset(torch.tensor(x_val,dtype=torch.float32),
                                         torch.tensor(y_val,dtype=torch.float32).unsqueeze(1))


class CIFAR10Dataset:
    def __init__(self, for_mlp=False, ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/cifar10/",
                 n_train=None, subsample_seed=0):
        if not os.path.isdir(ROOT):
            ROOT = "./torch_datasets/"
        transformations = [transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
        if for_mlp:
            transformations.append(transforms.Lambda(lambda x: x.view(-1)))
        transform = transforms.Compose(transformations)
        train_dataset = CIFAR10(root=ROOT, train=True, download=True, transform=transform)
        val_dataset = CIFAR10(root=ROOT, train=False, download=True, transform=transform)

        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        for data,label in torch.utils.data.DataLoader(train_dataset, batch_size=512):
            train_data.append(data)
            train_labels.append(label)
        for data,label in torch.utils.data.DataLoader(val_dataset, batch_size=512):
            val_data.append(data)
            val_labels.append(label)
        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        val_data = torch.cat(val_data, dim=0)
        val_labels = torch.cat(val_labels, dim=0)

        # Optional training-set subsampling (dataset-overparametrized experiments).
        if n_train is not None and n_train < train_data.shape[0]:
            g = torch.Generator().manual_seed(subsample_seed)
            idx = torch.randperm(train_data.shape[0], generator=g)[:n_train]
            train_data = train_data[idx]
            train_labels = train_labels[idx]

        self.train_dataset = TensorDataset(train_data, train_labels)
        self.val_dataset = TensorDataset(val_data, val_labels)
        

class CharTextDataset:
    """Char-level language-modeling dataset (tiny-shakespeare by default).

    Produces fixed (input, target) sequence pairs for next-token prediction:
    the corpus is encoded to char ids, split train/val by position, and chunked
    into non-overlapping blocks of length ``block_size`` (target = input shifted
    by one). Exposes ``vocab_size`` and ``block_size`` for model construction.
    ``n_train`` optionally subsamples the number of training sequences (for the
    critical-batch / data-scaling studies).
    """
    def __init__(self,
                 ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/shakespeare/",
                 block_size=128, val_fraction=0.1, n_train=None, subsample_seed=0):
        path = os.path.join(ROOT, "input.txt")
        if not os.path.isfile(path):
            path = "./torch_datasets/shakespeare/input.txt"
        with open(path, "r") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.block_size = block_size
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

        n_val = int(len(data) * val_fraction)
        train_ids, val_ids = data[:-n_val], data[-n_val:]

        def chunk(ids):
            # non-overlapping (block_size + 1) windows -> (x, y) each (N, block_size)
            n_seq = (len(ids) - 1) // block_size
            usable = ids[: n_seq * block_size + 1]
            x = torch.stack([usable[i * block_size: (i + 1) * block_size] for i in range(n_seq)])
            y = torch.stack([usable[i * block_size + 1: (i + 1) * block_size + 1] for i in range(n_seq)])
            return x, y

        xtr, ytr = chunk(train_ids)
        xva, yva = chunk(val_ids)

        if n_train is not None and n_train < xtr.shape[0]:
            g = torch.Generator().manual_seed(subsample_seed)
            idx = torch.randperm(xtr.shape[0], generator=g)[:n_train]
            xtr, ytr = xtr[idx], ytr[idx]

        self.train_dataset = TensorDataset(xtr, ytr)
        self.val_dataset = TensorDataset(xva, yva)
