import torch
from torch.utils.data import DataLoader, TensorDataset

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