import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import numpy as np
from itertools import combinations_with_replacement, product

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
    def __init__(self, ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/mnist/"):
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

        self.train_dataset = TensorDataset(train_data, train_labels)
        self.val_dataset = TensorDataset(val_data, val_labels)

class CIFAR10Dataset:
    def __init__(self,for_mlp=False,ROOT="/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/torch/cifar10/"):
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
        