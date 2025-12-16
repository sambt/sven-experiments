import copy
import json
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from .experiment_utils import train_loop_svd_mse
from sv3.svd_sgd import SVDOptimizer
from sv3.nn import FunctionalModelJac

import pickle

import os

def toy_1d_scan(cfg):
    """
    Hyperparameter scan over batch size, k fraction, and learning rate for the toy 1D regression task.
    Hydra passes this function as a partial; cfg holds top-level params (lr, rtol, num_epochs, etc.).
    """
    exp_cfg = cfg.experiment
    device = cfg.device

    if "batch_sizes" not in exp_cfg:
        batch_sizes = [32, 64, 128, 256, 512]
        print("No batch sizes specified; defaulting to", batch_sizes)
    else:
        batch_sizes = exp_cfg["batch_sizes"]

    if "k_fractions" not in exp_cfg:
        k_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        print("No k fractions specified; defaulting to", k_fractions)
    else:
        k_fractions = exp_cfg["k_fractions"]

    if "lrs" not in exp_cfg:
        lrs = [0.01, 0.1, 0.5, 1.0]
        print("No learning rates specified; defaulting to", lrs)
    else:
        lrs = exp_cfg["lrs"]

    dataset = instantiate(cfg.dataset)
    base_model = instantiate(cfg.model)
    init_state = copy.deepcopy(base_model.state_dict())
    del base_model # free memory

    def loss_fn(pred, y):
        loss = (pred - y) ** 2
        loss = loss.sum(dim=-1)  # shape (B,)
        return loss

    results = {}
    print(f"Starting toy_1d_scan with\n batch_sizes={batch_sizes}\n k_fractions={k_fractions}\n lrs={lrs}")

    for batch_size in batch_sizes:
        for k_fraction in k_fractions:
            for lr in lrs:
                k = max(1, int(k_fraction * batch_size))
                print(f"\nRunning batch_size={batch_size}, k_fraction={k_fraction} (k={k}), lr={lr}")

                model = instantiate(cfg.model)
                model.load_state_dict(init_state)
                
                train_model = FunctionalModelJac(model, loss_fn, device)
                optimizer = SVDOptimizer(train_model,lr=lr,k=k,rtol=exp_cfg["rtol"])

                train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(exp_cfg["loader_seed"]))
                val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

                train_model, losses, optimizer = train_loop_svd_mse(train_model, optimizer, train_loader, val_loader, exp_cfg["num_epochs"], device)

                results[(batch_size, k_fraction, lr)] = {
                    "losses": losses,
                    "svd_info": getattr(optimizer, "svd_info", {}),
                }

    output_file = exp_cfg['output_file']
    os.makedirs(f"experiment_results/{exp_cfg['name']}", exist_ok=True)
    with open(f"experiment_results/{exp_cfg['name']}/{output_file}", "wb") as f:
        pickle.dump(results, f)
    return results
