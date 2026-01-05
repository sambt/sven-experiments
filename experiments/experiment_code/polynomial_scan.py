import copy
import json
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
import pandas as pd

from .experiment_utils import train_loop_svd, train_loop_standard, listify, add_row
from sv3.svd_sgd import SVDOptimizer
from sv3.nn import FunctionalModelJac
import copy
from omegaconf import OmegaConf

import pickle

import os

def polynomial_scan(cfg):
    """
    Hyperparameter scan over batch size, k fraction, and learning rate for the toy 1D regression task.
    Hydra passes this function as a partial; cfg holds top-level params (lr, rtol, num_epochs, etc.).
    """
    exp_cfg = cfg.experiment
    device = cfg.device

    if "batch_size" not in exp_cfg:
        batch_size = 32
        print("No batch size specified; defaulting to", batch_size)
    else:
        batch_size = exp_cfg["batch_size"]

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

    if "lrs_standard" not in exp_cfg:
        lrs_standard = [1e-4,1e-3,1e-2,1e-1]
        print("No learning rates for standard optimizers specified; defaulting to", lrs_standard)
    else:
        lrs_standard = exp_cfg["lrs_standard"]

    if "optimizers_standard" not in exp_cfg:
        optimizers_standard = ['Adam','AdamW','SGD','RMSprop','Muon']
        print("No standard optimizers specified; defaulting to", optimizers_standard)
    else:
        optimizers_standard = exp_cfg["optimizers_standard"]


    # set up output file
    OUTPUT_DIR = f"experiment_results/{exp_cfg['name']}"
    OUTPUT_FILE = f"{OUTPUT_DIR}/{exp_cfg['output_file']}_bs{batch_size}_width{cfg.experiment.mlp_width}_df.pkl"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set up training
    dataset = instantiate(cfg.dataset)

    def loss_fn(pred,y):
        # assuming 1d output, loss should have shape (B,1)
        loss = (pred - y)**2
        loss = loss.squeeze()
        return loss

    results = []
    
    print(f"Starting polynomial scan with SVD optimizer")  
    
    # instantiate base model to get initial weights for all models
    base_model = instantiate(cfg.model)
    print("Model instantiated with config:\n", cfg.model)
    print(base_model)
    init_state = copy.deepcopy(base_model.state_dict())
    del base_model # free memory

    for k_fraction in k_fractions:
        for lr in lrs:
            k = max(1, int(k_fraction * batch_size))
            print(f"\nRunning batch_size={batch_size}, k_fraction={k_fraction} (k={k}), lr={lr}")

            model = instantiate(cfg.model)
            model.load_state_dict(init_state)
            
            train_model = FunctionalModelJac(model, loss_fn, device)
            optimizer = SVDOptimizer(train_model,lr=lr,k=k,rtol=exp_cfg["rtol"],track_svd_info=True)

            train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(exp_cfg["loader_seed"]))
            val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

            train_model, losses, optimizer = train_loop_svd(train_model, optimizer, loss_fn, train_loader, val_loader, exp_cfg["num_epochs"], device, track_acc=True)

            results.append({
                "batch_size": batch_size,
                "k_fraction": k_fraction,
                "k": k,
                "lr": lr,
                "mlp_width": cfg.experiment.mlp_width,
                "losses": losses,
                "svd_info": getattr(optimizer, "svd_info", {}),
                "optimizer":"SVD"
            })

            torch.compiler.reset()

    # run MLP trainings with other optimizers for comparison
    print("="*80)
    print("Running MLP trainings with standard optimizers")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    results_standard = []
    
    for lr in lrs_standard:
        for optim_name in optimizers_standard:
            print(f"\nRunning MLP with batch_size={batch_size}, lr={lr}, optimizer={optim_name}")
            model = instantiate(cfg.model)
            model.load_state_dict(init_state)
            model = model.to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
            train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(exp_cfg["loader_seed"]))
            val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

            train_model, losses = train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, exp_cfg["num_epochs"], device, track_acc=True)

            results_standard.append({
                "batch_size": batch_size,
                "k_fraction": None,
                "k": None,
                "lr": lr,
                "mlp_width": cfg.experiment.mlp_width,
                "losses": losses,
                "optimizer": optim_name,
                "svd_info": None
            })

    df = pd.DataFrame(results + results_standard)
    df.to_pickle(OUTPUT_FILE)
    print(f"Saved results to {OUTPUT_FILE}")