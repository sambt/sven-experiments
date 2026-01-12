import copy
import json
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
import pandas as pd

from .experiment_utils import train_loop_svd, train_loop_standard, set_seed, process_hparam_config
from sv3.svd_sgd import SVDOptimizer
from sv3.nn import FunctionalModelJac
import copy
from omegaconf import OmegaConf
from itertools import product

import pickle

import os

def mnist_scan_varyRtol(cfg):
    """
    Hyperparameter scan over batch size, k fraction, and learning rate for the toy 1D regression task.
    Hydra passes this function as a partial; cfg holds top-level params (lr, rtol, num_epochs, etc.).
    """
    exp_cfg = cfg.experiment
    device = cfg.device

    # create output directory
    os.makedirs(f"experiment_results/{exp_cfg['name']}", exist_ok=True) 

    hparams = process_hparam_config(OmegaConf.to_container(exp_cfg, resolve=True))
    k_scan_values = hparams['k_fractions'] if 'k_fractions' in hparams else hparams['k_values']
    use_k_values = 'k_values' in hparams
    hparam_scan = product(
        hparams['batch_size'],
        k_scan_values,
        hparams['lrs'],
        hparams['rtol'],
        hparams['svd_mode'])

    dataset = instantiate(cfg.dataset)

    def loss_fn(pred,y):
        fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = fn(pred, y).squeeze()
        return loss
    
    print(f"Starting MNIST scan with SVD optimizer")  
    
    # instantiate base model to get initial weights for all models
    set_seed(exp_cfg["model_seed"])
    base_model = instantiate(cfg.model)
    print("Model instantiated with config:\n", cfg.model)
    init_state = copy.deepcopy(base_model.state_dict())
    del base_model # free memory

    for batch_size, k_item, lr, rtol, svd_mode in hparam_scan:
        k = max(1, int(k_item * batch_size)) if not use_k_values else k_item
        print(f"\nRunning SVD optimization with batch_size={batch_size}, k={k}, lr={lr}, rtol={rtol}, svd_mode={svd_mode}")

        OUTPUT_FILE = f"{exp_cfg['output_file']}_bs{batch_size}_width{cfg.experiment.mlp_width}_k{k}_lr{lr}_rtol{rtol}_svd{svd_mode}"
        OUTPUT_PATH = f"experiment_results/{exp_cfg['name']}/{OUTPUT_FILE}_df.pkl"
        if os.path.exists(OUTPUT_PATH):
            print(f"Output file {OUTPUT_PATH} already exists; skipping this run.")
            continue

        model = instantiate(cfg.model)
        model.load_state_dict(init_state)
        
        train_model = FunctionalModelJac(model, loss_fn, device)
        optimizer = SVDOptimizer(train_model,lr=lr,k=k,rtol=rtol,track_svd_info=True,svd_mode=svd_mode)

        train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(exp_cfg["loader_seed"]))
        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

        train_model, losses, optimizer = train_loop_svd(train_model, optimizer, loss_fn, train_loader, val_loader, exp_cfg["num_epochs"], device, track_acc=True)

        results = [{
            "batch_size": batch_size,
            "k_fraction": k / batch_size,
            "k": k,
            "lr": lr,
            "rtol": rtol,
            "mlp_width": cfg.experiment.mlp_width,
            "losses": losses,
            "svd_info": getattr(optimizer, "svd_info", {}),
            "svd_mode": svd_mode,
            "optimizer":"SVD"
        }]

        df = pd.DataFrame(results)
        df.to_pickle(OUTPUT_PATH)
        
        torch.compiler.reset()

    # run MLP trainings with other optimizers for comparison
    print("="*80)
    print("Running MLP trainings with standard optimizers")

    hparam_scan_standard = product(
        hparams['batch_size'],
        hparams['lrs_standard'],
        hparams['optimizers_standard'])
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for batch_size, lr, optim_name in hparam_scan_standard:
        print(f"\nRunning MLP with batch_size={batch_size}, lr={lr}, optimizer={optim_name}")

        OUTPUT_FILE = f"{exp_cfg['output_file']}_bs{batch_size}_width{cfg.experiment.mlp_width}_lr{lr}_optim{optim_name}"
        OUTPUT_PATH = f"experiment_results/{exp_cfg['name']}/{OUTPUT_FILE}_df.pkl"
        if os.path.exists(OUTPUT_PATH):
            print(f"Output file {OUTPUT_PATH} already exists; skipping this run.")
            continue

        model = instantiate(cfg.model)
        model.load_state_dict(init_state)
        model = model.to(device)
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
        train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(exp_cfg["loader_seed"]))
        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

        train_model, losses = train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, exp_cfg["num_epochs"], device, track_acc=True)

        results = [{
            "batch_size": batch_size,
            "k_fraction": None,
            "k": None,
            "lr": lr,
            "rtol": None,
            "mlp_width": cfg.experiment.mlp_width,
            "losses": losses,
            "optimizer": optim_name,
            "svd_info": None,
            "svd_mode": None
        }]
        df = pd.DataFrame(results)
        df.to_pickle(OUTPUT_PATH)
    
    return 
