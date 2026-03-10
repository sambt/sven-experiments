import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Iterable
from typing import Any
import os
import pandas as pd
import fcntl
import tempfile
import shutil
import time
import random

def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: The random seed to use for all random number generators.
        deterministic: If True, enables CUDA deterministic algorithms for full
            reproducibility. This may impact performance. Default is False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def listify(settings):
    if type(settings) is list or type(settings) is tuple:
        return settings
    else:
        return [settings]
    
def process_hparam_config(cfg) -> dict[str,Iterable]:
    output = {}
    if "batch_size" not in cfg:
        output['batch_size'] = listify(32)
        print("No batch size specified; defaulting to", output['batch_size'])
    else:
        output['batch_size'] = listify(cfg["batch_size"])

    if "k_fractions" not in cfg and "k_values" not in cfg:
        output['k_fractions'] = [0.1, 0.25, 0.5, 0.75, 1.0]
        print("No k_values or k_fractions specified; defaulting to fractions = ", output['k_fractions'])
    else:
        assert ("k_values" in cfg) ^ ("k_fractions" in cfg), "Specify either k_values or k_fractions, not both."
        if "k_fractions" in cfg:
            output['k_fractions'] = listify(cfg["k_fractions"])
        else:
            output['k_values'] = listify(cfg["k_values"])

    if "lrs" not in cfg:
        output['lrs'] = [0.01, 0.1, 0.5, 1.0]
        print("No learning rates specified; defaulting to", output['lrs'])
    else:
        output['lrs'] = listify(cfg["lrs"])

    if "rtol" not in cfg:
        output['rtol'] = listify(1e-3)
        print("No rtol specified; defaulting to", output['rtol'])
    else:
        output['rtol'] = listify(cfg["rtol"])

    if "svd_mode" not in cfg:
        output['svd_mode'] = listify('randomized')
        print("No SVD mode specified; defaulting to 'randomized'")
    else:
        output['svd_mode'] = listify(cfg["svd_mode"])

    if "lrs_standard" not in cfg:
        output['lrs_standard'] = [1e-4,1e-3,1e-2,1e-1]
        print("No learning rates for standard optimizers specified; defaulting to", output['lrs_standard'])
    else:
        output['lrs_standard'] = listify(cfg["lrs_standard"])

    if "optimizers_standard" not in cfg:
        output['optimizers_standard'] = ['Adam','AdamW','SGD','RMSprop','Muon']
        print("No standard optimizers specified; defaulting to", output['optimizers_standard'])
    else:
        output['optimizers_standard'] = listify(cfg["optimizers_standard"])

    if "microbatch_sizes" in cfg:
        output['microbatch_sizes'] = listify(cfg["microbatch_sizes"])
    else:
        output['microbatch_sizes'] = [None]

    if "param_fractions" in cfg:
        output['param_fractions'] = listify(cfg["param_fractions"])
    else:
        output['param_fractions'] = [None]

    # LBFGS-specific hyperparameters (only used when "LBFGS" is in optimizers_standard)
    # Separate LR list for LBFGS since it typically needs much larger LRs than Adam/SGD
    output['lrs_lbfgs'] = listify(cfg.get("lrs_lbfgs", output['lrs_standard']))
    output['lbfgs_max_iter'] = listify(cfg.get("lbfgs_max_iter", 20))
    output['lbfgs_history_size'] = listify(cfg.get("lbfgs_history_size", 100))
    output['lbfgs_line_search_fn'] = listify(cfg.get("lbfgs_line_search_fn", "strong_wolfe"))

    # Weight decay sweep — default [0.0] so existing configs are unaffected.
    # Non-zero values are only applied to AdamW in the scan loop.
    output['weight_decays'] = listify(cfg.get("weight_decays", [0.0]))

    return output

def _compute_acc(ypred, yb):
    """Compute mean accuracy, handling both (B, C) and (num_models, B, C) outputs."""
    if ypred.dim() == 3:
        preds = torch.argmax(ypred, dim=2)  # (M, B)
        acc = (preds == yb.unsqueeze(0)).float().mean().item()
    else:
        preds = torch.argmax(ypred, dim=1)
        acc = (preds == yb).float().mean().item()
    return acc

def _compute_per_model_acc(ypred, yb):
    """Compute per-model accuracy for (M, B, C) predictions. Returns list of M floats."""
    preds = torch.argmax(ypred, dim=2)  # (M, B)
    return (preds == yb.unsqueeze(0)).float().mean(dim=1).tolist()

def _is_closure_optimizer(optimizer):
    """Check if an optimizer requires a closure (e.g. LBFGS)."""
    return isinstance(optimizer, torch.optim.LBFGS)


def train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False, track_param_norm=False) -> tuple[Any, dict[str,Any]]:
    losses = defaultdict(list)
    is_multi = None  # detected on first forward pass
    uses_closure = _is_closure_optimizer(optimizer)

    print("Using device {}".format(device))

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
            losses['val_init'].append(loss.item())
            if track_acc:
                    losses['val_init_acc'].append(_compute_acc(ypred, yb))
            if is_multi is None:
                is_multi = ypred.dim() == 3
                if is_multi:
                    num_models = ypred.shape[0]
                    losses['num_models'] = num_models
    losses['val'].append(np.mean(losses['val_init']))
    del losses['val_init']
    if track_acc:
        losses['val_acc'].append(np.mean(losses['val_init_acc']))
        del losses['val_init_acc']

    total_start_time = time.perf_counter()

    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.perf_counter()
        epoch_losses = defaultdict(list)
        epoch_pm = defaultdict(list)  # per-model metrics for this epoch
        model.train()
        for xb, yb in train_loader:
            batch_start_time = time.perf_counter()
            xb, yb = xb.to(device), yb.to(device)

            if uses_closure:
                # LBFGS requires a closure that re-evaluates the model
                closure_loss = [None]
                closure_ypred = [None]
                def closure():
                    optimizer.zero_grad()
                    ypred = model(xb)
                    loss = loss_fn(ypred, yb)
                    loss.backward()
                    closure_loss[0] = loss
                    closure_ypred[0] = ypred
                    return loss
                optimizer.step(closure)
                loss = closure_loss[0]
                ypred = closure_ypred[0]
            else:
                optimizer.zero_grad()
                ypred = model(xb)
                loss = loss_fn(ypred, yb)
                loss.backward()
                optimizer.step()

            batch_end_time = time.perf_counter()
            losses['batch_times_train'].append(batch_end_time - batch_start_time)
            epoch_losses['train'].append(loss.item())
            if is_multi:
                with torch.no_grad():
                    pm_losses = [loss_fn(ypred[i], yb).item() for i in range(num_models)]
                epoch_pm['train'].append(pm_losses)
                losses['train_batch_per_model'].append(pm_losses)
            if track_acc:
                epoch_losses['train_acc'].append(_compute_acc(ypred, yb))
                if is_multi:
                    epoch_pm['train_acc'].append(_compute_per_model_acc(ypred.detach(), yb))
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                ypred = model(xb)
                loss = loss_fn(ypred, yb)
                batch_end_time = time.perf_counter()
                losses['batch_times_val'].append(batch_end_time - batch_start_time)
                epoch_losses['val'].append(loss.item())
                if is_multi:
                    pm_losses = [loss_fn(ypred[i], yb).item() for i in range(num_models)]
                    epoch_pm['val'].append(pm_losses)
                if track_acc:
                    epoch_losses['val_acc'].append(_compute_acc(ypred, yb))
                    if is_multi:
                        epoch_pm['val_acc'].append(_compute_per_model_acc(ypred, yb))
        epoch_end_time = time.perf_counter()
        losses['epoch_times'].append(epoch_end_time - epoch_start_time)
        # Save batch-wise losses
        losses['train_batch'].extend(epoch_losses['train'])
        losses['val_batch'].extend(epoch_losses['val'])
        for k,v in epoch_losses.items():
            losses[k].append(np.mean(v))
        # Save per-model epoch averages (each is a list of M values)
        for k,v in epoch_pm.items():
            losses[f'{k}_per_model'].append(np.mean(v, axis=0).tolist())
        if track_param_norm:
            with torch.no_grad():
                pnorm = torch.cat([p.detach().flatten() for p in model.parameters()]).norm().item()
            losses['param_norm'].append(pnorm)

    total_end_time = time.perf_counter()
    losses: dict[str,Any] = dict(losses) # making type checker happy
    losses['total_time'] = total_end_time - total_start_time
    losses['avg_epoch_time'] = np.mean(losses['epoch_times'])
    losses['avg_batch_time_train'] = np.mean(losses['batch_times_train'])
    losses['avg_batch_time_val'] = np.mean(losses['batch_times_val'])

    torch.cuda.empty_cache()

    return model, losses

def train_loop_svd(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False, track_param_norm=False) -> tuple[Any, dict[str,Any], Any]:
    losses = defaultdict(list)
    is_multi = None  # detected on first forward pass

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model.evaluate(xb)
            loss = loss_fn(ypred, yb).mean()
            losses['val_init'].append(loss.item())
            if track_acc:
                    losses['val_init_acc'].append(_compute_acc(ypred, yb))
            if is_multi is None:
                is_multi = ypred.dim() == 3
                if is_multi:
                    num_models = ypred.shape[0]
                    losses['num_models'] = num_models
    losses['val'].append(np.mean(losses['val_init']))
    del losses['val_init']
    if track_acc:
        losses['val_acc'].append(np.mean(losses['val_init_acc']))
        del losses['val_init_acc']

    total_start_time = time.perf_counter()

    # Ensure all computations are done without gradients
    with torch.no_grad():
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            epoch_losses = defaultdict(list)
            epoch_pm = defaultdict(list)  # per-model metrics for this epoch
            for xb, yb in train_loader:
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                batch = (xb, yb)
                batch_losses, ypred = model.loss_and_grad(batch)
                optimizer.step(batch)
                batch_end_time = time.perf_counter()
                losses['batch_times_train'].append(batch_end_time - batch_start_time)
                epoch_losses['train'].append(batch_losses.mean().item())
                if is_multi:
                    pm_losses = batch_losses.reshape(num_models, -1).mean(dim=1).tolist()
                    epoch_pm['train'].append(pm_losses)
                    losses['train_batch_per_model'].append(pm_losses)
                if track_acc:
                    epoch_losses['train_acc'].append(_compute_acc(ypred, yb))
                    if is_multi:
                        epoch_pm['train_acc'].append(_compute_per_model_acc(ypred, yb))

            for xb, yb in val_loader:
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device).detach(), yb.to(device).detach()
                ypred = model.evaluate(xb)
                per_sample_loss = loss_fn(ypred, yb)
                loss = per_sample_loss.mean()
                batch_end_time = time.perf_counter()
                losses['batch_times_val'].append(batch_end_time - batch_start_time)
                epoch_losses['val'].append(loss.item())
                if is_multi:
                    pm_losses = per_sample_loss.reshape(num_models, -1).mean(dim=1).tolist()
                    epoch_pm['val'].append(pm_losses)
                if track_acc:
                    epoch_losses['val_acc'].append(_compute_acc(ypred, yb))
                    if is_multi:
                        epoch_pm['val_acc'].append(_compute_per_model_acc(ypred, yb))

            epoch_end_time = time.perf_counter()
            losses['epoch_times'].append(epoch_end_time - epoch_start_time)
            # Save batch-wise losses
            losses['train_batch'].extend(epoch_losses['train'])
            losses['val_batch'].extend(epoch_losses['val'])
            # Save epoch-averaged losses
            for k_name, v in epoch_losses.items():
                losses[k_name].append(np.mean(v))
            # Save per-model epoch averages (each is a list of M values)
            for k_name, v in epoch_pm.items():
                losses[f'{k_name}_per_model'].append(np.mean(v, axis=0).tolist())
            if track_param_norm:
                losses['param_norm'].append(model.params.norm().item())

    total_end_time = time.perf_counter()
    losses: dict[str,Any] = dict(losses) # making type checker happy
    losses['total_time'] = total_end_time - total_start_time
    losses['avg_epoch_time'] = np.mean(losses['epoch_times'])
    losses['avg_batch_time_train'] = np.mean(losses['batch_times_train'])
    losses['avg_batch_time_val'] = np.mean(losses['batch_times_val'])

    torch.cuda.empty_cache()

    return model, losses, optimizer

def build_standard_optimizer(model, optim_name, lr, **kwargs):
    """Construct a standard PyTorch optimizer by name."""
    if optim_name == "LBFGS":
        # LBFGS has specific parameters; filter out irrelevant kwargs
        lbfgs_kwargs = {
            k: kwargs[k] for k in ("max_iter", "history_size", "line_search_fn")
            if k in kwargs
        }
        return torch.optim.LBFGS(model.parameters(), lr=lr, **lbfgs_kwargs)
    return getattr(torch.optim, optim_name)(model.parameters(), lr=lr, **kwargs)
