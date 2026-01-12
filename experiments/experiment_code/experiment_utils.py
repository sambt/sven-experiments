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

    return output

def train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False) -> tuple[Any, dict[str,Any]]:
    losses = defaultdict(list)

    print("Using device {}".format(device))

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
            losses['val_init'].append(loss.item())
    losses['val'].append(np.mean(losses['val_init']))
    del losses['val_init']

    total_start_time = time.perf_counter()

    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.perf_counter()
        epoch_losses = defaultdict(list)
        model.train()
        for xb, yb in train_loader:
            batch_start_time = time.perf_counter()
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
            loss.backward()
            optimizer.step()
            batch_end_time = time.perf_counter()
            losses['batch_times_train'].append(batch_end_time - batch_start_time)
            epoch_losses['train'].append(loss.item())
            if track_acc:
                preds = torch.argmax(ypred, dim=1)
                acc = (preds == yb).float().mean().item()
                epoch_losses['train_acc'].append(acc)
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
                if track_acc:
                    preds = torch.argmax(ypred, dim=1)
                    acc = (preds == yb).float().mean().item()
                    epoch_losses['val_acc'].append(acc)
        epoch_end_time = time.perf_counter()
        losses['epoch_times'].append(epoch_end_time - epoch_start_time)
        # Save batch-wise losses
        losses['train_batch'].extend(epoch_losses['train'])
        losses['val_batch'].extend(epoch_losses['val'])
        for k,v in epoch_losses.items():
            losses[k].append(np.mean(v))

    total_end_time = time.perf_counter()
    losses: dict[str,Any] = dict(losses) # making type checker happy
    losses['total_time'] = total_end_time - total_start_time
    losses['avg_epoch_time'] = np.mean(losses['epoch_times'])
    losses['avg_batch_time_train'] = np.mean(losses['batch_times_train'])
    losses['avg_batch_time_val'] = np.mean(losses['batch_times_val'])

    torch.cuda.empty_cache()

    return model, losses

def train_loop_svd(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False) -> tuple[Any, dict[str,Any], Any]:
    losses = defaultdict(list)

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model.evaluate(xb)
            loss = loss_fn(ypred, yb).mean()
            losses['val_init'].append(loss.item())
    losses['val'].append(np.mean(losses['val_init']))
    del losses['val_init']

    total_start_time = time.perf_counter()

    # Ensure all computations are done without gradients
    with torch.no_grad():
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            epoch_losses = defaultdict(list)
            for xb, yb in train_loader:
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                batch = (xb, yb)
                batch_losses, ypred = model.loss_and_grad(batch)
                optimizer.step()
                batch_end_time = time.perf_counter()
                losses['batch_times_train'].append(batch_end_time - batch_start_time)
                epoch_losses['train'].append(batch_losses.mean().item())
                if track_acc:
                    preds = torch.argmax(ypred, dim=1)
                    acc = (preds == yb).float().mean().item()
                    epoch_losses['train_acc'].append(acc)

            for xb, yb in val_loader:
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device).detach(), yb.to(device).detach()
                ypred = model.evaluate(xb)
                loss = loss_fn(ypred, yb).mean()
                batch_end_time = time.perf_counter()
                losses['batch_times_val'].append(batch_end_time - batch_start_time)
                epoch_losses['val'].append(loss.item())
                if track_acc:
                    preds = torch.argmax(ypred, dim=1)
                    acc = (preds == yb).float().mean().item()
                    epoch_losses['val_acc'].append(acc)

            epoch_end_time = time.perf_counter()
            losses['epoch_times'].append(epoch_end_time - epoch_start_time)
            # Save batch-wise losses
            losses['train_batch'].extend(epoch_losses['train'])
            losses['val_batch'].extend(epoch_losses['val'])
            # Save epoch-averaged losses
            for k_name, v in epoch_losses.items():
                losses[k_name].append(np.mean(v))

    total_end_time = time.perf_counter()
    losses: dict[str,Any] = dict(losses) # making type checker happy
    losses['total_time'] = total_end_time - total_start_time
    losses['avg_epoch_time'] = np.mean(losses['epoch_times'])
    losses['avg_batch_time_train'] = np.mean(losses['batch_times_train'])
    losses['avg_batch_time_val'] = np.mean(losses['batch_times_val'])

    torch.cuda.empty_cache()

    return model, losses, optimizer
    