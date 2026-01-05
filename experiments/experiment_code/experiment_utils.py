import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Any
import os
import pandas as pd
import fcntl
import tempfile
import shutil
import time

def listify(settings) -> list[Any]:
    if isinstance(settings, list):
        return settings
    else:
        return [settings]

def add_row(row, file, max_retries=5, retry_delay=0.1):
    """
    Safely add a row to a pickle file with file locking to prevent corruption
    from concurrent writes.
    
    Args:
        row: Dictionary to add as a new row
        file: Path to the pickle file
        max_retries: Number of times to retry if lock acquisition fails
        retry_delay: Base delay between retries (exponential backoff)
    """
    lock_file = file + ".lock"
    
    for attempt in range(max_retries):
        try:
            # Create lock file if it doesn't exist
            with open(lock_file, 'w') as lock_fd:
                # Acquire exclusive lock (blocking)
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                try:
                    # Read existing data
                    if os.path.exists(file):
                        df = pd.read_pickle(file)
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    else:
                        df = pd.DataFrame([row])
                    
                    # Write to a temporary file first (atomic write pattern)
                    dir_name = os.path.dirname(file) or '.'
                    fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.pkl.tmp')
                    try:
                        os.close(fd)
                        df.to_pickle(temp_path)
                        # Atomic rename (on same filesystem)
                        shutil.move(temp_path, file)
                    except:
                        # Clean up temp file on failure
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise
                finally:
                    # Release lock
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            return  # Success
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to write to {file} after {max_retries} attempts: {e}")


def train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False):
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

    for epoch in tqdm(range(num_epochs)):
        epoch_losses = defaultdict(list)
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses['train'].append(loss.item())
            if track_acc:
                preds = torch.argmax(ypred, dim=1)
                acc = (preds == yb).float().mean().item()
                epoch_losses['train_acc'].append(acc)
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                ypred = model(xb)
                loss = loss_fn(ypred, yb)
                epoch_losses['val'].append(loss.item())
                if track_acc:
                    preds = torch.argmax(ypred, dim=1)
                    acc = (preds == yb).float().mean().item()
                    epoch_losses['val_acc'].append(acc)
        # Save batch-wise losses
        losses['train_batch'].extend(epoch_losses['train'])
        losses['val_batch'].extend(epoch_losses['val'])
        for k,v in epoch_losses.items():
            losses[k].append(np.mean(v))
    
    return model, losses

def train_loop_svd(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc=False):
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

    # Ensure all computations are done without gradients
    with torch.no_grad():
        for epoch in tqdm(range(num_epochs)):
            epoch_losses = defaultdict(list)
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                batch = (xb, yb)
                batch_losses, ypred = model.loss_and_grad(batch)
                optimizer.step()
                epoch_losses['train'].append(batch_losses.mean().item())
                if track_acc:
                    preds = torch.argmax(ypred, dim=1)
                    acc = (preds == yb).float().mean().item()
                    epoch_losses['train_acc'].append(acc)
            
            for xb, yb in val_loader:
                xb, yb = xb.to(device).detach(), yb.to(device).detach()
                ypred = model.evaluate(xb)
                loss = loss_fn(ypred, yb).mean()
                epoch_losses['val'].append(loss.item())
            
            # Save batch-wise losses
            losses['train_batch'].extend(epoch_losses['train'])
            losses['val_batch'].extend(epoch_losses['val'])
            # Save epoch-averaged losses
            for k_name, v in epoch_losses.items():
                losses[k_name].append(np.mean(v))
    
    return model, losses, optimizer
    