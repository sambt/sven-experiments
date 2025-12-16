import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device):
    losses = defaultdict(list)
    loss_fn = nn.MSELoss()

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
            losses['val_init'].append(loss.item())

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
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                ypred = model(xb)
                loss = loss_fn(ypred, yb)
                epoch_losses['val'].append(loss.item())
        # Save batch-wise losses
        losses['train_batch'].extend(epoch_losses['train'])
        losses['val_batch'].extend(epoch_losses['val'])
        for k,v in epoch_losses.items():
            losses[k].append(np.mean(v))
    
    return model, losses

def train_loop_svd_mse(model, optimizer, train_loader, val_loader, num_epochs, device):
    losses = defaultdict(list)
    def loss_fn(pred,y):
        loss = (pred-y)**2
        loss = loss.sum(dim=-1) # shape (B,)
        return loss

    # save untrained validation loss
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ypred = model.evaluate(xb)
            loss = loss_fn(ypred, yb).mean()
            losses['val_init'].append(loss.item())

    # Ensure all computations are done without gradients
    with torch.no_grad():
        for epoch in tqdm(range(num_epochs)):
            epoch_losses = defaultdict(list)
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                batch = (xb, yb)
                batch_losses = model.loss_and_grad(batch)
                optimizer.step()
                epoch_losses['train'].append(batch_losses.mean().item())
            
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
    