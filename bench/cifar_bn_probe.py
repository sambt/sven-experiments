"""CIFAR-10 ResNet18 Sven probe: why did accuracy collapse on the Gram backend?

Modes (all 3 epochs, full CIFAR train, val on the test set each epoch):
  hooks_frozen      GramSvenWrapper(capture=hooks, freeze_norm_stats=True)   -- the current scan path
  chunked_frozen    GramSvenWrapper(capture=chunked, freeze_norm_stats=True)
  chunked_batch     GramSvenWrapper(capture=chunked, freeze_norm_stats=False) -- BN uses batch statistics, like classic
  classic           SvenWrapper + Sven(svd_mode=randomized_v2)               -- the pre-Gram (paper) pipeline
  adam              torch.optim.Adam reference
"""
import argparse, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import torch
from torch.utils.data import DataLoader
from experiments.experiment_code.generic_scan import SVD_LOSS_FNS, STANDARD_LOSS_FNS
from experiments.experiment_code.experiment_utils import train_loop_svd, train_loop_standard, set_seed
from experiments.nn import resnet18_functional
from experiments.datasets import CIFAR10Dataset
from sven.nn import GramSvenWrapper, SvenWrapper
from sven.opt import SvenGram, Sven

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True); ap.add_argument("--k", type=int, default=128); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--rtol", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--loss", default="label_regression")
ap.add_argument("--out", default=""); a = ap.parse_args()
dev = "cuda"; set_seed(4000)
ds = CIFAR10Dataset(); model = resnet18_functional(num_classes=10)
tl = DataLoader(ds.train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(4000))
vl = DataLoader(ds.val_dataset, batch_size=128)
t0 = time.perf_counter()
if a.mode == "adam":
    model = model.to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    _, L = train_loop_standard(model, opt, STANDARD_LOSS_FNS[a.loss], tl, vl, a.epochs, dev, track_acc=True)
elif a.mode == "classic":
    w = SvenWrapper(model, SVD_LOSS_FNS[a.loss], dev)
    opt = Sven(w, lr=a.lr, k=a.k, rtol=a.rtol, svd_mode="randomized_v2", track_svd_info=True)
    _, L, _ = train_loop_svd(w, opt, SVD_LOSS_FNS[a.loss], tl, vl, a.epochs, dev, track_acc=True)
else:
    capture, frozen = {"hooks_frozen": ("hooks", True), "chunked_frozen": ("chunked", True), "chunked_batch": ("chunked", False)}[a.mode]
    w = GramSvenWrapper(model, SVD_LOSS_FNS[a.loss], dev, capture=capture, freeze_norm_stats=frozen)
    opt = SvenGram(w, lr=a.lr, k=a.k, rtol=a.rtol, track_svd_info=True)
    _, L, _ = train_loop_svd(w, opt, SVD_LOSS_FNS[a.loss], tl, vl, a.epochs, dev, track_acc=True)
res = {"mode": a.mode, "k": a.k, "lr": a.lr, "rtol": a.rtol, "loss": a.loss, "epochs": a.epochs, "wall_s": time.perf_counter() - t0,
       "train": [round(v, 4) for v in L["train"]], "val": [round(v, 4) for v in L["val"]], "val_acc": [round(v, 4) for v in L["val_acc"]],
       "peak_mem_mb": L.get("peak_gpu_mem_mb"), "step_ms": 1e3 * L["avg_batch_time_train"]}
print("RESULT " + json.dumps(res)); sys.stdout.flush()
if a.out:
    with open(a.out, "a") as f: f.write(json.dumps(res) + "\n")
