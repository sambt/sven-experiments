"""CIFAR ResNet18 Sven step-time probe vs chunked-capture grouping (batch-stat BN).
python bench/cifar_chunk_probe.py --chunk-numel 4194304 --steps 40 [--capture hooks]
"""
import argparse, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import torch
from torch.utils.data import DataLoader
from experiments.experiment_code.generic_scan import SVD_LOSS_FNS
from experiments.experiment_code.experiment_utils import set_seed
from experiments.nn import resnet18_functional
from experiments.datasets import CIFAR10Dataset
from sven.nn import GramSvenWrapper
from sven.opt import SvenGram
ap = argparse.ArgumentParser(); ap.add_argument("--chunk-numel", type=int, default=2**22); ap.add_argument("--capture", default="chunked")
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--k", type=int, default=64); ap.add_argument("--lr", type=float, default=1.0); ap.add_argument("--rtol", type=float, default=1e-3); ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"; set_seed(4000)
ds = CIFAR10Dataset(n_train=128 * (a.steps + 5)); model = resnet18_functional(num_classes=10)
tl = DataLoader(ds.train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(4000))
frozen = a.capture == "hooks"
w = GramSvenWrapper(model, SVD_LOSS_FNS["label_regression"], dev, capture=a.capture, freeze_norm_stats=frozen, chunk_numel=a.chunk_numel)
opt = SvenGram(w, lr=a.lr, k=a.k, rtol=a.rtol)
n_groups = len(w._param_groups()) if a.capture == "chunked" else None
torch.cuda.reset_peak_memory_stats(); times = []; losses = []
for i, (xb, yb) in enumerate(tl):
    if i >= a.steps: break
    xb, yb = xb.to(dev), yb.to(dev); torch.cuda.synchronize(); t0 = time.perf_counter()
    L, _ = w.loss_and_grad((xb, yb)); opt.step((xb, yb)); torch.cuda.synchronize(); times.append(time.perf_counter() - t0); losses.append(L.mean().item())
warm = times[5:]
res = {"capture": a.capture, "chunk_numel": a.chunk_numel, "n_groups": n_groups, "k": a.k, "steps": len(times),
       "step_ms_median": 1e3 * sorted(warm)[len(warm)//2], "step_ms_mean": 1e3 * sum(warm)/len(warm),
       "peak_mem_gb": torch.cuda.max_memory_allocated()/1e9, "loss_first": losses[0], "loss_last": losses[-1], "gpu": torch.cuda.get_device_name()}
print("RESULT " + json.dumps(res)); sys.stdout.flush()
if a.out:
    open(a.out, "a").write(json.dumps(res) + "\n")
