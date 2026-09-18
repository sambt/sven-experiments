"""Intra-GPU sharding benchmark: does running NPROC training processes on one
GPU raise throughput (runs/hour) versus one process at a time?

    python bench/bench_sharding.py --workload mnist --nproc 4 --n-runs 8 --epochs 2 --threads 1 --out x.json

The master launches NPROC workers (this same file with --worker), each taking
runs i with i % NPROC == shard, alternating Sven-Gram and Adam runs so both
kinds of load are represented. Wall time is measured from launch to the last
worker's exit (startup + dataset load included, as in the real launcher); each
worker also reports pure training time per run.
"""
import argparse, json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker(a):
    sys.path.insert(0, REPO)
    if a.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import torch
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    from torch.utils.data import DataLoader
    from experiments.experiment_code.generic_scan import SVD_LOSS_FNS, STANDARD_LOSS_FNS, SVD_RESIDUAL_FNS
    from experiments.experiment_code.experiment_utils import train_loop_svd, train_loop_standard, set_seed
    from experiments.nn import MLP, resnet18_functional
    from experiments.datasets import MNISTDataset, Toy1DRegressionDataset, CIFAR10Dataset
    from sven.nn import GramSvenWrapper
    from sven.opt import SvenGram
    dev = "cuda"
    t0 = time.perf_counter()
    if a.workload == "mnist":
        ds = MNISTDataset(); mk = lambda: MLP(784, [32, 32, 32], 10, "gelu"); loss = "label_regression"; bs, k, lr, cap = 64, 32, 0.5, "hooks"
    elif a.workload == "toy1d":
        ds = Toy1DRegressionDataset(); mk = lambda: MLP(1, [16, 16, 16], 1, "gelu"); loss = "mse"; bs, k, lr, cap = 32, 16, 0.5, "hooks"
    elif a.workload == "cifar":
        ds = CIFAR10Dataset(n_train=a.cifar_steps * 128); mk = lambda: resnet18_functional(num_classes=10); loss = "label_regression"; bs, k, lr, cap = 128, 128, 0.1, "chunked"
    else:
        raise ValueError(a.workload)
    load_s = time.perf_counter() - t0
    val_ds = torch.utils.data.Subset(ds.val_dataset, range(min(len(ds.val_dataset), 2048)))
    out = {"shard": a.shard, "load_s": load_s, "runs": []}
    for i in range(a.n_runs):
        if i % a.nproc != a.shard:
            continue
        kind = "sven" if i % 2 == 0 else "adam"
        set_seed(1000 + i); model = mk()
        tl = DataLoader(ds.train_dataset, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(1))
        vl = DataLoader(val_ds, batch_size=bs)
        t1 = time.perf_counter()
        if kind == "sven":
            w = GramSvenWrapper(model, SVD_LOSS_FNS[loss], dev, capture=cap,
                                residual_fn=SVD_RESIDUAL_FNS.get(loss))
            opt = SvenGram(w, lr=lr, k=k, rtol=1e-3, track_svd_info=True)
            _, L, _ = train_loop_svd(w, opt, SVD_LOSS_FNS[loss], tl, vl, a.epochs, dev, track_acc=(loss != "mse"))
        else:
            model = model.to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            _, L = train_loop_standard(model, opt, STANDARD_LOSS_FNS[loss], tl, vl, a.epochs, dev, track_acc=(loss != "mse"))
        torch.cuda.synchronize()
        out["runs"].append({"i": i, "kind": kind, "train_s": time.perf_counter() - t1,
                            "n_batches": len(L["train_batch"]) if "train_batch" in L else None,
                            "avg_batch_s": L["avg_batch_time_train"], "final_val": L["val"][-1],
                            "peak_mem_mb": L.get("peak_gpu_mem_mb")})
        del model, opt; torch.cuda.empty_cache()
    print(json.dumps(out)); sys.stdout.flush()


def master(a):
    t0 = time.perf_counter()
    procs = []
    for s in range(a.nproc):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--shard", str(s)] + sys.argv[1:]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO,
                                      env={**os.environ, "PYTHONPATH": REPO}))
    shards, errs = [], []
    for p in procs:
        so, se = p.communicate()
        if p.returncode != 0:
            errs.append(se[-2000:])
        else:
            shards.append(json.loads(so.strip().splitlines()[-1]))
    wall = time.perf_counter() - t0
    runs = [r for s in shards for r in s["runs"]]
    res = {"workload": a.workload, "nproc": a.nproc, "threads": a.threads, "n_runs": a.n_runs, "epochs": a.epochs,
           "wall_s": wall, "runs_per_hour": 3600 * len(runs) / wall if runs else 0.0, "n_done": len(runs),
           "load_s_max": max((s["load_s"] for s in shards), default=None),
           "sven_train_s_mean": _mean([r["train_s"] for r in runs if r["kind"] == "sven"]),
           "adam_train_s_mean": _mean([r["train_s"] for r in runs if r["kind"] == "adam"]),
           "sven_batch_ms": _mean([1e3 * r["avg_batch_s"] for r in runs if r["kind"] == "sven"]),
           "adam_batch_ms": _mean([1e3 * r["avg_batch_s"] for r in runs if r["kind"] == "adam"]),
           "peak_mem_mb": max((r["peak_mem_mb"] or 0 for r in runs), default=None),
           "errors": errs}
    print("RESULT " + json.dumps(res)); sys.stdout.flush()
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


def _mean(x):
    return sum(x) / len(x) if x else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="mnist", choices=["mnist", "toy1d", "cifar"])
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="OMP/torch threads per worker; 0 = torch default")
    ap.add_argument("--cifar-steps", type=int, default=40)
    ap.add_argument("--out", default="")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    worker(a) if a.worker else master(a)
