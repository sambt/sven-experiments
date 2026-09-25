"""CPU-only tests for the dataset layer (C-D1, C-D2, C-D3, C-E1).

No downloads: every test builds its own synthetic data, so MNIST / CIFAR-10 /
tiny-shakespeare image and text files are never needed.  The index-split logic of the
image datasets is tested through the pure functions they call.
"""
import importlib.util
import inspect
import json
import os
from itertools import product

import numpy as np
import pytest
import torch

from experiments.datasets import legacy_additive_powers as legacy_additive_powers_export
from experiments.datasets.all_datasets import (
    AdditiveCubicDataset,
    CIFAR10Dataset,
    MNISTDataset,
    RandomPolynomialDataset,
    Toy1DRegressionDataset,
    CharTextDataset,
    TokenBinDataset,
    contiguous_split_bounds,
    eval_polynomial,
    holdout_split_indices,
    legacy_additive_powers,
    monomial_powers,
    monomial_rms,
    subsample_indices,
)

DEGREE, NUM_VARS = 4, 6
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOT = 50256


def _load_module(relpath, name):
    """Import a script that is not part of a package (tools/, experiments/data_prep/)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# C-D1: monomial enumeration and evaluation
# ---------------------------------------------------------------------------

def test_monomial_enumeration_is_complete():
    """All 210 monomials of total degree <= 4 in 6 variables, constant and linear included."""
    powers = monomial_powers(DEGREE, NUM_VARS)
    assert len(powers) == 210
    assert len(set(powers)) == 210
    # exactly the set of exponent tuples with total degree <= DEGREE
    brute = {p for p in product(range(DEGREE + 1), repeat=NUM_VARS) if sum(p) <= DEGREE}
    assert set(powers) == brute
    assert (0,) * NUM_VARS in powers                                  # constant
    for j in range(NUM_VARS):                                         # linear
        assert tuple(1 if i == j else 0 for i in range(NUM_VARS)) in powers
    assert (DEGREE,) + (0,) * (NUM_VARS - 1) in powers                # x0**4
    # C(num_vars + degree, degree) for a few small cases
    from math import comb
    for d, v in ((0, 3), (1, 3), (2, 2), (3, 4), (4, 6), (5, 2)):
        assert len(monomial_powers(d, v)) == comb(v + d, d)


def test_known_monomials_evaluate_correctly():
    """eval_polynomial multiplies the per-variable factors (it used to add them)."""
    x = np.array([[2.0, 3.0], [-1.0, 0.5], [0.0, 4.0]])
    # 3 * x0*x1  -  2 * x0**2  +  5
    y = eval_polynomial(x, [3.0, -2.0, 5.0], [(1, 1), (2, 0), (0, 0)])
    expected = 3 * x[:, 0] * x[:, 1] - 2 * x[:, 0] ** 2 + 5
    assert np.allclose(y, expected, atol=0, rtol=1e-15)
    # a single pure monomial x0**2 * x1**3
    y = eval_polynomial(x, [1.0], [(2, 3)])
    assert np.allclose(y, x[:, 0] ** 2 * x[:, 1] ** 3, atol=0, rtol=1e-15)
    # the constant monomial is a constant, not a function of x
    assert np.allclose(eval_polynomial(x, [7.0], [(0, 0)]), 7.0)


def test_monomial_rms_matches_gaussian_moments():
    """sqrt(E[m^2]) = sqrt(prod_j (2 p_j - 1)!!) for x ~ N(0, I)."""
    assert monomial_rms((0, 0)) == 1.0                    # constant
    assert monomial_rms((1, 0)) == 1.0                    # E[x^2] = 1
    assert monomial_rms((2, 0)) == pytest.approx(3.0 ** 0.5)      # E[x^4] = 3
    assert monomial_rms((3, 0)) == pytest.approx(15.0 ** 0.5)     # E[x^6] = 15
    assert monomial_rms((4, 0)) == pytest.approx(105.0 ** 0.5)    # E[x^8] = 105
    assert monomial_rms((2, 2)) == pytest.approx(9.0 ** 0.5)      # 3 * 3
    assert monomial_rms((1, 1, 1)) == 1.0
    # Monte-Carlo: a normalised monomial has unit second moment
    x = np.random.default_rng(0).normal(size=(400_000, 3))
    for p in [(2, 0, 0), (2, 2, 0), (1, 1, 1)]:
        m = eval_polynomial(x, [1.0 / monomial_rms(p)], [p])
        assert np.mean(m ** 2) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# C-D1: the additive 19-feature basis fits the old target and not the new one
# ---------------------------------------------------------------------------

def _additive_design(x):
    """[1, x_j, x_j**2, x_j**3] for every variable: the 19 features that fit an
    additive cubic in 6 variables exactly."""
    cols = [np.ones(x.shape[0])]
    for j in range(x.shape[1]):
        cols += [x[:, j], x[:, j] ** 2, x[:, j] ** 3]
    return np.stack(cols, axis=1)


def _additive_relative_rmse(dataset):
    x, y = dataset.train_dataset.tensors
    x = x.double().numpy()
    y = y.double().numpy().ravel()
    design = _additive_design(x)
    assert design.shape[1] == 19
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(y ** 2)))


def test_new_polynomial_is_not_additive_cubic():
    """The variance-normalised monomial target is NOT fit by the 19 additive features."""
    ds = RandomPolynomialDataset(DEGREE, NUM_VARS, seed=0, n_train=4000, n_val=64, n_test=64,
                                 pool_size=4000)
    assert ds.num_terms == 210
    assert _additive_relative_rmse(ds) > 0.1


def test_additive_cubic_is_still_additive_cubic():
    """The legacy generator collapses to 19 effective features and is fit to < 1e-6."""
    ds = AdditiveCubicDataset(DEGREE, NUM_VARS, seed=0, n_train=4000, n_val=64, n_test=64)
    assert ds.num_terms == 185
    assert len(legacy_additive_powers(DEGREE, NUM_VARS)) == 185
    assert _additive_relative_rmse(ds) < 1e-6


def test_additive_cubic_reproduces_legacy_targets_bit_for_bit():
    """Same seed -> byte-identical train/val targets as the pre-2026-09 class."""
    seed, n_train, n_val = 3, 500, 400

    # verbatim copy of the pre-2026-09 RandomPolynomialDataset body
    rng = np.random.default_rng(seed)
    power_combinations = []
    for d in range(DEGREE + 1):
        for powers in product(range(d), repeat=NUM_VARS):
            if sum(powers) == d:
                power_combinations.append(powers)
    coeffs = rng.normal(size=(len(power_combinations),))
    x_train = rng.normal(size=(n_train, NUM_VARS))
    x_val = rng.normal(size=(n_val, NUM_VARS))
    y_train = np.zeros(n_train)
    y_val = np.zeros(n_val)
    for coeff, power in zip(coeffs, power_combinations):
        term_train = np.ones(n_train)
        term_val = np.ones(n_val)
        for j in range(NUM_VARS):
            term_train += x_train[:, j] ** power[j]
            term_val += x_val[:, j] ** power[j]
        y_train += coeff * term_train
        y_val += coeff * term_val
    mean, std = np.mean(y_train), np.std(y_train)
    y_train = (y_train - mean) / std
    y_val = (y_val - mean) / std

    ds = AdditiveCubicDataset(DEGREE, NUM_VARS, seed=seed, n_train=n_train, n_val=n_val, n_test=7)
    assert np.array_equal(ds.coeffs, coeffs)
    assert list(ds.power_combinations) == power_combinations
    for got, want_x, want_y in ((ds.train_dataset, x_train, y_train),
                                (ds.val_dataset, x_val, y_val)):
        gx, gy = got.tensors
        assert torch.equal(gx, torch.tensor(want_x, dtype=torch.float32))
        assert torch.equal(gy, torch.tensor(want_y, dtype=torch.float32).unsqueeze(1))


# ---------------------------------------------------------------------------
# C-D3: synthetic val/test and target scale do not move with n_train
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda n: Toy1DRegressionDataset(n_train=n, n_val=256, n_test=256, seed=1),
    lambda n: RandomPolynomialDataset(DEGREE, NUM_VARS, seed=1, n_train=n,
                                      n_val=256, n_test=256),
])
def test_val_and_test_are_identical_across_n_train(factory):
    small, big = factory(150), factory(1200)
    assert (small.n_train, big.n_train) == (150, 1200)
    for split in ("val_dataset", "test_dataset"):
        for a, b in zip(getattr(small, split).tensors, getattr(big, split).tensors):
            assert torch.equal(a, b)
    # the pool is fixed, so the target scale is too, and the draws are nested
    sx, sy = small.train_dataset.tensors
    bx, by = big.train_dataset.tensors
    assert torch.equal(sx, bx[:150]) and torch.equal(sy, by[:150])
    # val and test come from separate generators: they are not the same data
    assert not torch.equal(small.val_dataset.tensors[0], small.test_dataset.tensors[0])


def test_synthetic_datasets_record_sizes_and_split_seed():
    toy = Toy1DRegressionDataset(n_train=150, n_val=17, n_test=23, seed=5)
    assert (toy.n_train, toy.n_val, toy.n_test) == (150, 17, 23)
    assert toy.split_seed == 5
    poly = RandomPolynomialDataset(DEGREE, NUM_VARS, seed=5, n_train=150, n_val=17, n_test=23)
    assert (poly.n_train, poly.n_val, poly.n_test) == (150, 17, 23)
    assert poly.split_seed == 5


# ---------------------------------------------------------------------------
# C-E1: the image-dataset index split (pure, no image files needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,n_val", [(60_000, 10_000), (50_000, 5_000), (11, 3)])
def test_holdout_split_is_disjoint_and_covering(n, n_val):
    tr, va = holdout_split_indices(n, n_val, split_seed=1234)
    assert len(va) == n_val and len(tr) == n - n_val
    assert set(tr.tolist()).isdisjoint(va.tolist())
    assert sorted(tr.tolist() + va.tolist()) == list(range(n))


def test_holdout_split_depends_only_on_split_seed():
    a = holdout_split_indices(1000, 100, split_seed=1234)
    b = holdout_split_indices(1000, 100, split_seed=1234)
    c = holdout_split_indices(1000, 100, split_seed=4321)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])   # deterministic
    assert not torch.equal(a[1], c[1])                           # and seed-dependent
    # the global RNG is neither read nor advanced (C-S1: no position dependence)
    torch.manual_seed(0)
    before = torch.rand(4)
    torch.manual_seed(0)
    holdout_split_indices(1000, 100, split_seed=1234)
    assert torch.equal(before, torch.rand(4))


def test_split_is_independent_of_subsample_seed():
    """Changing the subsample seed changes which train examples are used, never the
    held-out set, and the subsample always stays inside the train part."""
    tr, va = holdout_split_indices(1000, 100, split_seed=1234)
    val_ids = set(va.tolist())
    picked = []
    for subsample_seed in (0, 1, 7):
        tr2, va2 = holdout_split_indices(1000, 100, split_seed=1234)
        assert torch.equal(tr, tr2) and torch.equal(va, va2)
        idx = subsample_indices(len(tr), 200, subsample_seed)
        chosen = tr[idx]
        assert len(chosen) == 200
        assert set(chosen.tolist()).isdisjoint(val_ids)
        picked.append(chosen.tolist())
    assert picked[0] != picked[1] != picked[2]


def test_subsample_is_nested_and_deterministic():
    small = subsample_indices(1000, 150, subsample_seed=3)
    assert torch.equal(small, subsample_indices(1000, 150, subsample_seed=3))
    mid = subsample_indices(1000, 400, subsample_seed=3)
    assert torch.equal(small, mid[:150])                   # nested draws
    assert torch.equal(subsample_indices(1000, None, 3), torch.arange(1000))
    assert torch.equal(subsample_indices(1000, 1000, 3), torch.arange(1000))   # exactly the pool


def test_n_train_above_the_pool_raises():
    """Scope update 2026-09-18: never clamp silently -- a clamp would stamp n_data=60000
    into a record trained on the 50,000-example MNIST train part."""
    with pytest.raises(ValueError, match="outside the available training pool of 50000"):
        subsample_indices(50_000, 60_000, subsample_seed=0)
    with pytest.raises(ValueError):                        # 0 examples is a bug too
        subsample_indices(1000, 0, subsample_seed=0)
    with pytest.raises(ValueError, match="pool_size"):
        Toy1DRegressionDataset(n_train=50_000, n_val=8, n_test=8, seed=0, pool_size=1000)
    with pytest.raises(ValueError, match="pool_size"):
        RandomPolynomialDataset(DEGREE, NUM_VARS, seed=0, n_train=2000, n_val=8, n_test=8,
                                pool_size=1000)


def test_image_dataset_subsample_path_raises_above_the_pool(tmp_path):
    """The `_subsample` helper the MNIST / CIFAR classes use raises as well (exercised
    through CharTextDataset so no image files are needed)."""
    (tmp_path / "input.txt").write_text("abcdefgh" * 500)
    ds = CharTextDataset(ROOT=str(tmp_path), block_size=16)
    assert ds.n_train == 199
    with pytest.raises(ValueError, match="outside the available training pool of 199"):
        CharTextDataset(ROOT=str(tmp_path), block_size=16, n_train=500)
    ok = CharTextDataset(ROOT=str(tmp_path), block_size=16, n_train=199)   # exactly the pool
    assert ok.n_train == 199


# ---------------------------------------------------------------------------
# EXPERIMENTS.md section 12 "Datasets": split_seed is a uniform constructor argument
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Toy1DRegressionDataset, MNISTDataset, CIFAR10Dataset,
                                 RandomPolynomialDataset, AdditiveCubicDataset,
                                 CharTextDataset, TokenBinDataset])
def test_every_dataset_class_accepts_split_seed(cls):
    """`split_seed` lands in the dataset config of any scan, so Hydra instantiate must
    not raise `unexpected keyword argument` for any of the classes."""
    params = inspect.signature(cls.__init__).parameters
    assert "split_seed" in params
    assert params["split_seed"].default is None or params["split_seed"].default == 1234


def test_split_seed_defaults_to_seed_and_reseeds_the_synthetic_splits():
    ref = RandomPolynomialDataset(DEGREE, NUM_VARS, seed=7, n_train=64, n_val=32, n_test=32)
    same = RandomPolynomialDataset(DEGREE, NUM_VARS, seed=7, n_train=64, n_val=32, n_test=32,
                                   split_seed=7)
    other = RandomPolynomialDataset(DEGREE, NUM_VARS, seed=7, n_train=64, n_val=32, n_test=32,
                                    split_seed=99)
    assert (ref.split_seed, same.split_seed, other.split_seed) == (7, 7, 99)
    # passing split_seed = seed explicitly is a no-op (all current configs)
    for split in ("train_dataset", "val_dataset", "test_dataset"):
        for a, b in zip(getattr(ref, split).tensors, getattr(same, split).tensors):
            assert torch.equal(a, b)
    # a different split_seed moves the data but NOT the target function
    assert np.array_equal(ref.coeffs, other.coeffs)
    assert not torch.equal(ref.val_dataset.tensors[0], other.val_dataset.tensors[0])

    toy = Toy1DRegressionDataset(n_train=64, n_val=32, n_test=32, seed=7)
    toy_same = Toy1DRegressionDataset(n_train=64, n_val=32, n_test=32, seed=7, split_seed=7)
    toy_other = Toy1DRegressionDataset(n_train=64, n_val=32, n_test=32, seed=7, split_seed=99)
    assert toy_other.split_seed == 99
    assert torch.equal(toy.val_dataset.tensors[0], toy_same.val_dataset.tensors[0])
    assert not torch.equal(toy.val_dataset.tensors[0], toy_other.val_dataset.tensors[0])


def test_legacy_additive_powers_is_exported_from_the_package():
    assert legacy_additive_powers_export is legacy_additive_powers


# ---------------------------------------------------------------------------
# C-E1: contiguous character-corpus split
# ---------------------------------------------------------------------------

def test_contiguous_split_bounds_are_disjoint_and_cover():
    for n in (1_115_394, 1000, 13):
        (a0, a1), (b0, b1), (c0, c1) = contiguous_split_bounds(n)
        assert a0 == 0 and a1 == b0 and b1 == c0 and c1 == n
        assert a1 - a0 > 0 and b1 - b0 >= 0 and c1 - c0 >= 0
    (a0, a1), (b0, b1), (c0, c1) = contiguous_split_bounds(1000)
    assert (a1 - a0, b1 - b0, c1 - c0) == (800, 100, 100)


def test_shakespeare_style_contiguous_split_has_no_overlap(tmp_path):
    """A synthetic corpus: the three splits are the three contiguous slices, in order,
    with no shared character position."""
    rng = np.random.default_rng(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz \n"
    text = "".join(rng.choice(list(alphabet), size=4000))
    root = tmp_path / "shakespeare"
    root.mkdir()
    (root / "input.txt").write_text(text)

    block_size = 16
    ds = CharTextDataset(ROOT=str(root), block_size=block_size)
    ids = torch.tensor([ds.stoi[c] for c in text], dtype=torch.long)
    (tr0, tr1), (va0, va1), (te0, te1) = ds.bounds
    assert (tr1 - tr0, va1 - va0, te1 - te0) == (3200, 400, 400)

    # every block equals the corresponding slice of its own segment ...
    for split, (s0, s1) in ((ds.train_dataset, (tr0, tr1)),
                            (ds.val_dataset, (va0, va1)),
                            (ds.test_dataset, (te0, te1))):
        x, y = split.tensors
        assert len(x) == (s1 - s0 - 1) // block_size
        for i in range(len(x)):
            start = s0 + i * block_size
            assert torch.equal(x[i], ids[start:start + block_size])
            assert torch.equal(y[i], ids[start + 1:start + block_size + 1])

    # ... so the character positions the three splits touch are pairwise disjoint
    def positions(n_blocks, s0):
        return set(range(s0, s0 + n_blocks * block_size + 1))
    p_tr = positions(len(ds.train_dataset), tr0)
    p_va = positions(len(ds.val_dataset), va0)
    p_te = positions(len(ds.test_dataset), te0)
    assert p_tr.isdisjoint(p_va) and p_tr.isdisjoint(p_te) and p_va.isdisjoint(p_te)
    assert (ds.n_train, ds.n_val, ds.n_test) == (199, 24, 24)
    assert ds.split_seed is None


# ---------------------------------------------------------------------------
# C-D2: token bins and the overlap checker
# ---------------------------------------------------------------------------

def _write_bins(directory, train, val, test):
    for name, arr in (("train", train), ("val", val), ("test", test)):
        np.asarray(arr, dtype=np.uint16).tofile(os.path.join(directory, name + ".bin"))
    with open(os.path.join(directory, "token_counts.json"), "w") as f:
        json.dump({"tokens": {"train": len(train), "val": len(val), "test": len(test)}}, f)


def _check_tool():
    return _load_module(os.path.join("tools", "check_token_split.py"), "check_token_split")


def _prepare_tokens():
    return _load_module(os.path.join("experiments", "data_prep", "prepare_tokens.py"),
                        "prepare_tokens")


def _docs_to_tokens(docs):
    """Concatenate documents (lists of token ids) with the eot separator, as the .bin
    writer does."""
    out = []
    for d in docs:
        out.extend(list(d) + [EOT])
    return np.array(out, dtype=np.uint16)


def test_token_bin_dataset_reads_three_bins(tmp_path):
    rng = np.random.default_rng(0)
    _write_bins(str(tmp_path), rng.integers(0, 50256, 4001), rng.integers(0, 50256, 1001),
                rng.integers(0, 50256, 501))
    ds = TokenBinDataset(ROOT=str(tmp_path), block_size=100, n_train_blocks=None,
                         val_blocks=5, test_blocks=None)
    assert (ds.n_train, ds.n_val, ds.n_test) == (40, 5, 5)
    assert ds.token_counts == {"train": 4001, "val": 1001, "test": 501}
    x, y = ds.test_dataset[0]
    assert x.shape == y.shape == (100,) and torch.equal(x[1:], y[:-1])


def test_check_token_split_accepts_disjoint_and_rejects_overlap(tmp_path):
    tool = _check_tool()
    rng = np.random.default_rng(1)
    train, val, test = (rng.integers(0, 50256, n) for n in (20_000, 3_000, 3_000))

    good = tmp_path / "good"; good.mkdir()
    _write_bins(str(good), train, val, test)
    splits = tool.load_splits(str(good))
    assert set(splits) == {"train", "val", "test"}
    assert not tool.is_prefix(splits["train"], splits["val"])
    assert tool.find_shared_blocks(splits["val"], splits["train"], 8, 64) == []

    bad = tmp_path / "bad"; bad.mkdir()
    _write_bins(str(bad), train, train[:3_000], test)      # the F4 bug: val is a prefix
    bsplits = tool.load_splits(str(bad))
    assert tool.is_prefix(bsplits["train"], bsplits["val"])
    assert tool.find_shared_blocks(bsplits["val"], bsplits["train"], 8, 64)

    mid = tmp_path / "mid"; mid.mkdir()
    # not a prefix, but the val tokens are lifted out of the middle of train
    _write_bins(str(mid), train, train[7_000:10_000], test)
    msplits = tool.load_splits(str(mid))
    assert not tool.is_prefix(msplits["train"], msplits["val"])
    assert tool.find_shared_blocks(msplits["val"], msplits["train"], 8, 64)


def test_token_bin_dataset_rejects_a_file_too_short_for_one_block(tmp_path):
    """A truncated / empty .bin used to give a negative dataset length."""
    rng = np.random.default_rng(0)
    _write_bins(str(tmp_path), rng.integers(0, 50256, 4001), rng.integers(0, 50256, 1001),
                rng.integers(0, 50256, 40))
    with pytest.raises(ValueError, match="too short for one block"):
        TokenBinDataset(ROOT=str(tmp_path), block_size=100, val_blocks=5)


# ---------------------------------------------------------------------------
# C-D2 / F4: the document-level disjointness check, exhaustive where the block
# sample is not (32 blocks of 256 tokens cover ~0.1% of an 8M-token val split)
# ---------------------------------------------------------------------------

def _rand_docs(rng, n_docs, doc_len, base):
    """`n_docs` documents of distinct token ids, so "shares a document" is unambiguous."""
    return [list(rng.integers(base, base + 1000, doc_len) % 50_000) for _ in range(n_docs)]


def test_document_hashes_find_a_single_shared_document(tmp_path):
    tool = _check_tool()
    rng = np.random.default_rng(2)
    train_docs = _rand_docs(rng, 40, 300, 0)
    val_docs = _rand_docs(rng, 6, 300, 10_000)
    test_docs = _rand_docs(rng, 6, 300, 20_000)

    good = tmp_path / "good"; good.mkdir()
    _write_bins(str(good), _docs_to_tokens(train_docs), _docs_to_tokens(val_docs),
                _docs_to_tokens(test_docs))
    hashes = {k: tool.document_hashes(v, EOT)
              for k, v in tool.load_splits(str(good)).items()}
    assert (len(hashes["train"]), len(hashes["val"]), len(hashes["test"])) == (40, 6, 6)
    assert hashes["train"].isdisjoint(hashes["val"])
    assert hashes["val"].isdisjoint(hashes["test"])

    # ONE of val's six documents is also in the middle of train: not a prefix, and only
    # 1/40 of train, which a sparse block sample can miss -- the hash sets cannot.
    bad = tmp_path / "bad"; bad.mkdir()
    contaminated = val_docs[:3] + [train_docs[17]] + val_docs[3:]
    _write_bins(str(bad), _docs_to_tokens(train_docs), _docs_to_tokens(contaminated),
                _docs_to_tokens(test_docs))
    bsplits = tool.load_splits(str(bad))
    assert not tool.is_prefix(bsplits["train"], bsplits["val"])
    shared = (tool.document_hashes(bsplits["train"], EOT)
              & tool.document_hashes(bsplits["val"], EOT))
    assert len(shared) == 1


def test_check_tool_exit_codes(tmp_path, monkeypatch):
    """The tool itself: 0 on disjoint files, 1 on a shared document, 1 on a zero tail."""
    tool = _check_tool()
    rng = np.random.default_rng(3)
    train_docs = _rand_docs(rng, 30, 200, 0)
    val_docs = _rand_docs(rng, 5, 200, 10_000)
    test_docs = _rand_docs(rng, 5, 200, 20_000)

    def run(directory):
        monkeypatch.setattr("sys.argv", ["check_token_split.py", str(directory)])
        return tool.main()

    good = tmp_path / "good"; good.mkdir()
    _write_bins(str(good), _docs_to_tokens(train_docs), _docs_to_tokens(val_docs),
                _docs_to_tokens(test_docs))
    assert run(good) == 0

    bad = tmp_path / "bad"; bad.mkdir()
    _write_bins(str(bad), _docs_to_tokens(train_docs),
                _docs_to_tokens(val_docs[:2] + [train_docs[9]] + val_docs[2:]),
                _docs_to_tokens(test_docs))
    assert run(bad) == 1

    # an unfilled tail of zeros (the file was not truncated to the tokens written)
    tail = tmp_path / "tail"; tail.mkdir()
    _write_bins(str(tail), np.concatenate([_docs_to_tokens(train_docs),
                                           np.zeros(2000, dtype=np.uint16)]),
                _docs_to_tokens(val_docs), _docs_to_tokens(test_docs))
    assert run(tail) == 1


def test_check_tool_tolerates_residual_corpus_duplicates(tmp_path, monkeypatch):
    """FineWeb-edu contains exact duplicate documents far apart in the stream, so a
    *handful* of shared documents is a corpus property, not a split bug (measured:
    3/7,729 val docs, 0.04%).  Systematic sharing still fails."""
    tool = _check_tool()
    assert tool.shared_doc_fraction(3, 242_742, 7_729) == pytest.approx(3 / 7_729)
    assert tool.shared_doc_fraction(0, 0, 0) == 0.0

    rng = np.random.default_rng(4)
    train_docs = _rand_docs(rng, 800, 20, 0)
    val_docs = _rand_docs(rng, 600, 20, 10_000)
    test_docs = _rand_docs(rng, 600, 20, 20_000)

    def run(directory):
        monkeypatch.setattr("sys.argv", ["check_token_split.py", str(directory)])
        return tool.main()

    dup = tmp_path / "dup"; dup.mkdir()
    _write_bins(str(dup), _docs_to_tokens(train_docs),
                _docs_to_tokens(val_docs[:-1] + [train_docs[500]]),   # 1/600 = 0.17%
                _docs_to_tokens(test_docs))
    assert run(dup) == 0

    contaminated = tmp_path / "bad"; contaminated.mkdir()
    _write_bins(str(contaminated), _docs_to_tokens(train_docs),
                _docs_to_tokens(train_docs[100:700]),                 # 600/600 = 100%
                _docs_to_tokens(test_docs))
    assert run(contaminated) == 1


# ---------------------------------------------------------------------------
# C-D2 / F4: prepare_tokens.write_split consumes ONE shared iterator
# ---------------------------------------------------------------------------

def _fake_encode(text):
    """'1 2 3' -> [1, 2, 3]; a list, because write_split appends the eot in place."""
    return [int(t) for t in text.split()]


def _fake_docs(n_docs, doc_len):
    """Document i holds the tokens 1000*(i+1) .. 1000*(i+1)+doc_len-1, so which document
    a token came from is visible in the file."""
    return [" ".join(str(1000 * (i + 1) + j) for j in range(doc_len)) for i in range(n_docs)]


def test_write_split_draws_the_three_splits_from_disjoint_documents(tmp_path):
    """The F4 fix: one shared iterator, so no document reaches two splits, and each file
    is exactly the tokens written."""
    pt = _prepare_tokens()
    texts = iter(_fake_docs(6, 10))          # 6 documents, 11 tokens each with the eot
    counts, tokens = {}, {}
    for name, budget in (("val", 15), ("test", 15), ("train", 1000)):
        path = str(tmp_path / f"{name}.bin")
        counts[name] = pt.write_split(texts, path, budget, _fake_encode, EOT)
        assert os.path.getsize(path) == counts[name] * 2      # truncated, no zero tail
        tokens[name] = np.fromfile(path, dtype=np.uint16)
        assert len(tokens[name]) == counts[name]

    assert counts == {"val": 15, "test": 15, "train": 22}     # stream ran dry on train
    # the documents each split saw are pairwise disjoint (distinct token ranges)
    def docs_of(arr):
        return {int(t) // 1000 for t in arr if t != EOT}
    assert docs_of(tokens["val"]) == {1, 2}       # doc 0 whole + the head of doc 1
    assert docs_of(tokens["test"]) == {3, 4}      # doc 1's tail was discarded, not reused
    assert docs_of(tokens["train"]) == {5, 6}
    for a, b in (("val", "test"), ("val", "train"), ("test", "train")):
        assert docs_of(tokens[a]).isdisjoint(docs_of(tokens[b]))
    # and the first split really is not a prefix of the last (the v1 bug)
    assert not np.array_equal(tokens["train"][:15], tokens["val"])


def test_write_split_short_stream_and_empty_stream(tmp_path):
    pt = _prepare_tokens()
    texts = iter(_fake_docs(2, 5))
    n = pt.write_split(texts, str(tmp_path / "val.bin"), 100, _fake_encode, EOT)
    assert n == 12 and os.path.getsize(str(tmp_path / "val.bin")) == 24   # short but exact
    # the iterator is now exhausted: the next split would silently get 0 tokens
    with pytest.raises(ValueError, match="ran dry"):
        pt.write_split(texts, str(tmp_path / "test.bin"), 100, _fake_encode, EOT)
    with pytest.raises(ValueError, match="budget"):
        pt.write_split(iter(_fake_docs(1, 5)), str(tmp_path / "x.bin"), 0, _fake_encode, EOT)


def test_prepare_tokens_refuses_to_overwrite_existing_bins(tmp_path, monkeypatch):
    """The guard runs before any HF / tiktoken import, so the verified directories cannot
    be clobbered by re-running the documented command."""
    pt = _prepare_tokens()
    assert pt.existing_bins(str(tmp_path)) == []
    np.zeros(4, dtype=np.uint16).tofile(str(tmp_path / "train.bin"))
    assert pt.existing_bins(str(tmp_path)) == ["train"]
    monkeypatch.setattr("sys.argv", ["prepare_tokens.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        pt.main()
    assert "refusing to overwrite" in str(exc.value)
    assert os.path.getsize(str(tmp_path / "train.bin")) == 8      # untouched
