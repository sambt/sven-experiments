"""CPU-only tests for the standard-optimizer factory (C-B2, C-B5, C-B6; F9/F13/F14).

No downloads and no GPU: every model is built from ``experiments.nn`` (the two big
ones -- torchvision ResNet18 and GPT-2 small -- on the ``meta`` device, since only
parameter names and shapes matter for the grouping rule).  The Muon/MuonConv
equivalence tests run in float64 so that everything outside the (bfloat16)
Newton-Schulz iteration is exact.
"""
import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.experiment_code import optim_factory
from experiments.experiment_code.optim_factory import (
    MUON_ADJUST_LR_FN,
    MUON_RULE_TOKEN,
    MUON_RULE_VERSION,
    MUON_VARIANT_CONV_FLAT,
    MUON_VARIANT_HIDDEN2D,
    MUON_VARIANT_LEGACY,
    _CUSTOM_OPTIMIZERS,
    _CombinedOptimizer,
    _DEFAULT_WEIGHT_DECAY,
    build_standard_optimizer,
    get_muon_variant,
    muon_param_groups,
    resolve_weight_decay,
)
from experiments.nn.nets import (
    MLP,
    MultiMLP,
    NanoGPT,
    SmallCNN,
    SmallResNet,
    resnet18_functional,
)
from experiments.optimizers.muon_conv import MuonConv

UTILS_PATH = Path(__file__).resolve().parents[1] / "experiments/experiment_code/experiment_utils.py"
FACTORY_PATH = Path(optim_factory.__file__)
# torchvision's ImageNet ResNet18 checkpoint, if this machine already has it (TORCH_HOME
# on the cluster does).  Tests never download, they skip.
_RESNET18_WEIGHTS = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"


def _names(groups):
    return {key: [name for name, _ in params] for key, params in groups.items()}


# ---------------------------------------------------------------------------
# C-B2: SGDm
# ---------------------------------------------------------------------------

def test_sgdm_is_sgd_with_momentum_and_plain_sgd_is_unchanged():
    model = MLP(4, [8], 2)
    sgdm = build_standard_optimizer(model, "SGDm", 1e-2, weight_decay=None)
    assert type(sgdm) is torch.optim.SGD
    assert sgdm.param_groups[0]["momentum"] == 0.9
    assert sgdm.param_groups[0]["lr"] == 1e-2
    assert sgdm.param_groups[0]["weight_decay"] == 0.0     # no default wd for SGDm
    assert len(sgdm.param_groups[0]["params"]) == len(list(model.parameters()))

    sgd = build_standard_optimizer(model, "SGD", 1e-2, weight_decay=None)
    assert type(sgd) is torch.optim.SGD
    assert sgd.param_groups[0]["momentum"] == 0            # plain SGD untouched

    # an explicit momentum in the config still wins
    other = build_standard_optimizer(model, "SGDm", 1e-2, momentum=0.5)
    assert other.param_groups[0]["momentum"] == 0.5


# ---------------------------------------------------------------------------
# C-B6 / F14: the two local schedule-free classes are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ScheduleFreeAdamW", "ScheduleFreeSGD"])
def test_schedule_free_removed_from_registry(name):
    assert name not in _CUSTOM_OPTIMIZERS
    with pytest.raises(ValueError) as exc:
        build_standard_optimizer(MLP(4, [8], 2), name, 1e-3)
    message = str(exc.value)
    assert name in message and "schedulefree" in message and "C-B6" in message


def test_surviving_custom_optimizers():
    assert set(_CUSTOM_OPTIMIZERS) == {"Lion", "SOAP"}


# ---------------------------------------------------------------------------
# F9: weight-decay resolution is exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,given,expected", [
    ("AdamW", None, 0.01), ("MuonW", None, 0.1), ("Muon", None, 0.0),
    ("Adam", None, 0.0), ("SGDm", None, 0.0), ("AdamW", 0.0, 0.0),
    ("AdamW", 0.1, 0.1), ("MuonW", 0.0, 0.0),
])
def test_resolve_weight_decay(name, given, expected):
    assert resolve_weight_decay(name, given) == expected


def test_default_weight_decay_table_unchanged():
    assert _DEFAULT_WEIGHT_DECAY == {"AdamW": 0.01, "MuonW": 0.1}


def test_adamw_and_muon_weight_decays_as_built():
    model = MLP(4, [8, 8, 8], 2)
    assert build_standard_optimizer(
        model, "AdamW", 1e-3, weight_decay=None).param_groups[0]["weight_decay"] == 0.01
    assert build_standard_optimizer(
        model, "Adam", 1e-3, weight_decay=None).param_groups[0]["weight_decay"] == 0.0

    # Muon: wd as given (0 by default) on BOTH halves; MuonW: 0.1 on Muon, 0.01 on AdamW
    muon, muonw = (build_standard_optimizer(model, n, 1e-3, weight_decay=None)
                   for n in ("Muon", "MuonW"))
    assert [type(o).__name__ for o in muon.optimizers] == ["Muon", "AdamW"]
    assert [pg["weight_decay"] for pg in muon.param_groups] == [0.0, 0.0]
    assert [pg["weight_decay"] for pg in muonw.param_groups] == [0.1, 0.01]
    swept = build_standard_optimizer(model, "Muon", 1e-3, weight_decay=0.1)
    assert [pg["weight_decay"] for pg in swept.param_groups] == [0.1, 0.1]


# ---------------------------------------------------------------------------
# C-B5 / F13: the grouping rule on the repo's actual models
# ---------------------------------------------------------------------------

def test_groups_mlp():
    # four nn.Linear: net.6 is the output head (AdamW), the other three -- including the
    # input layer net.0, see muon_param_groups rule 3 -- are hidden matrices for Muon
    groups = _names(muon_param_groups(MLP(4, [8, 8, 8], 3)))
    assert groups["muon_2d"] == ["net.0.weight", "net.2.weight", "net.4.weight"]
    assert groups["muon_conv"] == []
    assert groups["adamw"] == ["net.0.bias", "net.2.bias", "net.4.bias",
                               "net.6.weight", "net.6.bias"]


@pytest.mark.parametrize("width,expected_frac", [(32, 0.9845), (128, 0.9876)])
def test_muon_actually_covers_the_mnist_headline_model(width, expected_frac):
    """F13's substance: "Muon" must not be AdamW on most of the model.  With the input
    layer excluded, MLP(784, [32]*3, 10) would give Muon 7.4% of the parameters (the
    784x32 input matrix alone is 91% of the model)."""
    model = MLP(784, [width] * 3, 10)
    groups = muon_param_groups(model)
    muon = sum(p.numel() for key in ("muon_2d", "muon_conv") for _, p in groups[key])
    total = sum(p.numel() for p in model.parameters())
    assert muon / total == pytest.approx(expected_frac, abs=5e-4)
    assert muon / total > 0.98
    assert "net.0.weight" in [n for n, _ in groups["muon_2d"]]


def test_groups_nanogpt():
    model = NanoGPT(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=8)
    groups = _names(muon_param_groups(model))
    # every per-block projection is hidden; only the embeddings and lm_head are AdamW's
    assert groups["muon_2d"] == [
        f"blocks.{i}.{leaf}.weight" for i in range(2)
        for leaf in ("attn.q", "attn.k", "attn.v", "attn.proj", "mlp.0", "mlp.2")
    ]
    assert groups["muon_conv"] == []
    # F13's three offenders are now AdamW's, and nothing else 2-D is left over
    for name in ("tok_emb.weight", "pos_emb.weight", "lm_head.weight"):
        assert name in groups["adamw"]
    assert all(p.ndim == 1 or n in ("tok_emb.weight", "pos_emb.weight", "lm_head.weight")
               for n, p in muon_param_groups(model)["adamw"])


def test_groups_gpt2_small_on_meta_device():
    with torch.device("meta"):
        model = NanoGPT(vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    groups = _names(muon_param_groups(model))
    assert len(groups["muon_2d"]) == 12 * 6                  # 4 attn + 2 mlp per block
    assert all(n.startswith("blocks.") for n in groups["muon_2d"])
    assert {"tok_emb.weight", "pos_emb.weight", "lm_head.weight"} <= set(groups["adamw"])


def test_groups_resnet18():
    with torch.device("meta"):
        model = resnet18_functional(num_classes=10)
    groups = muon_param_groups(model)
    names = _names(groups)
    # all 20 convs (the stem, the blocks and the 1x1 downsample convs) are hidden
    # kernels and reach the vendored MuonConv; only fc is the head
    assert len(names["muon_conv"]) == 20
    assert names["muon_conv"][0] == "conv1.weight"
    assert names["muon_2d"] == []                            # fc is the head
    assert [n for n, p in groups["adamw"] if p.ndim > 1] == ["fc.weight"]
    muon = sum(p.numel() for _, p in groups["muon_conv"])
    assert muon / sum(p.numel() for p in model.parameters()) > 0.99


@pytest.mark.skipif(not _RESNET18_WEIGHTS.exists(),
                    reason=f"no cached ImageNet weights at {_RESNET18_WEIGHTS} "
                           "(tests never download)")
def test_groups_resnet18_pretrained():
    """The model `exp_finetune_cifar_smallN.yaml` actually runs Muon/MuonW on: a fresh
    `model.fc` is assigned after construction, and `replace_batchnorm` swaps modules --
    check that neither disturbs the registration order the head rule relies on."""
    from experiments.nn.nets import resnet18_pretrained_functional
    groups = muon_param_groups(resnet18_pretrained_functional(num_classes=10))
    names = _names(groups)
    assert len(names["muon_conv"]) == 20 and names["muon_2d"] == []
    assert [n for n, p in groups["adamw"] if p.ndim > 1] == ["fc.weight"]


def test_groups_small_convnets():
    cnn = _names(muon_param_groups(SmallCNN()))
    assert cnn["muon_conv"] == ["conv1.weight", "conv2.weight", "conv3.weight"]
    assert cnn["muon_2d"] == ["fc1.weight"]                       # fc2 is the head
    assert "conv1.weight" not in cnn["adamw"] and "fc2.weight" in cnn["adamw"]

    small = _names(muon_param_groups(SmallResNet(width=4, num_blocks=1)))
    assert small["muon_2d"] == []
    assert small["muon_conv"][0] == "conv1.weight"
    assert all(n.startswith("layer") and n.endswith("weight")
               for n in small["muon_conv"][1:])
    assert "conv1.weight" not in small["adamw"] and "fc.weight" in small["adamw"]


def test_multilinear_stacks_go_to_adamw():
    # MultiLinear's (num_models, out, in) weight is a STACK of independent matrices:
    # flattening it would orthogonalise across ensemble members, so it stays with AdamW
    # (and Muon then has nothing hidden to work on -- the factory must say so).
    model = MultiMLP(2, 4, [8, 8], 3)
    groups = muon_param_groups(model)
    assert groups["muon_2d"] == [] and groups["muon_conv"] == []
    assert len(groups["adamw"]) == len(list(model.parameters()))
    with pytest.raises(ValueError, match="no hidden weight matrix"):
        build_standard_optimizer(model, "Muon", 1e-3)


def test_single_layer_model_refuses_muon_but_two_layers_are_enough():
    # one Linear = the head, nothing hidden; two Linears = one hidden matrix (the rule
    # excludes only the head, so the smallest model Muon accepts has two projections)
    with pytest.raises(ValueError, match="no hidden weight matrix"):
        build_standard_optimizer(nn.Sequential(nn.Linear(4, 3)), "MuonW", 1e-3)
    groups = _names(muon_param_groups(MLP(4, [8], 3)))
    assert groups["muon_2d"] == ["net.0.weight"] and groups["muon_conv"] == []


_MODEL_BUILDERS = {
    "mlp": lambda: MLP(4, [8, 8, 8], 3),
    "multimlp": lambda: MultiMLP(2, 4, [8, 8], 3),
    "nanogpt": lambda: NanoGPT(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=8),
    "nanogpt_tied": lambda: NanoGPT(vocab_size=65, block_size=16, n_layer=1, n_head=2,
                                    n_embd=8, tie_weights=True),
    "smallcnn": SmallCNN,
    "smallresnet": lambda: SmallResNet(width=4, num_blocks=1),
    "resnet18": lambda: resnet18_functional(num_classes=10),   # on meta, see below
}


@pytest.mark.parametrize("tag", list(_MODEL_BUILDERS))
def test_groups_partition_the_parameters(tag):
    with torch.device("meta") if tag == "resnet18" else torch.device("cpu"):
        model = _MODEL_BUILDERS[tag]()
    groups = muon_param_groups(model)
    ids = [id(p) for params in groups.values() for _, p in params]
    assert len(ids) == len(set(ids))                                  # disjoint
    assert set(ids) == {id(p) for p in model.parameters()}            # complete
    assert all(p.ndim == 2 for _, p in groups["muon_2d"])
    assert all(p.ndim > 2 for _, p in groups["muon_conv"])


# ---------------------------------------------------------------------------
# C-B5: the vendored MuonConv
# ---------------------------------------------------------------------------

def _fixed_grads(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=g, dtype=torch.float64) for s in shapes]


@pytest.mark.parametrize("shape", [(8, 8), (16, 4), (4, 16)])
@pytest.mark.parametrize("adjust_lr_fn", [MUON_ADJUST_LR_FN, "original", None])
def test_muonconv_reproduces_torch_muon_on_2d_params(shape, adjust_lr_fn):
    """Square, tall and wide 2-D parameters, fed identical gradients for 6 steps."""
    init = torch.randn(*shape, generator=torch.Generator().manual_seed(7),
                       dtype=torch.float64)
    p_ref, p_new = (nn.Parameter(init.clone()) for _ in range(2))
    kwargs = dict(lr=0.03, weight_decay=0.1, adjust_lr_fn=adjust_lr_fn)
    ref = torch.optim.Muon([p_ref], **kwargs)
    new = MuonConv([p_new], **kwargs)

    for step, grad in enumerate(_fixed_grads([shape] * 6, seed=3)):
        p_ref.grad, p_new.grad = grad.clone(), grad.clone()
        ref.step()
        new.step()
        assert (p_new - p_ref).abs().max().item() < 1e-6, f"step {step}"
    assert not torch.equal(p_new, init)        # the steps actually moved the parameter


@pytest.mark.parametrize("kernel", [(8, 4, 3, 3), (6, 6, 1, 1), (4, 8, 5, 5)])
def test_muonconv_on_conv_kernel_equals_muon_on_the_flattened_matrix(kernel):
    """The flattening contract: MuonConv on (out, in, kh, kw) == torch Muon on
    (out, in*kh*kw), including the lr adjustment, which must see the flat shape."""
    flat = (kernel[0], kernel[1] * kernel[2] * kernel[3])
    init = torch.randn(*kernel, generator=torch.Generator().manual_seed(11),
                       dtype=torch.float64)
    p_conv = nn.Parameter(init.clone())
    p_flat = nn.Parameter(init.reshape(flat).clone())
    kwargs = dict(lr=0.05, weight_decay=0.01, adjust_lr_fn=MUON_ADJUST_LR_FN)
    conv_opt = MuonConv([p_conv], **kwargs)
    flat_opt = torch.optim.Muon([p_flat], **kwargs)

    for step, grad in enumerate(_fixed_grads([kernel] * 4, seed=5)):
        p_conv.grad = grad.clone()
        p_flat.grad = grad.reshape(flat).clone()
        conv_opt.step()
        flat_opt.step()
        assert (p_conv.reshape(flat) - p_flat).abs().max().item() < 1e-6, f"step {step}"


def test_muonconv_rejects_1d_parameters():
    with pytest.raises(ValueError, match="ndim >= 2"):
        MuonConv([nn.Parameter(torch.zeros(5))], lr=1e-3)


def test_muonconv_validates_and_scalarises_a_tensor_lr():
    """`torch.optim.Muon` accepts a 1-element tensor lr (LR schedulers) and rejects a
    longer one; the vendored copy must behave identically, not fail later inside add_."""
    with pytest.raises(ValueError, match="Tensor lr must be 1-element"):
        MuonConv([nn.Parameter(torch.zeros(4, 4))], lr=torch.tensor([1e-3, 2e-3]))

    init = torch.randn(6, 4, generator=torch.Generator().manual_seed(2),
                       dtype=torch.float64)
    p_ref, p_new = (nn.Parameter(init.clone()) for _ in range(2))
    kwargs = dict(lr=torch.tensor([0.02], dtype=torch.float64), weight_decay=0.1,
                  adjust_lr_fn=MUON_ADJUST_LR_FN)
    ref, new = torch.optim.Muon([p_ref], **kwargs), MuonConv([p_new], **kwargs)
    for grad in _fixed_grads([(6, 4)] * 3, seed=9):
        p_ref.grad, p_new.grad = grad.clone(), grad.clone()
        ref.step()
        new.step()
    assert (p_new - p_ref).abs().max().item() < 1e-6
    assert not torch.equal(p_new, init)


@pytest.mark.parametrize("shape", [(16, 4), (4, 16), (8, 8)])
def test_legacy_adjust_lr_fn_none_is_the_original_rule(shape):
    """F13/C-B5 bookkeeping: torch treats `adjust_lr_fn=None` (what every legacy Muon run
    used) as "original" = sqrt(max(1, A/B)), NOT as an unadjusted lr -- so the effective
    lr changes by 0.2*sqrt(max(A,B)) / sqrt(max(1, A/B)) when we switch to
    `match_rms_adamw`, and `MUON_VARIANT_LEGACY` must not claim "none"."""
    A, B = shape
    init = torch.randn(*shape, generator=torch.Generator().manual_seed(4),
                       dtype=torch.float64)
    params, opts = {}, {}
    for fn in (None, "original", MUON_ADJUST_LR_FN):
        params[fn] = nn.Parameter(init.clone())
        opts[fn] = torch.optim.Muon([params[fn]], lr=0.01, weight_decay=0.0,
                                    adjust_lr_fn=fn)
    grad = _fixed_grads([shape], seed=13)[0]
    for fn, opt in opts.items():
        params[fn].grad = grad.clone()
        opt.step()

    assert torch.equal(params[None], params["original"])          # None IS "original"
    expected = 0.2 * (max(A, B) ** 0.5) / max(1.0, A / B) ** 0.5
    delta_orig, delta_match = init - params["original"], init - params[MUON_ADJUST_LR_FN]
    assert delta_orig.abs().max() > 0                             # both moved
    ratio = (delta_match / delta_orig).flatten()
    assert torch.allclose(ratio, torch.full_like(ratio, expected), rtol=1e-9)
    assert "original" in MUON_VARIANT_LEGACY and "none" not in MUON_VARIANT_LEGACY


# ---------------------------------------------------------------------------
# C-B5: muon_variant on the record, and one real step through the factory
# ---------------------------------------------------------------------------

def test_muon_variant_is_exposed():
    mlp, cnn = MLP(4, [8, 8, 8], 3), SmallCNN()
    pure = build_standard_optimizer(mlp, "Muon", 1e-3, weight_decay=None)
    conv = build_standard_optimizer(cnn, "MuonW", 1e-3, weight_decay=None)
    assert get_muon_variant(pure) == MUON_VARIANT_HIDDEN2D
    assert get_muon_variant(conv) == MUON_VARIANT_CONV_FLAT
    assert [type(o).__name__ for o in conv.optimizers] == ["Muon", "MuonConv", "AdamW"]
    # the principled shared lr (C-B5) is actually requested
    assert all(pg["adjust_lr_fn"] == MUON_ADJUST_LR_FN
               for o in conv.optimizers if isinstance(o, (torch.optim.Muon, MuonConv))
               for pg in o.param_groups)
    assert get_muon_variant(build_standard_optimizer(mlp, "AdamW", 1e-3)) is None


def test_variant_and_rule_token_carry_the_rule_version():
    """C-R3: the recorded variant / the `record_extra` token must change when the grouping
    rule or the lr adjustment changes, otherwise finished runs of the old construction are
    skipped as up to date and one scan mixes two Muon definitions."""
    for variant in (MUON_VARIANT_HIDDEN2D, MUON_VARIANT_CONV_FLAT, MUON_RULE_TOKEN):
        assert variant.endswith(f":{MUON_RULE_VERSION}")
        assert MUON_ADJUST_LR_FN in variant
    assert MUON_RULE_VERSION.startswith("v")
    assert MUON_RULE_TOKEN not in (MUON_VARIANT_HIDDEN2D, MUON_VARIANT_CONV_FLAT)


def test_muonw_uses_its_default_weight_decay_without_an_explicit_kwarg():
    """CONTRACTS.md: MuonW runs at wd 0.1 in every scan.  A call site that passes no
    `weight_decay` at all (generic_scan's jd inner optimizer, optimizer_profile) must not
    silently get 0.0 on the Muon half."""
    model = MLP(4, [8, 8, 8], 2)
    muonw = build_standard_optimizer(model, "MuonW", 1e-3)
    assert [pg["weight_decay"] for pg in muonw.param_groups] == [0.1, 0.01]
    muon = build_standard_optimizer(model, "Muon", 1e-3)
    assert [pg["weight_decay"] for pg in muon.param_groups] == [0.0, 0.0]
    explicit = build_standard_optimizer(model, "MuonW", 1e-3, weight_decay=0.0)
    assert [pg["weight_decay"] for pg in explicit.param_groups] == [0.0, 0.01]


def test_muonw_step_on_a_conv_net():
    torch.manual_seed(0)
    model = SmallCNN().eval()                 # eval() only to switch dropout off
    optimizer = build_standard_optimizer(model, "MuonW", 1e-3, weight_decay=None)
    assert isinstance(optimizer, _CombinedOptimizer)
    before = [p.detach().clone() for p in model.parameters()]

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    for _ in range(2):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


# ---------------------------------------------------------------------------
# The factory was COPIED out of experiment_utils.py; until the integrator replaces
# that region by an import, the two must agree on names and signatures.
# ---------------------------------------------------------------------------

def _function_args(path, name):
    tree = ast.parse(path.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    return None if fn is None else ast.dump(fn.args)


@pytest.mark.parametrize("name", ["build_standard_optimizer", "resolve_weight_decay"])
def test_signature_matches_experiment_utils(name):
    old = _function_args(UTILS_PATH, name)
    if old is None:
        pytest.skip(f"{name} has been removed from experiment_utils.py (integrated)")
    assert old == _function_args(FACTORY_PATH, name)


def _legacy_factory():
    """`experiment_utils.build_standard_optimizer`, or a skip: the module belongs to
    another track (training loops) and disappears from this test once the integrator
    replaces its factory region by an import of `optim_factory`."""
    try:
        from experiments.experiment_code import experiment_utils
    except Exception as exc:                                   # mid-edit / import error
        pytest.skip(f"experiment_utils.py is not importable here: {exc!r}")
    legacy = getattr(experiment_utils, "build_standard_optimizer", None)
    if legacy is None or legacy is build_standard_optimizer:
        pytest.skip("experiment_utils.py now re-exports the factory (integrated)")
    return legacy


# every name the campaign builds through this factory whose branch C-B2/C-B5/C-B6 do NOT
# change (SGDm, Muon, MuonW are deliberately different; Shampoo/KFAC need extra packages)
_UNCHANGED_NAMES = [
    ("Adam", dict(weight_decay=None)),
    ("AdamW", dict(weight_decay=None)),
    ("AdamW", dict(weight_decay=0.1)),
    ("SGD", dict(weight_decay=None)),
    ("RMSprop", dict(weight_decay=None)),
    ("Lion", {}),
    ("SOAP", {}),
    ("LBFGS", dict(max_iter=5, history_size=10, line_search_fn="strong_wolfe")),
    ("PolyakSGD", dict(f_star=0.0, max_lr=1.0)),
]


@pytest.mark.parametrize("name,kwargs", _UNCHANGED_NAMES,
                         ids=[f"{n}{sorted(k)}" for n, k in _UNCHANGED_NAMES])
def test_copied_branches_build_the_same_optimizer_as_experiment_utils(name, kwargs):
    """Copy fidelity of every branch the move must NOT change: same class and the same
    param_groups (minus the parameters themselves) from both factories."""
    legacy = _legacy_factory()
    old = legacy(MLP(4, [8, 8], 2), name, 1e-3, **dict(kwargs))
    new = build_standard_optimizer(MLP(4, [8, 8], 2), name, 1e-3, **dict(kwargs))
    assert type(old) is type(new)
    assert len(old.param_groups) == len(new.param_groups)
    for old_group, new_group in zip(old.param_groups, new.param_groups):
        assert len(old_group["params"]) == len(new_group["params"])
        assert {k: v for k, v in old_group.items() if k != "params"} == \
               {k: v for k, v in new_group.items() if k != "params"}
