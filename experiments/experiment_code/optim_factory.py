"""The standard-optimizer factory: names from a config -> a torch optimizer (C-B2, C-B5, C-B6).

Moved here out of ``experiment_utils.py`` (which keeps the training loops) so the
baseline definitions live in one place.  Names and signatures are unchanged, so
``experiment_utils`` can simply re-export them:

    from experiments.experiment_code.optim_factory import (
        _CUSTOM_OPTIMIZERS, _CombinedOptimizer, _KFACOptimizer,
        _DEFAULT_WEIGHT_DECAY, resolve_weight_decay, build_standard_optimizer,
    )

Changes relative to the old in-place version:

* **C-B2** ``SGDm`` is a named baseline = ``torch.optim.SGD(momentum=0.9)``; plain
  ``SGD`` keeps falling through to ``torch.optim.SGD`` with momentum 0.
* **C-B5 / F13** ``Muon`` / ``MuonW`` now send only *hidden* weight matrices to Muon
  (see ``muon_param_groups``), use ``adjust_lr_fn="match_rms_adamw"`` so the single
  shared learning rate is principled, and handle conv kernels with the vendored
  ``MuonConv``.  The construction is reported as ``muon_variant`` on the record.
* **C-B6 / F14** the two local schedule-free classes are out of the registry; naming
  one in a config raises.
* Weight-decay resolution (``_DEFAULT_WEIGHT_DECAY`` / ``resolve_weight_decay``) and
  every other branch are byte-identical to the old version.  The one exception is
  inside the Muon branch: a call that passes **no** ``weight_decay`` at all now gets
  the optimizer's default (MuonW 0.1) instead of a bare 0.0, matching every other
  name.  Call sites that pass a resolved float are unaffected.
"""

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from sven.opt import PolyakSGD
from experiments.optimizers.baselines import Lion
from experiments.optimizers.muon_conv import MuonConv
from experiments.optimizers.soap import SOAP  # vendored zero-dep reference SOAP


_CUSTOM_OPTIMIZERS = {
    "Lion": Lion,
    "SOAP": SOAP,
}


# C-B6 / F14: the local ScheduleFreeAdamW / ScheduleFreeSGD differ from the reference
# algorithms (extra EMA / momentum) and no saved result uses them.  The reference
# `schedulefree` package is not installed, so the names are simply retired; a config
# that still lists one must fail loudly instead of silently running something else.
_REMOVED_OPTIMIZERS = {
    "ScheduleFreeAdamW": "schedulefree.AdamWScheduleFree",
    "ScheduleFreeSGD": "schedulefree.SGDScheduleFree",
}


class _KFACOptimizer:
    """Bundles a K-FAC preconditioner with a base optimizer so the standard
    training loop needs no changes.  K-FAC requires ``preconditioner.step()``
    to run AFTER ``loss.backward()`` and BEFORE the base ``optimizer.step()``;
    both are performed here inside a single ``.step()`` call.  The preconditioner
    installs its own forward/backward hooks on the model at construction time.
    """

    def __init__(self, base, preconditioner):
        self.base = base
        self.preconditioner = preconditioner

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # grads are already populated by loss.backward() in the training loop
        self.preconditioner.step()
        self.base.step()

    def state_dict(self):
        return {"base": self.base.state_dict()}

    def load_state_dict(self, sd):
        self.base.load_state_dict(sd["base"])


class _CombinedOptimizer:
    """Wraps two optimizers so they behave as one (zero_grad / step / state_dict)."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    # Expose a unified param_groups (needed by some LR schedulers / logging)
    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure=closure)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


# PyTorch defaults that a bare `weight_decay=0.0` used to override.  "Muon" keeps
# running at wd = 0 (its existing runs stay valid); "MuonW" is Muon at its PyTorch
# default wd = 0.1, the same split as Adam / AdamW.
_DEFAULT_WEIGHT_DECAY = {"AdamW": 0.01, "MuonW": 0.1}


def resolve_weight_decay(optim_name, weight_decay):
    """The weight decay a run actually uses: ``None`` means the optimizer's own default
    (AdamW: 0.01; everything else: 0.0), a number is taken as given."""
    if weight_decay is None:
        return _DEFAULT_WEIGHT_DECAY.get(optim_name, 0.0)
    return float(weight_decay)


# ---------------------------------------------------------------------------
# C-B5 / F13: which parameters Muon may touch
# ---------------------------------------------------------------------------

# Muon's single shared learning rate is only meaningful if the orthogonalised update
# is RMS-matched to the AdamW update; "match_rms_adamw" (Moonshot's rule, torch >= 2.9)
# scales it by 0.2*sqrt(max(A, B)).  The alternative in C-B5 -- sweeping the two lrs
# separately -- is not needed because this torch provides the adjustment.
MUON_ADJUST_LR_FN = "match_rms_adamw"

# Recorded on every Muon / MuonW result (C-B5 "record the variant in the result").
# The trailing token is the version of the grouping rule in `muon_param_groups`: bump it
# whenever the rule or `MUON_ADJUST_LR_FN` changes, so that the run hash changes with it
# and finished runs of the old construction are not silently treated as up to date (C-R3).
MUON_RULE_VERSION = "v1"
MUON_VARIANT_HIDDEN2D = f"hidden2d+adamw:match_rms_adamw:{MUON_RULE_VERSION}"
MUON_VARIANT_CONV_FLAT = f"hidden2d+convflat+adamw:match_rms_adamw:{MUON_RULE_VERSION}"
# Model-independent form of the same thing, for `record_extra` (and hence `run_hash`) of
# every Muon / MuonW spec: the grid does not know which groups a model ends up having.
MUON_RULE_TOKEN = f"hidden2d+convflat:match_rms_adamw:{MUON_RULE_VERSION}"
# The pre-campaign construction: every 2-D tensor incl. embeddings/heads to Muon and
# `adjust_lr_fn` never passed.  NOTE that torch treats `adjust_lr_fn=None` as the
# "original" rule (`_adjust_lr`, torch/optim/_muon.py: `sqrt(max(1, A/B))`), so the legacy
# runs were NOT unadjusted -- they used `original`, which is ~8x for GPT-2's head/embedding
# and 1x for every square matrix.  Kept as a name so legacy records can be relabelled.
MUON_VARIANT_LEGACY = "all2d+adamw:original"

_MUON_GROUP_KEYS = ("muon_2d", "muon_conv", "adamw")


def muon_param_groups(model):
    """Split ``model``'s parameters into the Muon / MuonConv / AdamW groups (C-B5).

    The rule, applied to the parameter-bearing modules in **registration order**
    (which is forward order for every model in ``experiments/nn``):

    1. ``nn.Embedding`` weights -> **AdamW** (they are not projection layers, so they
       fall through rule 4).  An embedding is a lookup table, not a linear map between
       feature spaces; orthogonalising a (vocab, d) table is what F13 objects to.
    2. Among the recognised projection layers (``nn.Linear`` and any ``nn.ConvNd``),
       the **last** one is the output head / final classifier -> **AdamW**.
    3. Every remaining projection weight is a hidden weight matrix: 2-D -> **Muon**,
       >2-D (conv kernels) -> **MuonConv** (flattened, see ``muon_conv``).  This
       **includes the input layer**: C-B5 sends only embeddings, the head and the 1-D
       parameters to AdamW, and excluding the input layer too would leave Muon with
       7% of the parameters of ``MLP(784, [32]*3, 10)`` (the MNIST headline model) --
       i.e. the F13 construction ("AdamW plus a couple of small matrices") under a
       different name.  Muon practice (torch's own docstring, modded-nanogpt) also
       keeps the first projection in Muon.
    4. Everything else -- all 1-D parameters (biases, LayerNorm / BatchNorm weights),
       and any >=2-D parameter that does not belong to a recognised projection layer
       -- goes to **AdamW**.  The one such case in this repo is ``MultiLinear``'s
       ``(num_models, out, in)`` weight: that is a *stack* of independent matrices,
       and flattening it would orthogonalise across ensemble members.

    Tied parameters (e.g. ``NanoGPT(tie_weights=True)``) are returned once, in the
    group of their first occurrence.

    Returns ``{"muon_2d": [(name, param), ...], "muon_conv": [...], "adamw": [...]}``
    -- names included so the grouping is inspectable and testable.
    """
    projections = []          # [(module name, module)] Linear / ConvNd, registration order
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, _ConvNd)):
            projections.append((mod_name, mod))

    # the output head / final classifier is the only projection AdamW owns
    head_name = projections[-1][0] if projections else None

    hidden = {}                             # id(weight) -> group key
    for mod_name, mod in projections:
        weight = getattr(mod, "weight", None)
        if weight is None or mod_name == head_name:
            continue
        hidden[id(weight)] = "muon_2d" if weight.ndim == 2 else "muon_conv"

    groups = {key: [] for key in _MUON_GROUP_KEYS}
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:               # tied weights show up twice
            continue
        seen.add(id(param))
        groups[hidden.get(id(param), "adamw")].append((name, param))
    return groups


def _build_muon(model, optim_name, lr, kwargs):
    """Muon / MuonW over the hidden weight matrices + AdamW over everything else.

    Muon:  wd as given (0 unless the config sweeps it) on both parts.
    MuonW: Muon at its PyTorch default wd (0.1) and AdamW at ITS default (0.01)
           for the remaining parameters -- "everything at its own default".
    """
    groups = muon_param_groups(model)
    muon_2d = [p for _, p in groups["muon_2d"]]
    muon_conv = [p for _, p in groups["muon_conv"]]
    adamw = [p for _, p in groups["adamw"]]
    if not muon_2d and not muon_conv:
        raise ValueError(
            f"{optim_name} has no hidden weight matrix in this model: every parameter "
            "is an embedding, the output head, 1-D, or unrecognised, so the run would "
            "be plain AdamW under a Muon label (F13). Group sizes: "
            + ", ".join(f"{k}={len(groups[k])}" for k in _MUON_GROUP_KEYS)
        )

    # `build_standard_optimizer` has already resolved an explicit `weight_decay`; resolve
    # again here so that a call that passes none still gets MuonW's default 0.1 (and
    # Muon's 0.0) instead of a bare zero (CONTRACTS.md: "MuonW at 0.1 in EVERY scan").
    weight_decay = resolve_weight_decay(optim_name, kwargs.get("weight_decay"))
    adamw_wd = _DEFAULT_WEIGHT_DECAY["AdamW"] if optim_name == "MuonW" else weight_decay

    optimizers = []
    if muon_2d:
        optimizers.append(torch.optim.Muon(
            muon_2d, lr=lr, weight_decay=weight_decay, adjust_lr_fn=MUON_ADJUST_LR_FN))
    if muon_conv:
        optimizers.append(MuonConv(
            muon_conv, lr=lr, weight_decay=weight_decay, adjust_lr_fn=MUON_ADJUST_LR_FN))
    if adamw:
        optimizers.append(torch.optim.AdamW(adamw, lr=lr, weight_decay=adamw_wd))

    optimizer = optimizers[0] if len(optimizers) == 1 else _CombinedOptimizer(*optimizers)
    optimizer.muon_variant = MUON_VARIANT_CONV_FLAT if muon_conv else MUON_VARIANT_HIDDEN2D
    return optimizer


def get_muon_variant(optimizer):
    """The Muon construction actually built, for the run record (C-B5); ``None`` for
    every other optimizer."""
    return getattr(optimizer, "muon_variant", None)


def build_standard_optimizer(model, optim_name, lr=None, **kwargs):
    """Construct a standard PyTorch optimizer by name."""
    if optim_name in _REMOVED_OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{optim_name}' was removed from the registry (C-B6 / F14): the "
            "local implementation differs from the reference algorithm (extra EMA / "
            f"momentum). Use the reference `{_REMOVED_OPTIMIZERS[optim_name]}` from the "
            "`schedulefree` package (not installed) or drop it from the config."
        )
    if "weight_decay" in kwargs:
        kwargs["weight_decay"] = resolve_weight_decay(optim_name, kwargs["weight_decay"])
    if optim_name == "LBFGS":
        lbfgs_kwargs = {
            k: kwargs[k] for k in ("max_iter", "history_size", "line_search_fn")
            if k in kwargs
        }
        return torch.optim.LBFGS(model.parameters(), lr=lr, **lbfgs_kwargs)
    elif optim_name == "PolyakSGD":
        return PolyakSGD(model.parameters(), **kwargs)
    elif optim_name == "SGDm":
        # C-B2: SGD with momentum 0.9 as a named baseline.  Plain "SGD" is unchanged
        # (it falls through to torch.optim.SGD, i.e. momentum 0).
        kwargs.setdefault("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optim_name in ("Muon", "MuonW"):
        return _build_muon(model, optim_name, lr, kwargs)
    elif optim_name == "Shampoo":
        # torch_optimizer.Shampoo — pure drop-in. NOTE: default lr=0.1 is too hot
        # for tiny MLPs; sweep lr down (grid already includes 1e-4..1e-1).
        import torch_optimizer
        return torch_optimizer.Shampoo(
            model.parameters(), lr=lr,
            weight_decay=kwargs.get("weight_decay", 0.0),
            update_freq=kwargs.get("update_freq", 1),
            epsilon=kwargs.get("epsilon", 1e-4),
        )
    elif optim_name == "KFAC":
        # kfac-pytorch: a KFAC preconditioner wrapping a base optimizer (classic
        # K-FAC uses SGD+momentum). Hooks are installed on `model` at construction.
        from kfac.preconditioner import KFACPreconditioner
        base = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=kwargs.get("momentum", 0.9),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
        precond = KFACPreconditioner(
            model,
            factor_update_steps=kwargs.get("factor_update_steps", 1),
            inv_update_steps=kwargs.get("inv_update_steps", 1),
            lr=lr,
            damping=kwargs.get("kfac_damping", 3e-3),
        )
        return _KFACOptimizer(base, precond)
    elif optim_name in _CUSTOM_OPTIMIZERS:
        cls = _CUSTOM_OPTIMIZERS[optim_name]
        return cls(model.parameters(), lr=lr, **kwargs)
    else:
        return getattr(torch.optim, optim_name)(model.parameters(), lr=lr, **kwargs)
