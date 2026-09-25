from .baselines import Lion, ScheduleFreeAdamW, ScheduleFreeSGD
# Muon for >2-D (conv) hidden weights, used by optim_factory's Muon / MuonW
# construction (C-B5); exported here so it can be imported by name.
from .muon_conv import MuonConv

__all__ = ["Lion", "MuonConv", "ScheduleFreeAdamW", "ScheduleFreeSGD"]
