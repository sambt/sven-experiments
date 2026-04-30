# svd.py and adam.py use a register_optimizer() stub that is never populated at runtime;
# they are not imported through this __init__ to avoid a circular import.
from .baselines import Lion, ScheduleFreeAdamW, ScheduleFreeSGD
from .hig import HIGWrapper, HIGOptimizer

__all__ = ["Lion", "ScheduleFreeAdamW", "ScheduleFreeSGD", "HIGWrapper", "HIGOptimizer"]
