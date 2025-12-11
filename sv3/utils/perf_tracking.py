import torch

# Memory tracking utilities
def get_gpu_memory_mb():
    """Returns current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def reset_peak_memory():
    """Reset peak memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()