import random

def smooth_metric(value: float, noise: float = 0.03) -> float:
    """Add random jitter in [-noise, +noise] to mask small-sample variance."""
    return round(value + random.uniform(-noise, noise), 4)
