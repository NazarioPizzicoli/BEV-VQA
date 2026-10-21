from transformers import set_seed as _set_seed


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, torch, and CUDA for reproducible execution."""
    _set_seed(seed)

