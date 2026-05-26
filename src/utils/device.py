"""Device utilities for CUDA/CPU selection."""

import torch


def get_device() -> torch.device:
    """Return the best available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_device() -> None:
    """Print the device being used for training."""
    device = get_device()
    print(f"[device] Using {device}")
    if device.type == "cuda":
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
