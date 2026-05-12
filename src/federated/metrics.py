"""Communication cost utilities for the federated simulator."""

import torch.nn as nn


def model_bytes(model: nn.Module) -> int:
    """Total bytes used by all parameter tensors (float32 = 4 bytes each)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def communication_mb_per_round(model: nn.Module, num_active_clients: int) -> float:
    """Upload + download cost for one FL round, in megabytes.

    Each round the server sends the global model to every active client
    (download) and each client sends its updated model back (upload).
    """
    bytes_per_model = model_bytes(model)
    total_bytes = bytes_per_model * num_active_clients * 2  # bidirectional
    return total_bytes / 1e6
