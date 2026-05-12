"""Federated averaging (FedAvg) aggregation.

McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.
"""

import torch


def fedavg(updates: list[tuple[dict, int]]) -> dict:
    """Weighted average of state dicts by number of local samples.

    Args:
        updates: list of (state_dict, n_samples) from each participating client.

    Returns:
        Aggregated state dict with the same keys as the inputs.
    """
    total_samples = sum(n for _, n in updates)
    avg_state: dict = {}

    for key in updates[0][0]:
        avg_state[key] = sum(
            sd[key].float() * (n / total_samples) for sd, n in updates
        )

    return avg_state
