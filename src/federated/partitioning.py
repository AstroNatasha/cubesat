"""IID and non-IID data partitioning for federated learning simulation.

Each partition corresponds to one CubeSat client.
"""

import random
from collections import defaultdict

from torch.utils.data import Subset


def iid_partition(train_ds, num_clients: int, seed: int = 42) -> list[Subset]:
    """Randomly split train_ds indices equally across clients."""
    indices = list(train_ds.indices)
    rng = random.Random(seed)
    rng.shuffle(indices)

    n = len(indices)
    splits = []
    start = 0
    for i in range(num_clients):
        size = n // num_clients + (1 if i < n % num_clients else 0)
        splits.append(Subset(train_ds.dataset, indices[start : start + size]))
        start += size

    return splits


def noniid_partition(
    train_ds,
    num_clients: int,
    dominant_classes_per_client: int,
    seed: int = 42,
) -> list[Subset]:
    """Non-IID partition: each client holds 80% of data from its dominant classes.

    Dominant classes are assigned cyclically across clients.  The remaining
    20% of each class is distributed equally among non-dominant clients so
    that every client sees at least some samples from every class.
    """
    rng = random.Random(seed)
    DOMINANT_FRAC = 0.8

    # Group training indices by class label
    all_targets = train_ds.dataset.targets
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in train_ds.indices:
        class_to_indices[all_targets[idx]].append(idx)

    num_classes = len(class_to_indices)
    classes = sorted(class_to_indices.keys())
    for c in classes:
        rng.shuffle(class_to_indices[c])

    k = min(dominant_classes_per_client, num_classes)
    client_dominant = [
        {classes[(i * k + j) % num_classes] for j in range(k)}
        for i in range(num_clients)
    ]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        idx_list = class_to_indices[cls]
        n = len(idx_list)

        dom     = [i for i in range(num_clients) if cls in client_dominant[i]]
        non_dom = [i for i in range(num_clients) if cls not in client_dominant[i]]

        if dom:
            n_dom    = int(n * DOMINANT_FRAC)
            dom_idx  = idx_list[:n_dom]
            rest_idx = idx_list[n_dom:]

            per = n_dom // len(dom)
            for j, ci in enumerate(dom):
                s = j * per
                e = s + per if j < len(dom) - 1 else n_dom
                client_indices[ci].extend(dom_idx[s:e])
        else:
            rest_idx = idx_list

        if non_dom and rest_idx:
            per = len(rest_idx) // len(non_dom)
            for j, ci in enumerate(non_dom):
                s = j * per
                e = s + per if j < len(non_dom) - 1 else len(rest_idx)
                client_indices[ci].extend(rest_idx[s:e])
        elif rest_idx:
            # Every client is dominant: share remainder evenly
            per = len(rest_idx) // num_clients
            for j in range(num_clients):
                s = j * per
                e = s + per if j < num_clients - 1 else len(rest_idx)
                client_indices[j].extend(rest_idx[s:e])

    for idxs in client_indices:
        rng.shuffle(idxs)

    return [Subset(train_ds.dataset, idxs) for idxs in client_indices]
