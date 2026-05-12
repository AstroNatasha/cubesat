"""Federated client — represents a single CubeSat node."""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        device: torch.device,
        num_classes: int,
    ):
        self.client_id   = client_id
        self.dataset     = dataset
        self.device      = device
        self.num_classes = num_classes

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train(
        self,
        global_model: nn.Module,
        local_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        verbose: bool = True,
    ) -> tuple[dict, int, float]:
        """Train a local copy of the global model.

        Returns:
            state_dict: updated local weights
            n_samples:  number of local training samples
            avg_loss:   mean cross-entropy loss over the last epoch
        """
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        loader    = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        last_epoch_loss = 0.0

        epoch_bar = tqdm(
            range(1, local_epochs + 1),
            desc=f"    client {self.client_id:02d} | epoch",
            leave=False,
            disable=not verbose,
        )
        for epoch in epoch_bar:
            total_loss, total = 0.0, 0

            batch_bar = tqdm(
                loader,
                desc=f"      ep {epoch}/{local_epochs}",
                leave=False,
                disable=not verbose,
            )
            for images, labels in batch_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(local_model(images), labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total      += images.size(0)
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            last_epoch_loss = total_loss / total
            epoch_bar.set_postfix(loss=f"{last_epoch_loss:.4f}")

        return local_model.state_dict(), self.num_samples, last_epoch_loss
