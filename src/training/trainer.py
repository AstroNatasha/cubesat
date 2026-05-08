"""Minimal training loop. KD and pruning hooks will be added in later phases."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        output_dir: str,
        device: torch.device,
        epochs: int,
        early_stopping_patience: int = 10,
    ):
        self.model     = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.epochs    = epochs
        self.patience  = early_stopping_patience

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.patience_counter = 0

    # ------------------------------------------------------------------
    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss,   val_acc   = self._eval_epoch(epoch)

            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("acc",  {"train": train_acc,  "val": val_acc},  epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self.writer.close()
        print(f"Training complete. Best val acc: {self.best_val_acc:.4f}")

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.compute_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    def _eval_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += images.size(0)

        loss_avg = total_loss / total
        acc      = correct / total
        print(f"  val  loss={loss_avg:.4f}  acc={acc:.4f}")
        return loss_avg, acc

    # ------------------------------------------------------------------
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Override in subclasses (e.g. KDTrainer) to inject extra losses."""
        return self.criterion(logits, labels)

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({"model_state": self.model.state_dict(), "val_acc": self.best_val_acc}, path)
