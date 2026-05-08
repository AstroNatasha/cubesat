"""Evaluation utilities: accuracy, macro-F1, model size, latency."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_params": count_parameters(model),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6


def measure_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 64, 64),
    device: torch.device = torch.device("cpu"),
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """Return mean inference latency in milliseconds."""
    model.eval().to(device)
    dummy = torch.randn(input_shape, device=device)

    for _ in range(n_warmup):
        model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = (time.perf_counter() - start) / n_runs

    return elapsed * 1000  # ms
