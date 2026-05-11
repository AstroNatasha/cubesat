"""Entry point for training experiments.

Usage:
    python train.py --config configs/baseline_cnn.yaml
    python train.py --config configs/resnet18_teacher.yaml
    python train.py --config configs/mobilenet_v3.yaml
    python train.py --config configs/mobilenet_kd.yaml
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CubeSat compression experiments")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> torch.nn.Module:
    name        = cfg["model"]["name"]
    num_classes = cfg["dataset"]["num_classes"]
    pretrained  = cfg["model"].get("pretrained", False)

    if name == "baseline_cnn":
        from src.models.baseline_cnn import BaselineCNN
        return BaselineCNN(num_classes=num_classes)

    if name == "resnet18_teacher":
        from src.models.resnet18_teacher import ResNet18Teacher
        return ResNet18Teacher(num_classes=num_classes, pretrained=pretrained)

    if name == "mobilenet_v3_small":
        from src.models.mobilenet_v3_small import MobileNetV3Small
        return MobileNetV3Small(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model: {name}")


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["training"]
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"], momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {t['optimizer']}")


def build_scheduler(optimizer, cfg: dict):
    t = cfg["training"]
    if t.get("scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    return None


def save_metrics(metrics: dict, output_dir: Path, cfg: dict):
    record = {
        "experiment": cfg["experiment"]["name"],
        "model":      cfg["model"]["name"],
        "pretrained": cfg["model"].get("pretrained", False),
        "image_size": cfg["dataset"]["image_size"],
        **metrics,
    }
    path = output_dir / "test_metrics.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"  Metrics saved → {path}")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device:     {device}")
    print(f"Experiment: {cfg['experiment']['name']}")

    # --- data ---
    from src.data.eurosat import build_dataloaders
    d = cfg["dataset"]
    train_loader, val_loader, test_loader = build_dataloaders(
        root        = d["root"],
        image_size  = d["image_size"],
        batch_size  = cfg["training"]["batch_size"],
        train_split = d["train_split"],
        val_split   = d["val_split"],
        num_workers = d["num_workers"],
        seed        = cfg["experiment"]["seed"],
    )

    # --- model ---
    model     = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- trainer ---
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = dict(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        output_dir   = str(output_dir),
        device       = device,
        epochs       = cfg["training"]["epochs"],
        early_stopping_patience = cfg["training"].get("early_stopping_patience", 10),
    )

    if cfg["training"].get("use_kd", False):
        from src.training.kd_trainer import KDTrainer, load_teacher
        t_cfg    = cfg["teacher"]
        kd_cfg   = cfg["distillation"]
        teacher  = load_teacher(
            checkpoint_path = t_cfg["checkpoint"],
            model_name      = t_cfg["name"],
            num_classes     = cfg["dataset"]["num_classes"],
            device          = device,
        )
        print(f"Teacher loaded from {t_cfg['checkpoint']}")
        trainer = KDTrainer(
            teacher     = teacher,
            temperature = kd_cfg["temperature"],
            alpha       = kd_cfg["alpha"],
            **trainer_kwargs,
        )
    else:
        from src.training.trainer import Trainer
        trainer = Trainer(**trainer_kwargs)

    trainer.fit()

    # --- test-set evaluation ---
    from src.evaluation.metrics import evaluate_loader, model_size_mb, measure_latency
    ckpt = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_loader(model, test_loader, device, num_classes=d["num_classes"])
    metrics["latency_ms"] = measure_latency(
        model, input_shape=(1, 3, d["image_size"], d["image_size"]), device=torch.device("cpu")
    )
    metrics["size_mb"] = model_size_mb(output_dir / "best.pt")

    print("\n=== Test results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    save_metrics(metrics, output_dir, cfg)


if __name__ == "__main__":
    main()
