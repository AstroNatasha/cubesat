"""Federated learning simulator entry point.

Usage:
    # IID (from config)
    python train_federated.py --config configs/federated_baseline.yaml

    # non-IID (CLI override — saves to results/federated_baseline_noniid/)
    python train_federated.py --config configs/federated_baseline.yaml --partitioning noniid

    # MobileNetV3-Small, non-IID
    python train_federated.py --config configs/federated_baseline.yaml --model mobilenet_v3_small --partitioning noniid
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CubeSat federated learning simulator")
    p.add_argument("--config",       required=True, type=Path)
    p.add_argument("--partitioning", default=None, choices=["iid", "noniid"],
                   help="Override config partitioning mode")
    p.add_argument("--model",        default=None,
                   help="Override config model name (baseline_cnn | mobilenet_v3_small)")
    p.add_argument("--device",       default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "baseline_cnn":
        from src.models.baseline_cnn import BaselineCNN
        return BaselineCNN(num_classes=num_classes)
    if name == "mobilenet_v3_small":
        from src.models.mobilenet_v3_small import MobileNetV3Small
        return MobileNetV3Small(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {name}")


def main():
    args = parse_args()
    cfg  = yaml.safe_load(args.config.read_text())

    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    fl           = cfg["federated"]
    partitioning = args.partitioning or fl["partitioning"]
    model_name   = args.model        or cfg["model"]["name"]

    # If CLI overrides are applied, suffix the output dir so runs don't collide
    output_dir = Path(cfg["experiment"]["output_dir"])
    suffix = []
    if args.partitioning:
        suffix.append(partitioning)
    if args.model:
        suffix.append(model_name)
    if suffix:
        output_dir = output_dir.parent / f"{output_dir.name}_{'_'.join(suffix)}"

    d = cfg["dataset"]
    t = cfg["training"]

    print(f"Device:       {device}")
    print(f"Model:        {model_name}")
    print(f"Partitioning: {partitioning}")
    print(f"Clients:      {fl['num_clients']}  rounds={fl['num_rounds']}  local_epochs={fl['local_epochs']}")
    print(f"Output:       {output_dir}")

    # --- data ---
    from src.data.eurosat import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        root        = d["root"],
        image_size  = d["image_size"],
        batch_size  = 128,
        train_split = d["train_split"],
        val_split   = d["val_split"],
        num_workers = d["num_workers"],
        seed        = seed,
    )
    train_ds = train_loader.dataset  # torch.utils.data.Subset

    # --- partition ---
    if partitioning == "iid":
        from src.federated.partitioning import iid_partition
        client_datasets = iid_partition(train_ds, fl["num_clients"], seed=seed)
    else:
        from src.federated.partitioning import noniid_partition
        client_datasets = noniid_partition(
            train_ds,
            fl["num_clients"],
            dominant_classes_per_client=fl["dominant_classes_per_client"],
            seed=seed,
        )

    sizes = [len(ds) for ds in client_datasets]
    print(f"Client sizes: min={min(sizes)}  max={max(sizes)}  total={sum(sizes)}")

    # --- clients ---
    from src.federated.client import FederatedClient
    clients = [
        FederatedClient(i, ds, device, d["num_classes"])
        for i, ds in enumerate(client_datasets)
    ]

    # --- global model ---
    global_model = build_model(model_name, d["num_classes"])
    n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"Parameters:   {n_params:,}")

    # --- server ---
    from src.federated.server import FederatedServer
    server = FederatedServer(
        global_model = global_model,
        clients      = clients,
        val_loader   = val_loader,
        test_loader  = test_loader,
        device       = device,
        num_classes  = d["num_classes"],
        output_dir   = str(output_dir),
    )

    records = server.run(
        num_rounds         = fl["num_rounds"],
        local_epochs       = fl["local_epochs"],
        batch_size         = fl["batch_size"],
        lr                 = t["lr"],
        weight_decay       = t["weight_decay"],
        clients_per_round  = fl.get("clients_per_round", fl["num_clients"]),
        seed               = seed,
        verbose            = cfg.get("verbose", True),
    )

    # --- final test evaluation ---
    from src.evaluation.metrics import evaluate_loader, model_size_mb, measure_latency
    final_metrics = evaluate_loader(global_model, test_loader, device, d["num_classes"])
    final_metrics["latency_ms"] = measure_latency(
        global_model,
        input_shape = (1, 3, d["image_size"], d["image_size"]),
        device      = torch.device("cpu"),
    )
    final_metrics["size_mb"]               = model_size_mb(output_dir / "final_model.pt")
    final_metrics["total_communication_MB"] = records[-1]["cumulative_communication_MB"]
    final_metrics["num_rounds"]             = fl["num_rounds"]
    final_metrics["num_clients"]            = fl["num_clients"]

    print("\n=== Final test metrics ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    (output_dir / "final_test_metrics.json").write_text(json.dumps(
        {
            "experiment":   cfg["experiment"]["name"],
            "model":        model_name,
            "partitioning": partitioning,
            **final_metrics,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
