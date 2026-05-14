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

    mc = cfg.get("mission_constraints", {})

    print(f"Device:       {device}")
    print(f"Model:        {model_name}")
    print(f"Partitioning: {partitioning}")
    print(f"Clients:      {fl['num_clients']}  rounds={fl['num_rounds']}  local_epochs={fl['local_epochs']}")
    print(f"Output:       {output_dir}")
    if mc:
        print(f"Budget:       {mc.get('max_total_communication_MB','∞')} MB  "
              f"link={mc.get('link_rate_kbps','?')} kbps  "
              f"{mc.get('contacts_per_day','?')} contacts/day × "
              f"{mc.get('contact_window_minutes','?')} min")

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

    from src.federated.compression import Compressor
    comp_cfg    = cfg.get("compression", {})
    comp_mode   = comp_cfg.get("mode", "none")
    compressor  = Compressor(
        mode          = comp_mode,
        topk_fraction = comp_cfg.get("topk_fraction", 0.1),
    )
    print(f"Compression:  {comp_mode}"
          + (f"  topk_fraction={comp_cfg['topk_fraction']}" if comp_mode == "topk_sparse" else ""))

    # --- mission budget: cap rounds if a communication budget is set ---
    comm_budget_mb = mc.get("max_total_communication_MB", None)
    actual_rounds  = fl["num_rounds"]

    if comm_budget_mb is not None:
        from src.federated.mission import max_rounds_from_budget
        n_active_est      = fl.get("clients_per_round", fl["num_clients"])
        tmp_sd            = global_model.state_dict()
        est_upload_mb     = compressor.compressed_bytes(tmp_sd) * n_active_est / 1e6
        est_download_mb   = compressor.download_bytes(tmp_sd)   * n_active_est / 1e6
        est_per_round_mb  = est_upload_mb + est_download_mb
        max_rounds_budget = max_rounds_from_budget(comm_budget_mb, est_per_round_mb)
        actual_rounds     = min(fl["num_rounds"], max_rounds_budget)
        print(
            f"  est {est_per_round_mb:.2f} MB/round → "
            f"budget allows {max_rounds_budget} rounds "
            f"(requested {fl['num_rounds']} → running {actual_rounds})"
        )

    records = server.run(
        num_rounds         = actual_rounds,
        local_epochs       = fl["local_epochs"],
        batch_size         = fl["batch_size"],
        lr                 = t["lr"],
        weight_decay       = t["weight_decay"],
        clients_per_round  = fl.get("clients_per_round", fl["num_clients"]),
        seed               = seed,
        verbose            = cfg.get("verbose", True),
        compressor         = compressor,
        comm_budget_mb     = comm_budget_mb,
    )

    # --- final test evaluation ---
    from src.evaluation.metrics import evaluate_loader, model_size_mb, measure_latency
    final_metrics = evaluate_loader(global_model, test_loader, device, d["num_classes"])
    final_metrics["latency_ms"] = measure_latency(
        global_model,
        input_shape = (1, 3, d["image_size"], d["image_size"]),
        device      = torch.device("cpu"),
    )
    used_comm = records[-1]["cumulative_communication_MB"] if records else 0.0
    final_metrics["size_mb"]                = model_size_mb(output_dir / "final_model.pt")
    final_metrics["total_communication_MB"] = used_comm
    final_metrics["rounds_completed"]       = len(records)
    final_metrics["num_rounds_requested"]   = fl["num_rounds"]
    final_metrics["num_clients"]            = fl["num_clients"]

    # --- mission constraint metrics (only when mc is configured) ---
    if mc:
        from src.federated.mission import compute_mission_metrics
        mission_m = compute_mission_metrics(
            used_comm_mb           = used_comm,
            comm_budget_mb         = mc.get("max_total_communication_MB", float("inf")),
            link_rate_kbps         = mc.get("link_rate_kbps", 9.6),
            contact_window_minutes = mc.get("contact_window_minutes", 10),
            contacts_per_day       = mc.get("contacts_per_day", 3),
            rounds_completed       = len(records),
            rounds_requested       = fl["num_rounds"],
        )
        final_metrics.update(mission_m)
        print("\n=== Mission constraint report ===")
        for k, v in mission_m.items():
            print(f"  {k}: {v}")

    print("\n=== Final test metrics ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    (output_dir / "final_test_metrics.json").write_text(json.dumps(
        {
            "experiment":        cfg["experiment"]["name"],
            "model":             model_name,
            "partitioning":      partitioning,
            "compression_mode":  comp_mode,
            "constrained":       bool(mc),
            **final_metrics,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
