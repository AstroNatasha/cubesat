"""Federated server — orchestrates rounds, aggregation, and evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import evaluate_loader
from src.federated.client import FederatedClient
from src.federated.fedavg import fedavg
from src.federated.metrics import communication_mb_per_round

_SEP = "─" * 50


class FederatedServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: list[FederatedClient],
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        num_classes: int,
        output_dir: str,
    ):
        self.global_model = global_model.to(device)
        self.clients      = clients
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device
        self.num_classes  = num_classes
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")

    # ------------------------------------------------------------------
    def run(
        self,
        num_rounds: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        clients_per_round: int | None = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[dict]:
        rng = random.Random(seed)
        n_active = clients_per_round or len(self.clients)
        cumulative_mb = 0.0
        records: list[dict] = []

        for r in range(1, num_rounds + 1):
            round_start = time.perf_counter()

            # Round header — always visible so the terminal never looks frozen
            print(f"\n{_SEP}")
            print(f"  Round {r}/{num_rounds}")
            print(_SEP)

            selected = rng.sample(self.clients, n_active)

            # ── local training ──────────────────────────────────────────
            updates: list[tuple[dict, int]] = []
            client_losses: list[float] = []
            train_start = time.perf_counter()

            client_bar = tqdm(
                selected,
                desc="  training clients",
                unit="client",
                leave=False,
                disable=not verbose,
            )
            for client in client_bar:
                client_bar.set_description(
                    f"  client {client.client_id:02d}/{len(self.clients) - 1}"
                )
                c_start = time.perf_counter()
                state, n, loss = client.train(
                    self.global_model, local_epochs, batch_size, lr, weight_decay,
                    verbose=verbose,
                )
                c_elapsed = time.perf_counter() - c_start

                updates.append((state, n))
                client_losses.append(loss)

                if verbose:
                    tqdm.write(
                        f"    client {client.client_id:02d}  "
                        f"samples={n:5d}  "
                        f"loss={loss:.4f}  "
                        f"time={c_elapsed:.1f}s"
                    )

            train_elapsed = time.perf_counter() - train_start

            # ── aggregation ─────────────────────────────────────────────
            agg_start = time.perf_counter()
            self.global_model.load_state_dict(fedavg(updates))
            agg_elapsed = time.perf_counter() - agg_start

            # ── global evaluation ────────────────────────────────────────
            val_m  = evaluate_loader(self.global_model, self.val_loader,  self.device, self.num_classes)
            test_m = evaluate_loader(self.global_model, self.test_loader, self.device, self.num_classes)

            comm_mb        = communication_mb_per_round(self.global_model, n_active)
            cumulative_mb += comm_mb
            round_elapsed  = time.perf_counter() - round_start

            record = {
                "round":                       r,
                "val_accuracy":                val_m["accuracy"],
                "val_macro_f1":                val_m["macro_f1"],
                "test_accuracy":               test_m["accuracy"],
                "test_macro_f1":               test_m["macro_f1"],
                "communication_MB_per_round":  comm_mb,
                "cumulative_communication_MB": cumulative_mb,
            }
            records.append(record)

            # TensorBoard
            self.writer.add_scalars(
                "accuracy", {"val": record["val_accuracy"], "test": record["test_accuracy"]}, r
            )
            self.writer.add_scalars(
                "macro_f1", {"val": record["val_macro_f1"], "test": record["test_macro_f1"]}, r
            )
            self.writer.add_scalar("communication/per_round_MB",  comm_mb,       r)
            self.writer.add_scalar("communication/cumulative_MB", cumulative_mb, r)

            # ── round summary — always visible ───────────────────────────
            avg_client_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
            timing = f"train={train_elapsed:.1f}s  agg={agg_elapsed:.2f}s  total={round_elapsed:.1f}s"
            if not verbose:
                timing = f"total={round_elapsed:.1f}s"
            print(
                f"  [{timing}]\n"
                f"  avg_client_loss={avg_client_loss:.4f}\n"
                f"  val_acc={record['val_accuracy']:.4f}   val_f1={record['val_macro_f1']:.4f}\n"
                f"  test_acc={record['test_accuracy']:.4f}  test_f1={record['test_macro_f1']:.4f}\n"
                f"  comm={comm_mb:.2f}MB  cumulative={cumulative_mb:.2f}MB"
            )

        self.writer.close()
        (self.output_dir / "round_metrics.json").write_text(json.dumps(records, indent=2))
        torch.save({"model_state": self.global_model.state_dict()}, self.output_dir / "final_model.pt")
        print(f"\nSaved round metrics and final model → {self.output_dir}/")
        return records
