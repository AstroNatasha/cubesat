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
from src.federated.compression import Compressor
from src.federated.fedavg import fedavg

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
        compressor: Compressor | None = None,
        comm_budget_mb: float | None = None,
    ) -> list[dict]:
        rng = random.Random(seed)
        n_active     = clients_per_round or len(self.clients)
        cumulative_mb = 0.0
        records: list[dict] = []

        # Fall back to a no-op compressor so the rest of the code is uniform
        comp = compressor or Compressor("none")

        for r in range(1, num_rounds + 1):
            round_start = time.perf_counter()

            # ── global state dict (needed for budget check + download estimate) ──
            global_sd = self.global_model.state_dict()

            # ── communication budget guard — check before any client training ──
            if comm_budget_mb is not None:
                est_upload   = comp.compressed_bytes(global_sd) * n_active
                est_download = comp.download_bytes(global_sd)   * n_active
                est_round_mb = (est_upload + est_download) / 1e6
                if cumulative_mb + est_round_mb > comm_budget_mb:
                    print(
                        f"\n{_SEP}\n"
                        f"  Budget exhausted: {cumulative_mb:.2f} MB used, "
                        f"round {r} costs ~{est_round_mb:.2f} MB "
                        f"(budget: {comm_budget_mb:.1f} MB). Stopping.\n"
                        f"{_SEP}"
                    )
                    break

            # Round header — always visible so the terminal never looks frozen
            print(f"\n{_SEP}")
            print(f"  Round {r}/{num_rounds}  [{comp.mode}]"
                  + (f"  budget={comm_budget_mb - cumulative_mb:.1f} MB remaining"
                     if comm_budget_mb is not None else ""))
            print(_SEP)

            selected = rng.sample(self.clients, n_active)

            # ── download byte estimate (server → clients, pre-aggregation) ──
            dl_bytes_each       = comp.download_bytes(global_sd)   # FP32 for topk (delta needs full model)
            dl_fp32_each        = comp.fp32_bytes(global_sd)
            download_comp_bytes = n_active * dl_bytes_each
            download_fp32_bytes = n_active * dl_fp32_each

            # ── local training ──────────────────────────────────────────────
            updates: list[tuple[dict, int]] = []
            client_losses: list[float] = []
            upload_comp_bytes = 0
            upload_fp32_bytes = 0
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

                # Apply compression — for topk, sparsifies the delta (local − global)
                comp_state, u_comp = comp.apply(state, global_sd)
                upload_comp_bytes += u_comp
                upload_fp32_bytes += comp.fp32_bytes(state)
                updates.append((comp_state, n))
                client_losses.append(loss)

                if verbose:
                    tqdm.write(
                        f"    client {client.client_id:02d}  "
                        f"samples={n:5d}  "
                        f"loss={loss:.4f}  "
                        f"time={c_elapsed:.1f}s"
                    )

            train_elapsed = time.perf_counter() - train_start

            # ── aggregation ─────────────────────────────────────────────────
            agg_start = time.perf_counter()
            self.global_model.load_state_dict(fedavg(updates))
            agg_elapsed = time.perf_counter() - agg_start

            # ── global evaluation ────────────────────────────────────────────
            val_m  = evaluate_loader(self.global_model, self.val_loader,  self.device, self.num_classes)
            test_m = evaluate_loader(self.global_model, self.test_loader, self.device, self.num_classes)

            # ── communication accounting ─────────────────────────────────────
            total_comp_bytes = upload_comp_bytes + download_comp_bytes
            total_fp32_bytes = upload_fp32_bytes + download_fp32_bytes
            comm_mb_compressed   = total_comp_bytes / 1e6
            comm_mb_uncompressed = total_fp32_bytes / 1e6
            compression_ratio    = comm_mb_uncompressed / max(comm_mb_compressed, 1e-9)
            cumulative_mb       += comm_mb_compressed
            round_elapsed        = time.perf_counter() - round_start

            record = {
                "round":                          r,
                "val_accuracy":                   val_m["accuracy"],
                "val_macro_f1":                   val_m["macro_f1"],
                "test_accuracy":                  test_m["accuracy"],
                "test_macro_f1":                  test_m["macro_f1"],
                "communication_MB_per_round":     comm_mb_compressed,
                "communication_MB_uncompressed":  comm_mb_uncompressed,
                "compression_ratio":              compression_ratio,
                "cumulative_communication_MB":    cumulative_mb,
            }
            records.append(record)

            # TensorBoard
            self.writer.add_scalars(
                "accuracy", {"val": record["val_accuracy"], "test": record["test_accuracy"]}, r
            )
            self.writer.add_scalars(
                "macro_f1", {"val": record["val_macro_f1"], "test": record["test_macro_f1"]}, r
            )
            self.writer.add_scalars(
                "communication_MB",
                {"compressed": comm_mb_compressed, "uncompressed": comm_mb_uncompressed},
                r,
            )
            self.writer.add_scalar("communication/compression_ratio", compression_ratio,  r)
            self.writer.add_scalar("communication/cumulative_MB",      cumulative_mb,      r)

            # ── round summary — always visible ───────────────────────────────
            avg_client_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
            timing = f"train={train_elapsed:.1f}s  agg={agg_elapsed:.2f}s  total={round_elapsed:.1f}s"
            if not verbose:
                timing = f"total={round_elapsed:.1f}s"
            comp_summary = (
                f"comm={comm_mb_uncompressed:.2f}MB → {comm_mb_compressed:.2f}MB "
                f"(ratio={compression_ratio:.2f}x)"
                if comp.mode != "none"
                else f"comm={comm_mb_compressed:.2f}MB"
            )
            print(
                f"  [{timing}]\n"
                f"  avg_client_loss={avg_client_loss:.4f}\n"
                f"  val_acc={record['val_accuracy']:.4f}   val_f1={record['val_macro_f1']:.4f}\n"
                f"  test_acc={record['test_accuracy']:.4f}  test_f1={record['test_macro_f1']:.4f}\n"
                f"  {comp_summary}  cumulative={cumulative_mb:.2f}MB"
            )
            if comp.mode == "topk_sparse":
                total_params     = sum(t.numel() for t in global_sd.values() if t.is_floating_point())
                transmitted      = sum(
                    max(1, int(t.numel() * comp.topk_fraction))
                    for t in global_sd.values() if t.is_floating_point()
                )
                eff_sparsity = 1.0 - transmitted / max(total_params, 1)
                print(
                    f"  [topk] fraction={comp.topk_fraction}  "
                    f"transmitted={transmitted:,}/{total_params:,}  "
                    f"sparsity={eff_sparsity:.1%}  "
                    f"upload={upload_comp_bytes/1e6:.3f}MB"
                )

        self.writer.close()
        (self.output_dir / "round_metrics.json").write_text(json.dumps(records, indent=2))
        torch.save({"model_state": self.global_model.state_dict()}, self.output_dir / "final_model.pt")
        print(f"\nSaved round metrics and final model → {self.output_dir}/")
        return records
