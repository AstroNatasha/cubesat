"""Communication compression for federated learning simulation.

Three modes
───────────
  none              – FP32 baseline, no information loss.
  int8_quantization – per-tensor min-max quantisation to INT8.
                      Introduces rounding error; aggregation stays in FP32.
  topk_sparse       – transmit only the top-k% largest-magnitude *update* values.
                      The update (delta = local − global) is sparsified; the
                      global weights remain intact in the reconstructed tensor.

Byte cost model
───────────────
  FP32 (baseline)    : P × 4 bytes
  INT8               : P × 1 byte  + 8 bytes/tensor (scale + offset)
  TopK (fraction f)  : K × 8 bytes upload (4 val + 4 idx per entry, FP32+INT32)
                       P × 4 bytes download (full FP32 needed to compute delta)
                       where K = max(1, floor(P × f))

TopK correctness note
─────────────────────
  Sparsifying raw weights (not deltas) zeros 90% of the model parameters.
  FedAvg then averages near-zero vectors → random-guessing accuracy (~10%).
  Correct approach: sparsify the update delta, reconstruct global + sparse_delta,
  and pass the reconstructed full tensor to FedAvg.  FedAvg(global + sparse_delta)
  with equal weights = global + FedAvg(sparse_delta), which is the intended result.
"""

from __future__ import annotations

import torch


class Compressor:
    def __init__(self, mode: str = "none", topk_fraction: float = 0.1):
        if mode not in ("none", "int8_quantization", "topk_sparse"):
            raise ValueError(f"Unknown compression mode: {mode!r}")
        self.mode          = mode
        self.topk_fraction = topk_fraction

    # ------------------------------------------------------------------
    def apply(
        self,
        local_sd: dict,
        global_sd: dict | None = None,
    ) -> tuple[dict, int]:
        """Return (compressed_state_dict, upload_bytes).

        compressed_state_dict:
          For none/int8: full weights (possibly rounded).
          For topk:      global + sparse_delta  (so FedAvg averages correctly).
        upload_bytes: estimated transmission size for the client→server direction.

        global_sd is required for topk_sparse (needed to compute the delta).
        """
        if self.mode == "none":
            return local_sd, self.fp32_bytes(local_sd)
        if self.mode == "int8_quantization":
            return _int8_apply(local_sd)
        if self.mode == "topk_sparse":
            if global_sd is None:
                raise ValueError("topk_sparse requires global_sd to compute update deltas")
            return _topk_apply_delta(local_sd, global_sd, self.topk_fraction)

    def compressed_bytes(self, state_dict: dict) -> int:
        """Upload byte estimate (client→server direction)."""
        if self.mode == "none":
            return self.fp32_bytes(state_dict)
        if self.mode == "int8_quantization":
            return sum(t.numel() + 8 for t in state_dict.values())
        if self.mode == "topk_sparse":
            total = 0
            for t in state_dict.values():
                if not t.is_floating_point():
                    total += t.numel() * t.element_size()
                else:
                    total += max(1, int(t.numel() * self.topk_fraction)) * 8
            return total

    def download_bytes(self, state_dict: dict) -> int:
        """Download byte estimate (server→client direction).

        TopK: clients need the full FP32 model to compute their delta.
        Other modes: use compressed format for the download as well.
        """
        if self.mode == "topk_sparse":
            return self.fp32_bytes(state_dict)
        return self.compressed_bytes(state_dict)

    @staticmethod
    def fp32_bytes(state_dict: dict) -> int:
        """FP32 baseline byte count (4 bytes per parameter)."""
        return sum(t.numel() * 4 for t in state_dict.values())


# ── INT8 quantisation ──────────────────────────────────────────────────────

def _int8_apply(state_dict: dict) -> tuple[dict, int]:
    """Simulate INT8 transmission via per-tensor min-max quantisation.

    Storage model: 1 byte per parameter + 8 bytes per tensor (FP32 scale + offset).
    The returned state-dict is in FP32 after dequantisation.
    """
    compressed: dict = {}
    total_bytes: int = 0

    for key, tensor in state_dict.items():
        t = tensor.float()
        n = t.numel()
        t_min = t.min().item()
        t_max = t.max().item()

        if t_min == t_max or n == 0:
            compressed[key] = tensor.clone()
            total_bytes += n + 8
            continue

        scale = (t_max - t_min) / 255.0
        q  = ((t - t_min) / scale - 128).round().clamp(-128, 127).to(torch.int8)
        dq = (q.float() + 128) * scale + t_min
        compressed[key] = dq.to(tensor.dtype)
        total_bytes += n + 8  # 1 byte/param + 2×FP32 overhead (scale, t_min)

    return compressed, total_bytes


# ── TopK update sparsification ─────────────────────────────────────────────

def _topk_apply_delta(
    local_sd: dict,
    global_sd: dict,
    fraction: float,
) -> tuple[dict, int]:
    """Sparsify the update delta and reconstruct global + sparse_delta.

    Algorithm:
      1. delta = local − global  (the model update for this client)
      2. Keep only the top-k% entries of delta by absolute value; zero the rest.
      3. Return global + sparse_delta as the compressed state dict.

    FedAvg then computes weighted_avg(global + sparse_delta_i) for each client i,
    which equals global + weighted_avg(sparse_delta_i) — the correct aggregation.

    Non-floating-point tensors (e.g. BatchNorm num_batches_tracked) are passed
    through unchanged; they are excluded from the sparsity budget.
    """
    compressed: dict = {}
    total_bytes: int = 0

    for key in local_sd:
        local   = local_sd[key]
        global_ = global_sd[key]

        if not local.is_floating_point():
            compressed[key] = local.clone()
            total_bytes += local.numel() * local.element_size()
            continue

        delta = local.float() - global_.float()
        flat  = delta.flatten()
        n     = flat.numel()
        k     = max(1, int(n * fraction))

        if k >= n:
            # Sending all entries — no sparsification needed
            compressed[key] = local.clone()
            total_bytes += n * 8
            continue

        _, top_idx = flat.abs().topk(k, largest=True, sorted=False)
        mask         = torch.zeros(n, dtype=torch.bool, device=delta.device)
        mask[top_idx] = True
        sparse_delta  = torch.where(mask, flat, torch.zeros_like(flat))

        # Reconstruct: global weights + sparse update
        reconstructed = (global_.float().flatten() + sparse_delta).view_as(local)
        compressed[key] = reconstructed.to(local.dtype)
        total_bytes += k * 8  # 4 bytes value (FP32) + 4 bytes index (INT32)

    return compressed, total_bytes
