"""Knowledge distillation trainer.

KDTrainer is a Trainer subclass that adds a soft-label KL divergence loss
from a frozen teacher.  Total loss formula (Hinton et al., 2015):

    loss = alpha * T^2 * KL(softmax(s/T) || softmax(t/T))
           + (1 - alpha) * CE(s, labels)

where s = student logits, t = teacher logits, T = temperature.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.trainer import Trainer


class KDTrainer(Trainer):
    def __init__(
        self,
        *,
        teacher: nn.Module,
        temperature: float,
        alpha: float,
        **trainer_kwargs,
    ):
        super().__init__(**trainer_kwargs)

        self.teacher     = teacher.to(self.device).eval()
        self.temperature = temperature
        self.alpha       = alpha

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        print(
            f"[KDTrainer] "
            f"student={type(self.model).__name__}  "
            f"teacher={type(self.teacher).__name__}  "
            f"alpha={self.alpha}  T={self.temperature}  "
            f"teacher_frozen={not any(p.requires_grad for p in self.teacher.parameters())}  "
            f"teacher_eval={not self.teacher.training}"
        )

        # side-channel filled by _train_epoch before compute_loss is called
        self._images: torch.Tensor | None = None

        # running batch components — read by _train_epoch for logging
        self._last_ce: float = 0.0
        self._last_kd: float = 0.0

    # ------------------------------------------------------------------
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(logits, labels)

        T = self.temperature
        with torch.no_grad():
            t_logits = self.teacher(self._images)

        s_log_soft = F.log_softmax(logits   / T, dim=1)
        t_soft     = F.softmax(t_logits / T, dim=1)
        kd_loss    = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T * T)

        self._last_ce = ce_loss.item()
        self._last_kd = kd_loss.item()

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = total_ce = total_kd = 0.0
        correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self._images = images  # expose to compute_loss

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.compute_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            total_loss += loss.item()       * bs
            total_ce   += self._last_ce     * bs
            total_kd   += self._last_kd     * bs
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.writer.add_scalars(
            "kd_loss_components",
            {
                "ce":    total_ce   / total,
                "kd":    total_kd   / total,
                "total": total_loss / total,
            },
            epoch,
        )

        return total_loss / total, correct / total


def load_teacher(
    checkpoint_path: str | Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate a teacher model and load weights from a training checkpoint."""
    if model_name in ("resnet18", "resnet18_teacher"):
        from src.models.resnet18_teacher import ResNet18Teacher
        teacher = ResNet18Teacher(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported teacher model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    return teacher
