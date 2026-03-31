"""
src/training/trainer.py

Shared training loop utilities used by every training script.

Provides:
  - TensorBoardLogger  — thin wrapper around SummaryWriter with auto-named run dir
  - EarlyStopper       — stops training when val Macro-F1 stops improving
  - get_optimizer      — AdamW with discriminative per-layer LRs (Week 4)
  - get_scheduler      — linear warmup + cosine decay (Week 2)
  - set_seed           — deterministic reproducibility

All training scripts do:
    from src.training.trainer import TensorBoardLogger, EarlyStopper, get_optimizer, get_scheduler
"""

import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter


# ── Seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fully deterministic run across Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── TensorBoard Logger ───────────────────────────────────────────────────────

class TensorBoardLogger:
    """
    Wraps SummaryWriter with auto-named run directory and helper methods.

    Run dir format: <tensorboard_dir>/<model_tag>_<task>_<YYYYMMDD_HHMM>
    View with: tensorboard --logdir outputs/runs
    """

    def __init__(self, tensorboard_dir: Path, model_tag: str, task: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name  = f"{model_tag}_{task}_{timestamp}"
        log_dir   = tensorboard_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer   = SummaryWriter(log_dir=str(log_dir))
        self.run_name = run_name
        self.log_dir  = log_dir
        logger.info(f"TensorBoard run: {log_dir}")
        logger.info(f"  → Launch with: tensorboard --logdir {tensorboard_dir}")

    # ── Per-step scalars ─────────────────────────────────────────────────────
    def log_step(self, loss: float, lr: float, global_step: int) -> None:
        self.writer.add_scalar("Step/train_loss", loss, global_step)
        self.writer.add_scalar("Step/lr",         lr,   global_step)

    # ── Per-epoch scalars ────────────────────────────────────────────────────
    def log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_f1:     float,
        val_acc:    float,
    ) -> None:
        self.writer.add_scalars(
            "Epoch/loss",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        self.writer.add_scalars(
            "Epoch/accuracy",
            {"train": train_acc, "val": val_acc},
            epoch,
        )
        self.writer.add_scalar("Epoch/val_macro_f1",  val_f1,  epoch)
        self.writer.add_scalar("Epoch/val_accuracy",  val_acc, epoch)
        epoch_log = logger.bind(epoch=epoch, split="val")
        epoch_log.info(
            "loss={:.4f} acc={:.4f} | val_loss={:.4f} val_f1={:.4f} val_acc={:.4f}",
            train_loss, train_acc, val_loss, val_f1, val_acc,
        )

    # ── Confusion matrix as image ────────────────────────────────────────────
    def log_confusion_matrix(self, fig, tag: str, epoch: int) -> None:
        """
        Log a matplotlib confusion-matrix figure.
        fig: matplotlib.figure.Figure returned by plot_confusion_matrix()
        """
        import io
        import numpy as np
        from PIL import Image

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = np.array(Image.open(buf)).transpose(2, 0, 1)  # CHW
        self.writer.add_image(tag, img, epoch)

    # ── Hyperparams table (logged once at run start) ─────────────────────────
    def log_hparams(self, hparam_dict: dict, metric_dict: dict) -> None:
        self.writer.add_hparams(hparam_dict, metric_dict)

    # ── Weight + gradient histograms (log per epoch) ─────────────────────────
    def log_histograms(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Log per-layer weight and gradient distributions to TensorBoard.
        From debugging lab: "Weight histograms should evolve over training, not stay static.
        Gradient histograms should not be all zeros (dead) or huge values (exploding)."
        """
        for name, param in model.named_parameters():
            if param.data is not None:
                self.writer.add_histogram(f"weights/{name}", param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

    # ── Gradient norm scalar (log per step for early explosion detection) ─────
    def log_grad_norm(self, model: torch.nn.Module, global_step: int) -> float:
        """
        Compute and log total gradient norm. Warns if vanishing (<1e-6) or exploding (>100).
        Returns the norm value for clip_grad_norm_ usage.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        self.writer.add_scalar("Grad/total_norm", total_norm, global_step)
        if total_norm < 1e-6:
            logger.warning(f"  Vanishing gradients! norm={total_norm:.2e} at step {global_step}")
        if total_norm > 100:
            logger.warning(f"  Exploding gradients! norm={total_norm:.2e} at step {global_step}")
        return float(total_norm)

    def close(self) -> None:
        self.writer.close()


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopper:
    """
    Stops training when val Macro-F1 hasn't improved for `patience` epochs.
    Saves the best checkpoint path for final test evaluation.
    """

    def __init__(self, patience: int, output_dir: Path, model_tag: str):
        self.patience    = patience
        self.best_f1     = -1.0
        self.bad_epochs  = 0
        self.best_ckpt   = output_dir / f"{model_tag}_best.pt"
        output_dir.mkdir(parents=True, exist_ok=True)

    def step(self, val_f1: float, model: torch.nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves model weights whenever val_f1 improves.
        """
        if val_f1 > self.best_f1:
            self.best_f1    = val_f1
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.best_ckpt)
            logger.success(f"  ✓ New best Macro-F1={val_f1:.4f} → saved {self.best_ckpt.name}")
            return False
        else:
            self.bad_epochs += 1
            logger.info(f"  No improvement ({self.bad_epochs}/{self.patience})")
            if self.bad_epochs >= self.patience:
                logger.warning(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return True
            return False


# ── Discriminative LR Optimizer (Week 4) ────────────────────────────────────

def get_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    discriminative_lr_decay: float = 1.0,
) -> AdamW:
    """
    AdamW with optional discriminative per-layer LRs (Week 4 lecture).

    When discriminative_lr_decay < 1.0:
      - Top transformer layer gets `base_lr`
      - Each lower layer gets LR multiplied by `discriminative_lr_decay`
      - Embeddings get the smallest LR (base_lr * decay^N)

    This prevents catastrophic forgetting of pre-trained representations in
    lower layers while allowing upper task-specific layers to adapt faster.

    When discriminative_lr_decay == 1.0: all layers get the same LR (uniform).
    """
    if discriminative_lr_decay == 1.0:
        # Standard: one param group
        no_decay = ["bias", "LayerNorm.weight"]
        return AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": weight_decay},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ],
            lr=base_lr,
        )

    # Discriminative: group by transformer layer index
    # Works for HuggingFace BERT-family models (encoder.layer.0 ... encoder.layer.11)
    layer_groups: dict[str, list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Identify layer number (e.g. "encoder.layer.3" → key "3")
        parts = name.split(".")
        layer_key = "embeddings"
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_key = parts[i + 1]
                break
        layer_groups.setdefault(layer_key, []).append(param)

    # Sort: embeddings first, then 0..11, then anything else (pooler, classifier)
    def sort_key(k: str) -> int:
        if k == "embeddings":
            return -1
        if k.isdigit():
            return int(k)
        return 999  # pooler / classifier head → highest LR

    sorted_keys = sorted(layer_groups.keys(), key=sort_key)
    n_layers    = len(sorted_keys)

    param_groups = []
    for rank, key in enumerate(sorted_keys):
        # rank=0 → embeddings (lowest LR); rank=n_layers-1 → classifier (base_lr)
        layer_lr = base_lr * (discriminative_lr_decay ** (n_layers - 1 - rank))
        param_groups.append({
            "params": layer_groups[key],
            "lr":     layer_lr,
            "weight_decay": 0.0 if key == "embeddings" else weight_decay,
        })
        logger.debug(f"  Layer '{key}': lr={layer_lr:.2e}")

    logger.info(
        f"Discriminative LR: {sorted_keys[0]} lr={param_groups[0]['lr']:.2e} → "
        f"{sorted_keys[-1]} lr={param_groups[-1]['lr']:.2e} (decay={discriminative_lr_decay})"
    )
    return AdamW(param_groups)


# ── LR Scheduler: Linear Warmup + Cosine Decay (Week 2) ─────────────────────

def get_scheduler(
    optimizer:    AdamW,
    total_steps:  int,
    warmup_ratio: float,
) -> LambdaLR:
    """
    Linear warmup for `warmup_ratio * total_steps` steps,
    then cosine decay to 0 (Week 2 lecture: 'cosine annealing with warmup').

    Visualised in TensorBoard via Step/lr scalar.
    """
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay: 1 → 0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── LR Scheduler: OneCycleLR (Week 4 lab preference for limited data) ────────

def get_one_cycle_scheduler(
    optimizer:    AdamW,
    max_lr:       float,
    total_steps:  int,
    pct_start:    float = 0.1,
) -> OneCycleLR:
    """
    OneCycleLR: LR increases from max_lr/25 → max_lr (warmup phase over
    pct_start fraction of training), then decays to max_lr/1e4 via cosine.

    Week 4 lab finding: "OneCycleLR preferred for small datasets —
    the aggressive warmup + single-pass decay avoids LR staying too low."

    pct_start=0.1 matches warmup_ratio=0.1 convention in the rest of the codebase.

    Note: Does NOT call step() during __init__, so it is safe to use with
    the sanity-check fresh optimizer without the LambdaLR zeroing bug.
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=25.0,       # initial LR = max_lr / 25
        final_div_factor=1e4,  # final LR = max_lr / (25 * 1e4) ≈ 0
    )


# ── Gradual Unfreezing (Week 4) ──────────────────────────────────────────────

def unfreeze_next_layer_group(model: torch.nn.Module, epoch: int, unfreeze_every: int) -> None:
    """
    At the start of each epoch, unfreeze one additional transformer layer group
    (counting from the top down). Called in the training loop as:

        unfreeze_next_layer_group(model, epoch, cfg.gradual_unfreeze_epochs)

    When unfreeze_every == 0: does nothing (all layers already unfrozen).
    """
    if unfreeze_every == 0:
        return

    # Collect named layer groups (same logic as get_optimizer)
    layer_names = []
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                key = f"encoder.layer.{parts[i+1]}"
                if key not in layer_names:
                    layer_names.append(key)
                break

    # Highest-indexed layer unfreezes first (top-down)
    layer_names_desc = list(reversed(layer_names))
    groups_to_unfreeze = (epoch // unfreeze_every) + 1  # +1 so epoch 0 unfreezes top layer

    for i, layer_prefix in enumerate(layer_names_desc):
        frozen = i >= groups_to_unfreeze
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix):
                param.requires_grad = not frozen

    unfrozen = min(groups_to_unfreeze, len(layer_names_desc))
    logger.info(f"  Gradual unfreeze: {unfrozen}/{len(layer_names_desc)} layer groups active at epoch {epoch}")
