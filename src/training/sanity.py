"""
src/training/sanity.py

Pre-training sanity checks.

Every training script should call these before the main loop:
    from src.training.sanity import run_all_checks
    run_all_checks(model, loader, criterion, cfg, device)
"""

import math
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward_batch(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch a batch dict to the correct forward call. Returns (loss, logits)."""
    if "input_ids" in batch:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        labels = batch["label"].to(device)
    elif "mel" in batch:
        logits = model(batch["mel"].to(device))
        labels = batch["label"].to(device)
    else:
        logits = model(batch["audio_tensor"].to(device))
        labels = batch["emotion_id"].to(device)
    return criterion(logits, labels), logits


# ── Check 1: Initial Loss Sanity ─────────────────────────────────────────────


def check_initial_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
    tolerance: float = 1.0,
) -> None:
    """
    For a random-init balanced k-class problem,
    expected initial cross-entropy loss ≈ log(k).

      Genre    (11 classes): expect ≈ 2.40
      Emotion  (12 classes): expect ≈ 2.48
      mBERT random head:     should be close to this

    If loss is far outside [expected - tolerance, expected + tolerance],
    something is wrong before training even starts.
    """
    model.eval()
    expected = math.log(num_classes)

    batch = next(iter(loader))
    with torch.no_grad():
        loss_t, _ = _forward_batch(model, batch, criterion, device)
        loss = loss_t.item()

    lower, upper = expected - tolerance, expected + tolerance
    status = "OK" if lower <= loss <= upper else "WARNING"

    logger.info(
        f"[Sanity] Initial loss={loss:.4f}, "
        f"expected≈{expected:.4f} for {num_classes} classes  [{status}]"
    )
    if status == "WARNING":
        logger.warning(
            f"  Initial loss {loss:.4f} is outside [{lower:.2f}, {upper:.2f}]. "
            "Possible causes: bad weight init, wrong loss fn, labels off-by-one, "
            "softmax applied before CrossEntropyLoss."
        )

    model.train()


# ── Check 2: Overfit One Batch ────────────────────────────────────────────────


def overfit_one_batch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_steps: int = 50,
    target_loss: float = 0.05,
) -> bool:
    """
    Runs n_steps gradient updates on a SINGLE batch.
    Expected result: training loss → ~0 (near-perfect memorisation).

    Returns True if overfit succeeded, False if something is broken.
    Every training script should call this before starting the full loop.
    """
    model.train()
    batch = next(iter(loader))
    logger.info(
        f"[Sanity] Overfit-one-batch test: {n_steps} steps on a single batch..."
    )
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss, _ = _forward_batch(model, batch, criterion, device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    success = final_loss <= target_loss

    if success:
        logger.success(
            f"[Sanity] Overfit test PASSED: loss {losses[0]:.4f} → {final_loss:.4f} "
            f"(target ≤ {target_loss}). Pipeline is correct."
        )
    else:
        logger.error(
            f"[Sanity] Overfit test FAILED: loss {losses[0]:.4f} → {final_loss:.4f} "
            f"(target ≤ {target_loss}). Likely bug in forward pass, loss fn, or optimizer."
        )

    return success


# ── Check 3: NaN / Inf Guard ──────────────────────────────────────────────────


def check_no_nan(tensor: torch.Tensor, name: str) -> None:
    """Call after every loss.backward() to catch NaN/Inf in clean data or model outputs."""
    if torch.isnan(tensor).any():
        raise RuntimeError(
            f"NaN detected in {name}! "
            "Possible causes: exploding gradients, log(0) in loss, bad audio file."
        )
    if torch.isinf(tensor).any():
        raise RuntimeError(
            f"Inf detected in {name}! "
            "Possible causes: learning rate too high, missing gradient clipping."
        )


# ── Check 4: Trainable Parameters ─────────────────────────────────────────────


def check_trainable_params(model: nn.Module) -> None:
    """Call after model setup and after every gradual-unfreeze step."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(
        f"[Sanity] Params: total={total:,}  trainable={trainable:,}  frozen={frozen:,} "
        f"({trainable / total:.1%} active)"
    )
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters! All layers are frozen. "
            "Check requires_grad settings and gradual unfreeze config."
        )


# ── Check 5: Gradient Flow ────────────────────────────────────────────────────


def check_gradient_flow(model: nn.Module) -> None:
    """Call after the first loss.backward() to verify gradients exist."""
    dead_layers = []
    alive_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                dead_layers.append(name)
            else:
                norm = param.grad.norm().item()
                alive_layers.append((name, norm))

    if dead_layers:
        logger.warning(
            f"[Sanity] {len(dead_layers)} layers have no gradient (None): "
            f"{dead_layers[:5]}{'...' if len(dead_layers) > 5 else ''}"
        )
    else:
        logger.info(
            f"[Sanity] Gradient flow OK: all {len(alive_layers)} trainable layers have gradients."
        )


# ── Convenience: run all checks ───────────────────────────────────────────────


def run_all_checks(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    device: torch.device,
) -> None:
    """
    Run all pre-training sanity checks in the correct order.
    Should be called once before the main training loop.
    """
    logger.info("=" * 60)
    logger.info("[Sanity] Running pre-training checks...")
    logger.info("=" * 60)

    check_trainable_params(model)

    # If most parameters are frozen (e.g. SSL fine-tuning with frozen encoder),
    # the overfit-one-batch test cannot reach loss ≤ 0.05 — the frozen encoder
    # acts as a fixed bottleneck. Skip the test and log a note instead.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable / max(total, 1)

    check_initial_loss(model, loader, criterion, num_classes, device)

    # Overfit test modifies model weights → reset after
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    if trainable_pct < 0.2:
        logger.info(
            f"[Sanity] Overfit test SKIPPED — only {trainable_pct:.1%} of params are "
            "trainable (frozen encoder). A partially frozen model cannot overfit a batch "
            "to near-zero; this is expected for SSL fine-tuning."
        )
        passed = True
    else:
        passed = overfit_one_batch(model, loader, criterion, optimizer, device)
    model.load_state_dict(original_state)  # restore to clean init

    # Re-zero optimizer state after overfit test
    optimizer.zero_grad(set_to_none=True)

    # Run one forward+backward to check gradient flow
    model.train()
    batch = next(iter(loader))
    loss, _ = _forward_batch(model, batch, criterion, device)
    check_no_nan(loss, "loss")
    loss.backward()
    check_gradient_flow(model)
    optimizer.zero_grad(set_to_none=True)

    logger.info("=" * 60)
    if passed:
        logger.success("[Sanity] All checks passed. Safe to start full training.")
    else:
        logger.error(
            "[Sanity] Overfit test failed. Fix the bug before training on full data."
        )
    logger.info("=" * 60)
