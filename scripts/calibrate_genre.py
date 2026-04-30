"""
Fit a scalar temperature on the genre classifier's validation logits.

Outputs a JSON report with before/after NLL, ECE, and accuracy so calibrated
genre probabilities can be used in the demo and final report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import GENRE_CLASSES, encode_genre
from src.training.trainer import set_seed


class GenreTextDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int = 32):
        self.samples: list[dict[str, int | str]] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        skipped = 0
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = (record.get("text_corrected") or "").strip()
            label = encode_genre(record.get("genre_en", ""))
            if not text or label == -1:
                skipped += 1
                continue
            self.samples.append({"text": text, "label": label})

        logger.info(
            f"GenreTextDataset: {len(self.samples)} samples from {jsonl_path.name} "
            f"(skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def compute_ece(
    probabilities: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probabilities.device)

    ece = torch.tensor(0.0, device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if float(upper) == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            ece = ece + mask.float().mean() * torch.abs(
                correctness[mask].mean() - confidences[mask].mean()
            )
    return float(ece.item())


def collect_logits(
    model, loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            logits_list.append(logits.cpu())
            labels_list.append(batch["label"].cpu())

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def nll_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(logits / temperature, labels)


def evaluate_split(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> dict[str, float]:
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=1)
    preds = probs.argmax(dim=1)
    return {
        "nll": round(float(F.cross_entropy(scaled, labels).item()), 4),
        "ece": round(compute_ece(probs, labels), 4),
        "accuracy": round(float(preds.eq(labels).float().mean().item()), 4),
    }


def build_model(model_name: str, checkpoint_path: Path, device: torch.device):
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    config.num_labels = len(GENRE_CLASSES)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"Loaded genre checkpoint {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="faisalq/bert-base-arapoembert")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/models/arapoem_genre/arapoem_genre_best.pt"),
    )
    parser.add_argument(
        "--val-jsonl", type=Path, default=Path("data/processed/val.jsonl")
    )
    parser.add_argument(
        "--test-jsonl", type=Path, default=Path("data/processed/test.jsonl")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/genre_temperature_scaling.json"),
    )
    parser.add_argument(
        "--temperature-out",
        type=Path,
        default=Path("outputs/reports/genre_temperature.txt"),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> float:
    logger.add("logs/calibrate_genre.log", rotation="10 MB")
    set_seed(args.seed)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Genre checkpoint not found: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    val_ds = GenreTextDataset(args.val_jsonl, tokenizer, max_seq_len=args.max_seq_len)
    test_ds = GenreTextDataset(args.test_jsonl, tokenizer, max_seq_len=args.max_seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model_name, args.checkpoint, device)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    val_logits_device = val_logits.to(device)
    val_labels_device = val_labels.to(device)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = nll_from_logits(val_logits_device, val_labels_device, temperature)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).item())

    report = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "num_classes": len(GENRE_CLASSES),
        "temperature": round(temperature, 6),
        "validation_before": evaluate_split(val_logits, val_labels, temperature=1.0),
        "validation_after": evaluate_split(
            val_logits, val_labels, temperature=temperature
        ),
        "test_before": evaluate_split(test_logits, test_labels, temperature=1.0),
        "test_after": evaluate_split(test_logits, test_labels, temperature=temperature),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.temperature_out.write_text(f"{temperature:.8f}\n", encoding="utf-8")

    logger.success(
        f"Temperature scaling complete | T={temperature:.4f} | "
        f"val ECE {report['validation_before']['ece']:.4f} -> {report['validation_after']['ece']:.4f}"
    )
    logger.info(f"Calibration report → {args.output}")
    logger.info(f"Temperature value → {args.temperature_out}")
    return temperature


if __name__ == "__main__":
    main(parse_args())
