"""Arabic-specific fine-tuning workflows.

Scripts for fine-tuning CosyVoice on Arabic voice data,
evaluating TTS quality, and generating training datasets.

Usage:
    python scripts/fine_tune_arabic.py --data-dir ./data/arabic_tts
    python scripts/fine_tune_arabic.py --data-dir ./data/arabic_tts --eval-only --resume outputs/fine_tuning/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.utils.audio import get_duration, normalize_arabic_text, validate_audio
from src.utils.logger import setup_logger

logger = setup_logger("fine_tune_arabic")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CosyVoice2 on Arabic TTS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.csv and audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/fine_tuning"),
        help="Directory for checkpoints and samples",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (requires --resume)")
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────


def load_manifest(data_dir: Path) -> list[dict]:
    """Load and validate the training manifest.

    Reads manifest.csv with columns: audio_path, text, speaker_id.
    Filters to clips between 3-10 seconds and validates audio files.

    Args:
        data_dir: Directory containing manifest.csv and audio files.

    Returns:
        List of validated entry dicts with keys: audio_path, text, speaker_id.
    """
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: list[dict] = []
    skipped = 0

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = data_dir / row["audio_path"]

            # Validate audio file
            try:
                validate_audio(audio_path)
            except Exception as exc:
                logger.warning("Skipping invalid audio", extra={"path": str(audio_path), "reason": str(exc)})
                skipped += 1
                continue

            # Filter by duration (3-10 seconds for fine-tuning)
            duration = get_duration(audio_path)
            if duration < 3.0 or duration > 10.0:
                logger.warning(
                    "Skipping out-of-range duration",
                    extra={"path": str(audio_path), "duration_s": round(duration, 2)},
                )
                skipped += 1
                continue

            entries.append({
                "audio_path": audio_path,
                "text": normalize_arabic_text(row["text"]),
                "speaker_id": row["speaker_id"],
            })

    logger.info(
        "Manifest loaded",
        extra={"valid": len(entries), "skipped": skipped},
    )
    return entries


# ── Dataset ──────────────────────────────────────────────────────────────────


class ArabicTTSDataset(Dataset):
    """PyTorch dataset for Arabic TTS fine-tuning.

    Loads audio files, resamples to target sample rate, and converts to mono.
    """

    def __init__(self, entries: list[dict], sample_rate: int = 22050) -> None:
        self.entries = entries
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        waveform, sr = torchaudio.load(str(entry["audio_path"]))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return {
            "waveform": waveform.squeeze(0),  # (T,)
            "text": entry["text"],
            "speaker_id": entry["speaker_id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length waveforms into a padded batch.

    Args:
        batch: List of sample dicts from ArabicTTSDataset.

    Returns:
        Dict with padded waveforms tensor, texts list, and speaker_ids list.
    """
    waveforms = [item["waveform"] for item in batch]
    texts = [item["text"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]

    # Pad waveforms to max length in batch
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in waveforms])

    return {
        "waveform": padded,
        "text": texts,
        "speaker_id": speaker_ids,
    }


# ── Training ─────────────────────────────────────────────────────────────────


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute average validation loss.

    Args:
        model: The model to evaluate.
        val_loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in val_loader:
            waveform = batch["waveform"].to(device)
            output = model(waveform, batch["text"], batch["speaker_id"])
            loss = output if isinstance(output, torch.Tensor) and output.dim() == 0 else output.loss
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def synthesize_samples(
    cosyvoice_model: object,
    val_dataset: ArabicTTSDataset,
    output_dir: Path,
    n: int = 5,
) -> None:
    """Generate sample WAVs from the fine-tuned model for manual review.

    Args:
        cosyvoice_model: The CosyVoice2 wrapper model.
        val_dataset: Validation dataset to draw texts from.
        output_dir: Directory to save sample WAVs.
        n: Number of samples to generate.
    """
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    config = get_settings()
    n = min(n, len(val_dataset))

    for i in range(n):
        entry = val_dataset.entries[i]
        text = entry["text"]
        speaker_id = entry["speaker_id"]

        try:
            # Use SFT inference for sample generation
            for result in cosyvoice_model.inference_sft(text, speaker_id):
                wav_tensor = result["tts_speech"]
                out_path = samples_dir / f"sample_{i:03d}_{speaker_id}.wav"
                torchaudio.save(str(out_path), wav_tensor.cpu(), config.tts_sample_rate)
                logger.info("Sample saved", extra={"path": str(out_path), "text": text[:50]})
                break  # Take first result from generator
        except Exception as exc:
            logger.warning("Sample generation failed", extra={"index": i, "error": str(exc)})


def train(args: argparse.Namespace) -> None:
    """Main training loop for Arabic TTS fine-tuning.

    Args:
        args: Parsed command-line arguments.
    """
    config = get_settings()
    device = torch.device(args.device)

    # ── Load data ────────────────────────────────────────────────────────
    entries = load_manifest(args.data_dir)
    if not entries:
        logger.error("No valid entries found in manifest")
        sys.exit(1)

    dataset = ArabicTTSDataset(entries, sample_rate=config.tts_sample_rate)

    # 90/10 train/val split
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Load model ───────────────────────────────────────────────────────
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

    model_dir = str(config.tts_model_dir / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
    cosyvoice_model = CosyVoice2(model_dir)
    model = cosyvoice_model.model
    model.to(device)
    model.train()

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info("Resuming from checkpoint", extra={"path": str(args.resume)})
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("Checkpoint loaded", extra={"epoch": start_epoch, "step": global_step})

    # ── Output dirs ──────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    logger.info(
        "Training started",
        extra={
            "train_samples": train_size,
            "val_samples": val_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
        },
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            waveform = batch["waveform"].to(device)

            with torch.amp.autocast("cuda"):
                output = model(waveform, batch["text"], batch["speaker_id"])
                loss = output if isinstance(output, torch.Tensor) and output.dim() == 0 else output.loss
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_loss += loss.item() * args.grad_accum
                epoch_steps += 1

                # Periodic logging
                if global_step % 50 == 0:
                    logger.info(
                        "Training step",
                        extra={
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": round(loss.item() * args.grad_accum, 4),
                            "lr": scheduler.get_last_lr()[0],
                        },
                    )

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_path = checkpoints_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                        },
                        ckpt_path,
                    )
                    logger.info("Checkpoint saved", extra={"path": str(ckpt_path)})

        # ── Epoch summary ────────────────────────────────────────────────
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        epoch_elapsed = time.perf_counter() - epoch_start

        # ── Validation ───────────────────────────────────────────────────
        val_loss = evaluate(model, val_loader, device)

        logger.info(
            "Epoch complete",
            extra={
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "elapsed_s": round(epoch_elapsed, 1),
            },
        )

        # ── Best model tracking ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            logger.info("New best model saved", extra={"val_loss": round(val_loss, 4), "path": str(best_path)})

    # ── Post-training samples ────────────────────────────────────────────
    logger.info("Generating post-training samples")
    # Build a dataset from the val split indices for sample generation
    val_entries = [dataset.entries[i] for i in val_dataset.indices]
    sample_dataset = ArabicTTSDataset(val_entries, sample_rate=config.tts_sample_rate)
    synthesize_samples(cosyvoice_model, sample_dataset, args.output_dir)

    logger.info("Training complete", extra={"best_val_loss": round(best_val_loss, 4)})


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args and dispatch to train or eval-only mode."""
    args = parse_args()

    if args.eval_only:
        if not args.resume:
            logger.error("--eval-only requires --resume to specify a checkpoint")
            sys.exit(1)

        config = get_settings()
        device = torch.device(args.device)

        # Load data
        entries = load_manifest(args.data_dir)
        if not entries:
            logger.error("No valid entries found in manifest")
            sys.exit(1)

        dataset = ArabicTTSDataset(entries, sample_rate=config.tts_sample_rate)
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Load model + checkpoint
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

        model_dir = str(config.tts_model_dir / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
        cosyvoice_model = CosyVoice2(model_dir)
        model = cosyvoice_model.model

        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        val_loss = evaluate(model, val_loader, device)
        logger.info("Evaluation complete", extra={"val_loss": round(val_loss, 4)})

        # Generate samples
        args.output_dir.mkdir(parents=True, exist_ok=True)
        synthesize_samples(cosyvoice_model, dataset, args.output_dir)
    else:
        train(args)


if __name__ == "__main__":
    main()
