#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentation import RandomFlipRotate  # noqa: E402
from dataset import MicroscopyTiffDataset, get_tif_paths_for_split  # noqa: E402
from models import TinyUNet, count_parameters  # noqa: E402
from utilities.mapping import get_input_indices, get_input_names, get_target_index, get_target_name, set_task_by_names  # noqa: E402


DEFAULT_MANIFEST = Path("data/splits/stacked_raw_split.json")
DEFAULT_STATS = Path("data/stats/stacked_raw_stats_train.json")
DEFAULT_OUTPUT_DIR = Path("runs")


@dataclass
class EpochMetrics:
    epoch: int
    train_l1: float
    val_l1: float
    val_mse: float
    val_ssim: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TinyUNet baseline with train/val splits.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the split manifest JSON.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help="Path to the train-only stats JSON used for normalization.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for training runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of train samples for quick validation runs.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional cap on the number of val samples for quick validation runs.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base feature width for TinyUNet.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Akt",
        help="Target protein name (must match a name in CHANNEL_NAMES).",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience (epochs with no val_l1 improvement before reducing lr).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau reduction factor.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable joint random flip/rotate augmentation on the train split.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="StainNet",
        help="WandB project name when WandB logging is enabled.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_paths(paths: list[Path], limit: Optional[int], seed: int) -> list[Path]:
    if limit is None:
        return list(paths)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}.")
    if limit > len(paths):
        raise ValueError(f"Requested limit={limit}, but only {len(paths)} samples are available.")
    if limit == len(paths):
        return list(paths)
    return random.Random(seed).sample(paths, limit)


def make_dataloader(
    split_name: str,
    manifest_path: Path,
    stats_path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    limit: Optional[int] = None,
    augment: bool = False,
) -> tuple[MicroscopyTiffDataset, DataLoader]:
    tif_paths = get_tif_paths_for_split(split_name, manifest_path)
    selected_paths = sample_paths(tif_paths, limit=limit, seed=seed)
    dataset = MicroscopyTiffDataset(
        tif_paths=selected_paths,
        stats=stats_path,
        joint_transform=RandomFlipRotate() if augment else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def build_model(base_channels: int) -> nn.Module:
    return TinyUNet(
        in_channels=len(get_input_indices()),
        out_channels=1,
        base_channels=base_channels,
    )


def compute_batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM expects matching shapes, got {tuple(pred.shape)} vs {tuple(target.shape)}.")

    padding = window_size // 2
    data_range = target.amax(dim=(1, 2, 3), keepdim=True) - target.amin(dim=(1, 2, 3), keepdim=True)
    data_range = torch.clamp(data_range, min=1e-6)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_pred = torch.nn.functional.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=padding)
    mu_target = torch.nn.functional.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padding)

    sigma_pred = (
        torch.nn.functional.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=padding)
        - mu_pred * mu_pred
    )
    sigma_target = (
        torch.nn.functional.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padding)
        - mu_target * mu_target
    )
    sigma_cross = (
        torch.nn.functional.avg_pool2d(pred * target, kernel_size=window_size, stride=1, padding=padding)
        - mu_pred * mu_target
    )

    numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred * mu_pred + mu_target * mu_target + c1) * (sigma_pred + sigma_target + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-12)
    return ssim_map.mean(dim=(1, 2, 3)).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Train {epoch:02d}/{total_epochs}", leave=False)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite loss during training: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(train_l1=f"{loss.item():.4f}")

    if total_samples == 0:
        raise RuntimeError("Training loader produced zero samples.")
    return running_loss / total_samples


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    l1_loss: nn.Module,
    mse_loss: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Val   {epoch:02d}/{total_epochs}", leave=False)
    with torch.no_grad():
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            batch_l1 = l1_loss(pred, y)
            batch_mse = mse_loss(pred, y)
            batch_ssim = compute_batch_ssim(pred, y)

            if not torch.isfinite(batch_l1) or not torch.isfinite(batch_mse) or not torch.isfinite(batch_ssim):
                raise RuntimeError("Encountered non-finite validation metric.")

            batch_size = x.shape[0]
            total_l1 += batch_l1.item() * batch_size
            total_mse += batch_mse.item() * batch_size
            total_ssim += batch_ssim.item() * batch_size
            total_samples += batch_size
            progress.set_postfix(val_l1=f"{batch_l1.item():.4f}", val_ssim=f"{batch_ssim.item():.4f}")

    if total_samples == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    return {
        "val_l1": total_l1 / total_samples,
        "val_mse": total_mse / total_samples,
        "val_ssim": total_ssim / total_samples,
    }


def build_run_dir(output_root: Path, run_name: Optional[str]) -> Path:
    resolved_output_root = resolve_repo_path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    final_run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolved_output_root / final_run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_l1: float,
    run_config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_l1": best_val_l1,
            "config": run_config,
        },
        path,
    )


def init_wandb(args: argparse.Namespace, run_dir: Path, run_config: dict[str, Any]):
    if args.no_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed in the active environment. "
            "Install it or rerun with --no-wandb."
        ) from exc

    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    return wandb.init(
        project=args.wandb_project,
        name=run_dir.name,
        config=run_config,
        dir=str(wandb_dir),
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    set_task_by_names(input_names=get_input_names(), target_name=args.target)

    manifest_path = resolve_repo_path(args.manifest)
    stats_path = resolve_repo_path(args.stats)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    device = resolve_device(args.device)
    train_dataset, train_loader = make_dataloader(
        split_name="train",
        manifest_path=manifest_path,
        stats_path=stats_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        limit=args.train_limit,
        augment=args.augment,
    )
    val_dataset, val_loader = make_dataloader(
        split_name="val",
        manifest_path=manifest_path,
        stats_path=stats_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed + 1,
        limit=args.val_limit,
    )

    model = build_model(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    train_loss_fn = nn.L1Loss()
    val_l1_loss = nn.L1Loss()
    val_mse_loss = nn.MSELoss()

    run_dir = build_run_dir(args.output_dir, args.run_name)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoints_dir = run_dir / "checkpoints"

    run_config = {
        "manifest": str(manifest_path),
        "stats": str(stats_path),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "base_channels": args.base_channels,
        "augment": args.augment,
        "model_name": "TinyUNet",
        "loss_name": "L1Loss",
        "metric_names": ["L1", "MSE", "SSIM"],
        "input_indices": get_input_indices(),
        "target_index": get_target_index(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "wandb_enabled": not args.no_wandb,
        "wandb_project": args.wandb_project,
    }
    save_json(run_dir / "config.json", run_config)

    wandb_run = init_wandb(args, run_dir, run_config)

    print(f"Run directory: {run_dir}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Input indices: {get_input_indices()}")
    print(f"Target index: {get_target_index()}")
    print(f"Trainable parameters: {count_parameters(model)}")

    best_val_l1 = float("inf")
    best_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_l1 = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=train_loss_fn,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                l1_loss=val_l1_loss,
                mse_loss=val_mse_loss,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
            )

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_l1=train_l1,
                val_l1=val_metrics["val_l1"],
                val_mse=val_metrics["val_mse"],
                val_ssim=val_metrics["val_ssim"],
                lr=optimizer.param_groups[0]["lr"],
            )
            payload = asdict(epoch_metrics)
            append_jsonl(metrics_path, payload)
            print(
                f"Epoch {epoch:02d}/{args.epochs}: "
                f"train_l1={epoch_metrics.train_l1:.6f}, "
                f"val_l1={epoch_metrics.val_l1:.6f}, "
                f"val_mse={epoch_metrics.val_mse:.6f}, "
                f"val_ssim={epoch_metrics.val_ssim:.6f}, "
                f"lr={epoch_metrics.lr:.6f}"
            )

            save_checkpoint(
                checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_l1=min(best_val_l1, epoch_metrics.val_l1),
                run_config=run_config,
            )

            scheduler.step(epoch_metrics.val_l1)

            if epoch_metrics.val_l1 < best_val_l1:
                best_val_l1 = epoch_metrics.val_l1
                best_epoch = epoch
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_l1=best_val_l1,
                    run_config=run_config,
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch_metrics.epoch,
                        "train/l1": epoch_metrics.train_l1,
                        "val/l1": epoch_metrics.val_l1,
                        "val/mse": epoch_metrics.val_mse,
                        "val/ssim": epoch_metrics.val_ssim,
                        "lr": epoch_metrics.lr,
                        "best_val_l1": best_val_l1,
                        "best_epoch": best_epoch,
                    }
                )
    finally:
        if wandb_run is not None:
            wandb_run.summary["best_val_l1"] = best_val_l1
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_checkpoint"] = str(checkpoints_dir / "best.pt")
            wandb_run.summary["last_checkpoint"] = str(checkpoints_dir / "last.pt")
            wandb_run.finish()

    print(f"Best val_l1: {best_val_l1:.6f} at epoch {best_epoch}")
    print(f"Best checkpoint: {checkpoints_dir / 'best.pt'}")
    print(f"Last checkpoint: {checkpoints_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
