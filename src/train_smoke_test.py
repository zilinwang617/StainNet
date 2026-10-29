#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import MicroscopyTiffDataset, get_tif_paths_for_split  # noqa: E402
from models import TinyUNet, count_parameters  # noqa: E402
from utilities.mapping import get_input_indices, get_target_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end training smoke test.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/stacked_raw_split.json"),
        help="Path to the split manifest JSON.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=32,
        help="Number of train samples to draw for the smoke test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling and training.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    return parser.parse_args()


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


def build_train_loader(
    manifest_path: Path, subset_size: int, batch_size: int, seed: int
) -> tuple[MicroscopyTiffDataset, DataLoader]:
    train_paths = get_tif_paths_for_split("train", manifest_path)
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}.")
    if len(train_paths) < subset_size:
        raise ValueError(
            f"Requested subset_size={subset_size}, but train split only has {len(train_paths)} samples."
        )

    sampled_paths = random.Random(seed).sample(train_paths, subset_size)
    dataset = MicroscopyTiffDataset(tif_paths=sampled_paths)
    effective_batch_size = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    return dataset, loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")
    return running_loss / num_batches


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = (REPO_ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    device = resolve_device(args.device)
    dataset, loader = build_train_loader(
        manifest_path=manifest_path,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = TinyUNet(in_channels=len(get_input_indices()), out_channels=1, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"Manifest: {manifest_path}")
    print(f"Subset size: {len(dataset)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Input indices: {get_input_indices()}")
    print(f"Target index: {get_target_index()}")
    print(f"Trainable parameters: {count_parameters(model)}")

    first_loss = None
    last_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        if first_loss is None:
            first_loss = epoch_loss
        last_loss = epoch_loss
        print(f"Epoch {epoch:02d}/{args.epochs}: mean_train_loss={epoch_loss:.6f}")

    if first_loss is None or last_loss is None:
        raise RuntimeError("No training epochs were executed.")

    improved = last_loss < first_loss
    print(f"First epoch loss: {first_loss:.6f}")
    print(f"Last epoch loss:  {last_loss:.6f}")
    print(f"Loss decreased:   {improved}")

    if not improved:
        raise RuntimeError("Smoke test failed: final loss did not decrease.")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
