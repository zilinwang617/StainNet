#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import MicroscopyTiffDataset, load_channel_stats  # noqa: E402
from models import TinyUNet  # noqa: E402
from utilities.mapping import index_to_name, set_task_by_names  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained TinyUNet checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a .pt checkpoint file (e.g. runs/my-run/checkpoints/best.pt).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .tif file paths, or a single directory containing .tif files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write predictions. Defaults to <checkpoint_dir>/../predictions/.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to stats JSON. Defaults to the path recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target protein name. Defaults to the target recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--no-tiff",
        action="store_true",
        help="Skip saving predicted TIFF files.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG visualizations (disabled by default).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_tif_paths(inputs: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        if p.is_dir():
            found = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
            if not found:
                raise FileNotFoundError(f"No .tif files found in directory: {p}")
            paths.extend(found)
        else:
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {p}")
            paths.append(p)
    return paths


def save_png(array: np.ndarray, path: Path) -> None:
    """Save a 2D float array as a viridis-colormapped PNG."""
    vmin, vmax = array.min(), array.max()
    if vmax > vmin:
        normalized = (array - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(array)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    # --- Load checkpoint ---
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    # --- Resolve stats path ---
    stats_path = args.stats.resolve() if args.stats is not None else Path(config["stats"]).resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # --- Restore mapping to match training ---
    input_indices: List[int] = config["input_indices"]
    target_index: int = config["target_index"]

    # Allow CLI override of target
    if args.target is not None:
        from utilities.mapping import name_to_index  # noqa: E402
        target_index = name_to_index(args.target)

    input_names = [index_to_name(i) for i in input_indices]
    target_name = index_to_name(target_index)
    set_task_by_names(input_names=input_names, target_name=target_name)

    # --- Build and load model ---
    base_channels: int = config["base_channels"]
    model = TinyUNet(
        in_channels=len(input_indices),
        out_channels=1,
        base_channels=base_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()

    # --- Load stats for de-normalization ---
    stats = load_channel_stats(stats_path)
    target_mean = float(stats["means"][target_index])
    target_std = float(stats["stds"][target_index])

    # --- Collect input files ---
    tif_paths = collect_tif_paths(args.input)
    print(f"Found {len(tif_paths)} input file(s).")

    # --- Output directory ---
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = (ckpt_path.parent.parent / "predictions").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving predictions to: {out_dir}")

    # --- Dataset + DataLoader ---
    dataset = MicroscopyTiffDataset(tif_paths=tif_paths, stats=stats_path)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Inference ---
    file_idx = 0
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            pred_batch = model(x_batch)  # (B, 1, H, W)

            for b in range(pred_batch.shape[0]):
                pred_norm = pred_batch[b, 0].cpu().numpy()  # (H, W), normalized
                pred_raw = pred_norm * target_std + target_mean  # de-normalize

                stem = tif_paths[file_idx].stem
                file_idx += 1

                if not args.no_tiff:
                    tiff_out = out_dir / f"{stem}_pred_{target_name}.tif"
                    tifffile.imwrite(str(tiff_out), pred_raw.astype(np.float32))

                if args.save_png:
                    png_out = out_dir / f"{stem}_pred_{target_name}.png"
                    save_png(pred_raw, png_out)

    print(f"Done. {file_idx} prediction(s) saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
