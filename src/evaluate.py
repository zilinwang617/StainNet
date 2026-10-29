#!/usr/bin/env python3
"""Evaluate a trained TinyUNet checkpoint on val or test split.

Outputs (under <run_dir>/eval/<split>/ by default):
  - <stem>_eval.tif         per-image stack, shape (4, H, W) =
                              [input_PCNA, pred, gt, diff]
                              all in DE-NORMALIZED units (matching raw TIFFs).
  - metrics.csv             per-image L1/MSE/SSIM (in normalized space, same
                              as the training-time validation loop), sorted by
                              val_ssim descending.
  - summary.json            split-level mean/median/std for each metric, plus
                              checkpoint metadata.

Usage:
  python evaluate.py --checkpoint runs/Akt_bc64/checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
from scipy import stats as scipy_stats
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    MicroscopyTiffDataset,
    get_tif_paths_for_split,
    load_channel_stats,
)
from models import TinyUNet  # noqa: E402
from utilities.mapping import index_to_name, set_task_by_names  # noqa: E402

# Reuse the same SSIM implementation training uses, so metrics are comparable.
from train import compute_batch_ssim  # noqa: E402


# Mask channel indices (same as scripts/inspect_mask.py).
NUC_CH  = 55
CYTO_CH = 57


def _get_center_cell_id(nuc_mask: np.ndarray) -> int | None:
    """Strategy A (center pixel) with Strategy B (closest centroid) as fallback."""
    H, W = nuc_mask.shape
    val = int(nuc_mask[H // 2, W // 2])
    if val > 0:
        return val
    ids = [int(v) for v in np.unique(nuc_mask) if v > 0]
    if not ids:
        return None
    img_cy, img_cx = H / 2.0, W / 2.0
    best_id, best_dist = None, float("inf")
    for cid in ids:
        ys, xs = np.where(nuc_mask == cid)
        dy = ys.mean() - img_cy
        dx = xs.mean() - img_cx
        dist = (dy * dy + dx * dx) ** 0.5
        if dist < best_dist:
            best_dist, best_id = dist, cid
    return best_id


def _extract_cell_intensity(
    pred_raw: np.ndarray,
    gt_raw: np.ndarray,
    mask: np.ndarray,
    cell_id: int | None,
) -> dict:
    nan4 = {"pred_mean": np.nan, "gt_mean": np.nan,
            "pred_median": np.nan, "gt_median": np.nan}
    if cell_id is None:
        return nan4
    region = mask == cell_id
    if not region.any():
        return nan4
    return {
        "pred_mean":   float(pred_raw[region].mean()),
        "gt_mean":     float(gt_raw[region].mean()),
        "pred_median": float(np.median(pred_raw[region])),
        "gt_median":   float(np.median(gt_raw[region])),
    }


def _corr_block(pred_vals: np.ndarray, gt_vals: np.ndarray) -> dict:
    """Pearson + Spearman for a pair of 1-D arrays (NaN rows already dropped)."""
    n = len(pred_vals)
    if n < 3:
        return {"pearson": None, "spearman": None, "n": n}
    pearson_r,  _ = scipy_stats.pearsonr(pred_vals, gt_vals)
    spearman_r, _ = scipy_stats.spearmanr(pred_vals, gt_vals)
    return {"pearson": float(pearson_r), "spearman": float(spearman_r), "n": n}


# Channel index used as the spatial reference layer in the eval TIFF.
# PCNA-mTq2 is index 22 in CHANNEL_NAMES; it is the most informative input
# channel for visual orientation (cell density, nuclei outlines).
PCNA_INDEX = 22

# Fallback paths used when the checkpoint's recorded paths don't exist.
# (Training writes node-local scratch paths like $SCRATCH/data/... into the
# checkpoint; those paths get cleaned up when the training job exits.)
DEFAULT_MANIFEST_FALLBACK = REPO_ROOT / "data" / "splits" / "stacked_raw_split.json"
DEFAULT_STATS_FALLBACK = REPO_ROOT / "data" / "stats" / "stacked_raw_stats_train.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TinyUNet checkpoint on val/test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a .pt checkpoint (typically runs/<run>/checkpoints/best.pt).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Which split to evaluate. Defaults to 'test'.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the split manifest. Defaults to the one in the checkpoint config.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Override the normalization stats. Defaults to the one in the checkpoint config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per-image TIFFs + metrics. Defaults to <run_dir>/eval/<split>/.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="cpu",
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


def resolve_path(
    value: Optional[Path],
    config_fallback: str,
    repo_fallback: Path,
    label: str,
) -> Path:
    """Pick a path with the priority CLI > config > repo default.

    The checkpoint config records the manifest/stats paths from training time,
    but training runs on node-local scratch storage that gets wiped
    when the job exits. So fall back to the repo's data directory if the
    config-recorded path no longer exists.
    """
    if value is not None:
        return value.resolve()
    config_path = Path(config_fallback)
    if config_path.exists():
        return config_path.resolve()
    print(f"  [{label}] config path missing ({config_path}); using repo default")
    return repo_fallback.resolve()


def main() -> int:
    args = parse_args()

    # --- Load checkpoint ---
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    manifest_path = resolve_path(
        args.manifest, config["manifest"], DEFAULT_MANIFEST_FALLBACK, "manifest",
    )
    stats_path = resolve_path(
        args.stats, config["stats"], DEFAULT_STATS_FALLBACK, "stats",
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    # --- Restore mapping so dataset reads the same channels as training ---
    input_indices = list(config["input_indices"])
    target_index = int(config["target_index"])
    input_names = [index_to_name(i) for i in input_indices]
    target_name = index_to_name(target_index)
    set_task_by_names(input_names=input_names, target_name=target_name)

    if PCNA_INDEX not in input_indices:
        raise ValueError(
            f"PCNA channel (index {PCNA_INDEX}) is not in input_indices {input_indices}; "
            "cannot extract it for visualization. Either change the model's inputs or "
            "edit PCNA_INDEX in this script."
        )
    pcna_input_pos = input_indices.index(PCNA_INDEX)

    # --- Build model and load weights ---
    model = TinyUNet(
        in_channels=len(input_indices),
        out_channels=1,
        base_channels=int(config["base_channels"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()

    # --- Stats for de-normalization (visualization only; metrics use normalized space) ---
    stats = load_channel_stats(stats_path)
    target_mean = float(stats["means"][target_index])
    target_std = float(stats["stds"][target_index])
    pcna_mean = float(stats["means"][PCNA_INDEX])
    pcna_std = float(stats["stds"][PCNA_INDEX])

    # --- Dataset + DataLoader for the requested split ---
    tif_paths = get_tif_paths_for_split(args.split, manifest_path)
    dataset = MicroscopyTiffDataset(tif_paths=tif_paths, stats=stats_path)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # --- Output directory ---
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        run_dir = ckpt_path.parent.parent  # runs/<run_name>/
        out_dir = (run_dir / "eval" / args.split).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Split:      {args.split} ({len(dataset)} samples)")
    print(f"Target:     {target_name} (index {target_index})")
    print(f"Manifest:   {manifest_path}")
    print(f"Stats:      {stats_path}")
    print(f"Device:     {device}")
    print(f"Output:     {out_dir}")

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    rows = []
    sample_idx = 0
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred_batch = model(x_batch)  # (B, 1, H, W)

            batch_size = pred_batch.shape[0]
            for b in range(batch_size):
                # Per-sample metrics in normalized space (matches training val).
                pred_b = pred_batch[b : b + 1]
                y_b = y_batch[b : b + 1]
                l1 = l1_loss(pred_b, y_b).item()
                mse = mse_loss(pred_b, y_b).item()
                ssim = compute_batch_ssim(pred_b, y_b).item()

                # De-normalize for visualization layers.
                pred_norm = pred_b[0, 0].cpu().numpy()
                gt_norm = y_b[0, 0].cpu().numpy()
                pcna_norm = x_batch[b, pcna_input_pos].cpu().numpy()

                pred_raw = pred_norm * target_std + target_mean
                gt_raw = gt_norm * target_std + target_mean
                pcna_raw = pcna_norm * pcna_std + pcna_mean
                diff_raw = pred_raw - gt_raw  # signed; same units as target

                # --- Per-cell intensity (center cell only) ---
                raw_tiff = tifffile.imread(str(tif_paths[sample_idx]))
                nuc_mask  = raw_tiff[NUC_CH]
                cyto_mask = raw_tiff[CYTO_CH]
                center_id = _get_center_cell_id(nuc_mask)
                nuc_vals  = _extract_cell_intensity(pred_raw, gt_raw, nuc_mask,  center_id)
                cyto_vals = _extract_cell_intensity(pred_raw, gt_raw, cyto_mask, center_id)

                stack = np.stack(
                    [pcna_raw, pred_raw, gt_raw, diff_raw],
                    axis=0,
                ).astype(np.float32)

                file_id = tif_paths[sample_idx].stem
                tiff_out = images_dir / f"{file_id}_eval.tif"
                # ImageJ-style metadata so napari can label the channels.
                tifffile.imwrite(
                    str(tiff_out),
                    stack,
                    imagej=True,
                    metadata={"axes": "CYX", "Labels": ["PCNA", "pred", "gt", "diff"]},
                )

                rows.append(
                    {
                        "file_id":          file_id,
                        "val_l1":           l1,
                        "val_mse":          mse,
                        "val_ssim":         ssim,
                        "nuc_pred_mean":    nuc_vals["pred_mean"],
                        "nuc_gt_mean":      nuc_vals["gt_mean"],
                        "nuc_pred_median":  nuc_vals["pred_median"],
                        "nuc_gt_median":    nuc_vals["gt_median"],
                        "cyto_pred_mean":   cyto_vals["pred_mean"],
                        "cyto_gt_mean":     cyto_vals["gt_mean"],
                        "cyto_pred_median": cyto_vals["pred_median"],
                        "cyto_gt_median":   cyto_vals["gt_median"],
                    }
                )
                sample_idx += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  processed {sample_idx}/{len(dataset)}")

    if sample_idx != len(dataset):
        raise RuntimeError(
            f"Sample count mismatch: wrote {sample_idx}, expected {len(dataset)}."
        )

    # --- metrics.csv (sorted by ssim desc so the best samples come first) ---
    rows.sort(key=lambda r: r["val_ssim"], reverse=True)
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_id", "val_l1", "val_mse", "val_ssim",
            "nuc_pred_mean", "nuc_gt_mean", "nuc_pred_median", "nuc_gt_median",
            "cyto_pred_mean", "cyto_gt_mean", "cyto_pred_median", "cyto_gt_median",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # --- summary.json ---
    l1_arr = np.array([r["val_l1"] for r in rows], dtype=np.float64)
    mse_arr = np.array([r["val_mse"] for r in rows], dtype=np.float64)
    ssim_arr = np.array([r["val_ssim"] for r in rows], dtype=np.float64)

    def stats_block(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }

    summary = {
        "checkpoint": str(ckpt_path),
        "run_name": config.get("run_name"),
        "split": args.split,
        "target_name": target_name,
        "target_index": target_index,
        "base_channels": int(config["base_channels"]),
        "num_samples": int(len(dataset)),
        "metrics": {
            "val_l1": stats_block(l1_arr),
            "val_mse": stats_block(mse_arr),
            "val_ssim": stats_block(ssim_arr),
        },
    }
    # --- Cross-sample cell-level correlations ---
    def _valid_pairs(pred_key: str, gt_key: str):
        pairs = [(r[pred_key], r[gt_key]) for r in rows
                 if not (np.isnan(r[pred_key]) or np.isnan(r[gt_key]))]
        if not pairs:
            return np.array([]), np.array([])
        p, g = zip(*pairs)
        return np.array(p), np.array(g)

    cell_corr = {}
    for label, pk, gk in [
        ("nuc_mean",    "nuc_pred_mean",    "nuc_gt_mean"),
        ("nuc_median",  "nuc_pred_median",  "nuc_gt_median"),
        ("cyto_mean",   "cyto_pred_mean",   "cyto_gt_mean"),
        ("cyto_median", "cyto_pred_median", "cyto_gt_median"),
    ]:
        pv, gv = _valid_pairs(pk, gk)
        cell_corr[label] = _corr_block(pv, gv)

    summary["cell_correlations"] = cell_corr

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print()
    print(f"val_l1   mean={summary['metrics']['val_l1']['mean']:.4f} "
          f"median={summary['metrics']['val_l1']['median']:.4f}")
    print(f"val_mse  mean={summary['metrics']['val_mse']['mean']:.4f} "
          f"median={summary['metrics']['val_mse']['median']:.4f}")
    print(f"val_ssim mean={summary['metrics']['val_ssim']['mean']:.4f} "
          f"median={summary['metrics']['val_ssim']['median']:.4f}")
    print()
    print(f"Wrote {sample_idx} TIFFs to: {out_dir}")
    print(f"Metrics CSV:   {metrics_path}")
    print(f"Summary JSON:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
