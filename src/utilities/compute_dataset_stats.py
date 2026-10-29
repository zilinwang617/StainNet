#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities.mapping import get_channel_names, get_num_channels  # noqa: E402

STD_EPSILON = 1e-12
VALID_SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset-level per-channel mean/std statistics for stacked TIFF data."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/stacked_raw_split.json"),
        help="Path to the split manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stats"),
        help="Directory to write the stats JSON files.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_manifest_entry(entry: str, manifest_path: Path) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        resolved_entry = entry_path.resolve()
        if not resolved_entry.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {resolved_entry}")
        return resolved_entry

    candidate_roots = [manifest_path.parent, *manifest_path.parents]
    seen_candidates = set()
    for root in candidate_roots:
        candidate = (root / entry_path).resolve()
        candidate_key = str(candidate)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve manifest entry '{entry}' from {manifest_path}.")


def load_split_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Split manifest root must be a JSON object.")

    for split_name in VALID_SPLIT_NAMES:
        if split_name not in manifest:
            raise ValueError(f"Split manifest is missing '{split_name}'.")
        if not isinstance(manifest[split_name], list):
            raise ValueError(f"Split manifest field '{split_name}' must be a list.")

    return manifest


def get_tif_paths_for_split(split_name: str, manifest_path: Path) -> list[Path]:
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {VALID_SPLIT_NAMES}, got '{split_name}'.")

    manifest = load_split_manifest(manifest_path)
    return [resolve_manifest_entry(entry, manifest_path) for entry in manifest[split_name]]


def get_all_tif_paths(manifest_path: Path) -> list[Path]:
    all_paths: list[Path] = []
    for split_name in VALID_SPLIT_NAMES:
        all_paths.extend(get_tif_paths_for_split(split_name, manifest_path))
    return all_paths


def accumulate_channel_statistics(tif_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_channels = get_num_channels()
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(num_channels, dtype=np.float64)
    pixel_count = np.zeros(num_channels, dtype=np.int64)

    for file_index, tif_path in enumerate(tif_paths, start=1):
        stack = np.asarray(tifffile.imread(str(tif_path)), dtype=np.float64)
        if stack.ndim != 3:
            raise ValueError(f"Expected TIFF with 3 dims (C, H, W), got {stack.shape} for {tif_path}.")
        if stack.shape[0] != num_channels:
            raise ValueError(
                f"Expected channel dimension C={num_channels}, got {stack.shape[0]} for {tif_path}."
            )

        flattened = stack.reshape(num_channels, -1)
        channel_sum += flattened.sum(axis=1, dtype=np.float64)
        channel_sum_sq += np.square(flattened, dtype=np.float64).sum(axis=1, dtype=np.float64)
        pixel_count += flattened.shape[1]

        if file_index % 500 == 0 or file_index == len(tif_paths):
            print(f"Processed {file_index}/{len(tif_paths)} files")

    return channel_sum, channel_sum_sq, pixel_count


def build_stats_payload(
    tif_paths: list[Path],
    manifest_path: Path,
    split_scope: str,
) -> dict:
    channel_sum, channel_sum_sq, pixel_count = accumulate_channel_statistics(tif_paths)
    means = channel_sum / pixel_count
    variances = np.maximum(channel_sum_sq / pixel_count - np.square(means), 0.0)
    stds = np.sqrt(variances)
    safe_stds = np.where(stds > STD_EPSILON, stds, 1.0)
    channel_names = get_channel_names()

    channels = []
    for index, name in enumerate(channel_names):
        channels.append(
            {
                "index": index,
                "name": name,
                "mean": float(means[index]),
                "std": float(stds[index]),
                "safe_std": float(safe_stds[index]),
                "pixel_count": int(pixel_count[index]),
                "used_safe_std": bool(stds[index] <= STD_EPSILON),
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "split_scope": split_scope,
        "num_samples": len(tif_paths),
        "num_channels": get_num_channels(),
        "std_epsilon": STD_EPSILON,
        "channels": channels,
    }


def write_stats_file(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    output_dir = resolve_repo_path(args.output_dir)

    train_paths = get_tif_paths_for_split("train", manifest_path)
    all_paths = get_all_tif_paths(manifest_path)

    train_payload = build_stats_payload(train_paths, manifest_path, split_scope="train")
    all_payload = build_stats_payload(all_paths, manifest_path, split_scope="all")

    train_output = output_dir / "stacked_raw_stats_train.json"
    all_output = output_dir / "stacked_raw_stats_all.json"
    write_stats_file(train_output, train_payload)
    write_stats_file(all_output, all_payload)

    print(f"Saved train stats to: {train_output}")
    print(f"Saved all-data stats to: {all_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
