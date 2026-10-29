#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_DATA_DIR = Path("data/stacked_raw")
DEFAULT_OUTPUT = Path("data/splits/stacked_raw_split.json")
DEFAULT_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/val/test split manifest for TIFF samples."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing .tif files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON manifest. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic shuffling. Default: %(default)s",
    )
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def to_repo_relative_posix(path: Path, repo_root: Path) -> str:
    try:
        relative_path = path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the repository root: {path}") from exc
    return relative_path.as_posix()


def collect_tif_paths(data_dir: Path, repo_root: Path) -> list[str]:
    tif_paths = sorted(path.resolve() for path in data_dir.glob("*.tif"))
    relative_paths = [to_repo_relative_posix(path, repo_root) for path in tif_paths]

    if not relative_paths:
        raise ValueError(f"No TIFF files found in {data_dir}")

    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Duplicate TIFF paths detected while building the split manifest.")

    return relative_paths


def compute_split_counts(total: int) -> tuple[int, int, int]:
    train_count = round(total * TRAIN_RATIO)
    val_count = round(total * VAL_RATIO)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def build_manifest(tif_paths: list[str], seed: int, data_dir: Path, repo_root: Path) -> dict:
    shuffled_paths = list(tif_paths)
    random.Random(seed).shuffle(shuffled_paths)

    train_count, val_count, test_count = compute_split_counts(len(shuffled_paths))
    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count : train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count :]

    validate_split(train_paths, val_paths, test_paths, len(shuffled_paths))

    return {
        "data_dir": to_repo_relative_posix(data_dir.resolve(), repo_root),
        "seed": seed,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "total": len(shuffled_paths),
        },
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }


def validate_split(
    train_paths: list[str], val_paths: list[str], test_paths: list[str], total_count: int
) -> None:
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    if train_set & val_set:
        raise ValueError("Train and val splits overlap.")
    if train_set & test_set:
        raise ValueError("Train and test splits overlap.")
    if val_set & test_set:
        raise ValueError("Val and test splits overlap.")

    combined_count = len(train_paths) + len(val_paths) + len(test_paths)
    if combined_count != total_count:
        raise ValueError(
            f"Split counts do not add up to total. Expected {total_count}, got {combined_count}."
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]  # src/split.py -> repo root
    data_dir = resolve_path(args.data_dir, repo_root).resolve()
    output_path = resolve_path(args.output, repo_root).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")
    if output_path.exists():
        raise FileExistsError(f"Output manifest already exists: {output_path}")

    tif_paths = collect_tif_paths(data_dir, repo_root)
    manifest = build_manifest(tif_paths, args.seed, data_dir, repo_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    counts = manifest["counts"]
    print(f"Saved split manifest to: {output_path}")
    print(f"Total samples: {counts['total']}")
    print(f"Train: {counts['train']}")
    print(f"Val: {counts['val']}")
    print(f"Test: {counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
