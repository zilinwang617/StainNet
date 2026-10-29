import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

try:
    from utilities.mapping import (
        get_channel_names,
        get_input_indices as get_mapping_input_indices,
        get_num_channels,
        get_target_index as get_mapping_target_index,
        validate_mapping,
    )
except ImportError:
    from .utilities.mapping import (
        get_channel_names,
        get_input_indices as get_mapping_input_indices,
        get_num_channels,
        get_target_index as get_mapping_target_index,
        validate_mapping,
    )

VALID_SPLIT_NAMES = ("train", "val", "test")
REQUIRED_MANIFEST_KEYS = ("data_dir", "counts", "train", "val", "test")
DEFAULT_STD_EPSILON = 1e-12


def get_default_input_indices() -> List[int]:
    """Return the current default input-channel indices from the mapping config."""
    validate_mapping()
    return list(get_mapping_input_indices())


def get_default_target_index() -> int:
    """Return the current default target-channel index from the mapping config."""
    validate_mapping()
    return int(get_mapping_target_index())


def load_channel_stats(
    stats_source: Union[str, Path, Dict[str, Any]],
    std_epsilon: float = DEFAULT_STD_EPSILON,
) -> Dict[str, Any]:
    """Load and validate channel statistics from a path or parsed JSON payload."""
    if isinstance(stats_source, (str, Path)):
        stats_path = Path(stats_source).resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as f:
            stats_payload = json.load(f)
    elif isinstance(stats_source, dict):
        stats_payload = stats_source
    else:
        raise TypeError(
            "stats_source must be a path or a parsed dictionary, "
            f"got {type(stats_source).__name__}."
        )

    channel_entries = stats_payload.get("channels")
    if not isinstance(channel_entries, list):
        raise ValueError("Stats payload must contain a 'channels' list.")

    expected_names = get_channel_names()
    expected_num_channels = get_num_channels()
    if len(channel_entries) != expected_num_channels:
        raise ValueError(
            f"Stats payload must contain {expected_num_channels} channel entries, "
            f"got {len(channel_entries)}."
        )

    means = np.empty(expected_num_channels, dtype=np.float32)
    stds = np.empty(expected_num_channels, dtype=np.float32)

    for expected_index, expected_name in enumerate(expected_names):
        entry = channel_entries[expected_index]
        if not isinstance(entry, dict):
            raise ValueError(f"Stats entry at index {expected_index} must be a dictionary.")

        entry_index = int(entry["index"])
        entry_name = str(entry["name"])
        if entry_index != expected_index:
            raise ValueError(
                f"Stats entry index mismatch at position {expected_index}: "
                f"expected {expected_index}, got {entry_index}."
            )
        if entry_name != expected_name:
            raise ValueError(
                f"Stats entry name mismatch at index {expected_index}: "
                f"expected '{expected_name}', got '{entry_name}'."
            )

        mean_value = float(entry["mean"])
        std_value = float(entry["std"])
        safe_std_value = float(entry.get("safe_std", std_value))
        if safe_std_value <= std_epsilon:
            safe_std_value = 1.0

        means[expected_index] = mean_value
        stds[expected_index] = safe_std_value

    return {
        "metadata": {
            key: value for key, value in stats_payload.items() if key != "channels"
        },
        "means": means,
        "stds": stds,
    }


def _validate_manifest_keys(manifest: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Split manifest is missing required keys: {missing}.")

    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ValueError("Split manifest field 'counts' must be a dictionary.")

    missing_count_keys = [key for key in (*VALID_SPLIT_NAMES, "total") if key not in counts]
    if missing_count_keys:
        missing = ", ".join(missing_count_keys)
        raise ValueError(f"Split manifest counts are missing required keys: {missing}.")


def _validate_manifest_counts_and_overlap(manifest: Dict[str, Any]) -> None:
    split_sets = {}
    total_count = 0

    for split_name in VALID_SPLIT_NAMES:
        split_entries = manifest[split_name]
        if not isinstance(split_entries, list):
            raise ValueError(f"Split manifest field '{split_name}' must be a list.")

        if len(split_entries) != manifest["counts"][split_name]:
            raise ValueError(
                f"Split manifest count mismatch for '{split_name}': "
                f"expected {manifest['counts'][split_name]}, got {len(split_entries)}."
            )

        split_set = set(split_entries)
        if len(split_set) != len(split_entries):
            raise ValueError(f"Split manifest contains duplicate entries in '{split_name}'.")

        split_sets[split_name] = split_set
        total_count += len(split_entries)

    if split_sets["train"] & split_sets["val"]:
        raise ValueError("Split manifest has overlapping entries between train and val.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Split manifest has overlapping entries between train and test.")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("Split manifest has overlapping entries between val and test.")

    if total_count != manifest["counts"]["total"]:
        raise ValueError(
            f"Split manifest total count mismatch: expected {manifest['counts']['total']}, "
            f"got {total_count}."
        )


def _resolve_manifest_entry(entry: str, manifest_path: Path) -> Path:
    if not isinstance(entry, str) or not entry:
        raise ValueError("Split manifest entries must be non-empty strings.")

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

    searched_from = ", ".join(str(root) for root in candidate_roots)
    raise FileNotFoundError(
        f"Could not resolve manifest entry '{entry}' from manifest anchor(s): {searched_from}."
    )


def load_split_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a train/val/test split manifest."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Split manifest root must be a JSON object.")

    _validate_manifest_keys(manifest)
    _validate_manifest_counts_and_overlap(manifest)

    for split_name in VALID_SPLIT_NAMES:
        for entry in manifest[split_name]:
            _resolve_manifest_entry(entry, manifest_path)

    return manifest


def get_tif_paths_for_split(split_name: str, manifest_path: Union[str, Path]) -> List[Path]:
    """Resolve filesystem paths for one split from the JSON manifest."""
    if split_name not in VALID_SPLIT_NAMES:
        valid_names = ", ".join(VALID_SPLIT_NAMES)
        raise ValueError(f"split_name must be one of: {valid_names}. Got '{split_name}'.")

    manifest_path = Path(manifest_path).resolve()
    manifest = load_split_manifest(manifest_path)
    return [_resolve_manifest_entry(entry, manifest_path) for entry in manifest[split_name]]


def build_dataset_for_split(
    split_name: str,
    manifest_path: Union[str, Path],
    target_index: Optional[int] = None,
    transform_x: Optional[Callable] = None,
    transform_y: Optional[Callable] = None,
    stats: Optional[Union[str, Path, Dict[str, Any]]] = None,
    normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> "MicroscopyTiffDataset":
    """Build a dataset instance for one split defined in the JSON manifest."""
    tif_paths = get_tif_paths_for_split(split_name, manifest_path)
    return MicroscopyTiffDataset(
        tif_paths=tif_paths,
        target_index=target_index,
        transform_x=transform_x,
        transform_y=transform_y,
        stats=stats,
        normalize_fn=normalize_fn,
    )


class MicroscopyTiffDataset(Dataset):
    """
    Minimal PyTorch dataset for stacked TIFF microscopy data.

    Each TIFF is expected to have shape (59, H, W), where axis 0 is channel.
    """

    def __init__(
        self,
        tif_paths: Sequence[Union[str, Path]],
        target_index: Optional[int] = None,
        transform_x: Optional[Callable] = None,
        transform_y: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        stats: Optional[Union[str, Path, Dict[str, Any]]] = None,
        normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if not tif_paths:
            raise ValueError("tif_paths is empty. Provide at least one TIFF path.")
        if stats is not None and normalize_fn is not None:
            raise ValueError("Provide either stats or normalize_fn, not both.")

        self.tif_paths = [Path(p) for p in tif_paths]
        self.input_indices = get_default_input_indices()
        self.target_index = (
            get_default_target_index() if target_index is None else int(target_index)
        )
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.joint_transform = joint_transform
        self.stats = load_channel_stats(stats) if stats is not None else None
        self.normalize_fn = normalize_fn

        num_channels = get_num_channels()
        if not (0 <= self.target_index < num_channels):
            raise ValueError(
                f"target_index must be in [0, {num_channels - 1}], got {self.target_index}."
            )

    def _normalize_by_stats(self, channel_image: np.ndarray, channel_index: int) -> np.ndarray:
        """Normalize one channel using dataset-level mean/std statistics."""
        assert self.stats is not None  # Guaranteed by caller.
        mean_value = float(self.stats["means"][channel_index])
        std_value = float(self.stats["stds"][channel_index])
        return ((channel_image - mean_value) / std_value).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.tif_paths)

    def _normalize_channel(self, channel_image: np.ndarray, channel_name: str) -> np.ndarray:
        """Apply normalize_fn to one channel with basic shape/type checks."""
        normalized = self.normalize_fn(channel_image)
        normalized = np.asarray(normalized, dtype=np.float32)
        if normalized.shape != channel_image.shape:
            raise ValueError(
                f"normalize_fn changed shape for {channel_name}: "
                f"expected {channel_image.shape}, got {normalized.shape}."
            )
        return normalized

    @staticmethod
    def _to_float_tensor(arr_or_tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert numpy/tensor input to torch.float32 tensor."""
        if isinstance(arr_or_tensor, torch.Tensor):
            return arr_or_tensor.to(torch.float32)
        return torch.as_tensor(arr_or_tensor, dtype=torch.float32)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        tif_path = self.tif_paths[idx]
        if not tif_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tif_path}")

        try:
            stack = tifffile.imread(str(tif_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to read TIFF file: {tif_path}") from exc

        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError(f"Expected TIFF with 3 dims (C, H, W), got shape {stack.shape}.")
        expected_num_channels = get_num_channels()
        if stack.shape[0] != expected_num_channels:
            raise ValueError(
                f"Expected channel dimension C={expected_num_channels}, "
                f"got {stack.shape[0]} for {tif_path}."
            )

        # Select fixed input channels and configurable target channel.
        x_np = stack[self.input_indices, :, :].astype(np.float32, copy=False)
        y_np = stack[self.target_index, :, :].astype(np.float32, copy=False)

        if self.stats is not None:
            x_norm = np.empty_like(x_np, dtype=np.float32)
            for i, channel_index in enumerate(self.input_indices):
                x_norm[i] = self._normalize_by_stats(x_np[i], channel_index)
            x_np = x_norm
            y_np = self._normalize_by_stats(y_np, self.target_index)
        elif self.normalize_fn is not None:
            x_norm = np.empty_like(x_np, dtype=np.float32)
            for i in range(x_np.shape[0]):
                x_norm[i] = self._normalize_channel(x_np[i], channel_name=f"x[{i}]")
            x_np = x_norm
            y_np = self._normalize_channel(y_np, channel_name="y")

        # Add channel dim so y is (1, H, W).
        y_np = y_np[None, :, :]

        x = self._to_float_tensor(x_np)
        y = self._to_float_tensor(y_np)

        if self.transform_x is not None:
            x = self._to_float_tensor(self.transform_x(x))
        if self.transform_y is not None:
            y = self._to_float_tensor(self.transform_y(y))
        if self.joint_transform is not None:
            x, y = self.joint_transform(x, y)

        # Basic shape checks to catch transform mistakes early.
        expected_input_channels = len(self.input_indices)
        if x.ndim != 3 or x.shape[0] != expected_input_channels:
            raise ValueError(
                f"x must have shape ({expected_input_channels}, H, W). Got {tuple(x.shape)}."
            )
        if y.ndim != 3 or y.shape[0] != 1:
            raise ValueError(f"y must have shape (1, H, W). Got {tuple(y.shape)}.")

        return x, y


if __name__ == "__main__":
    data_dir = Path("data/stacked_raw")
    manifest_path = Path("data/splits/stacked_raw_split.json")
    tif_paths = sorted(data_dir.glob("*.tif"))

    if not tif_paths:
        print(f"No TIFF files found in: {data_dir.resolve()}")
    else:
        raw_dataset = MicroscopyTiffDataset(
            tif_paths=tif_paths,
        )
        x, y = raw_dataset[0]
        print(f"Raw dataset samples: {len(raw_dataset)}")
        print(f"x: shape={tuple(x.shape)}, dtype={x.dtype}")
        print(f"y: shape={tuple(y.shape)}, dtype={y.dtype}")
        print(f"Default input indices: {raw_dataset.input_indices}")
        print(f"Default target index: {raw_dataset.target_index}")

        if manifest_path.exists():
            train_dataset = build_dataset_for_split(
                split_name="train",
                manifest_path=manifest_path,
            )
            print(f"Train split samples: {len(train_dataset)}")
        else:
            print(f"Split manifest not found: {manifest_path.resolve()}")
