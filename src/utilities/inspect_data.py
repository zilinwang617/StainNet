#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np

try:
    import tifffile
except ImportError:
    print("请先安装 tifffile：pip install tifffile")
    sys.exit(1)


# ===== 默认数据路径（可用命令行参数覆盖）=====
DEFAULT_DATA_DIR = Path("data/stacked_raw")
DEFAULT_MAX_FILES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 TIFF 文件的形状、dtype 和各通道统计量。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="TIFF 所在目录。默认: %(default)s")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="最多检查几个文件。默认: %(default)s")
    return parser.parse_args()


def find_tiff_files(data_dir: Path):
    files = list(data_dir.rglob("*.tif")) + list(data_dir.rglob("*.tiff"))
    return sorted(set(files))


def safe_stats(arr: np.ndarray):
    return (
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.std(arr)),
    )


def guess_channel_axis(shape):
    ndim = len(shape)
    candidates = [i for i, s in enumerate(shape) if 1 <= s <= 64]

    if not candidates or ndim == 2:
        return None

    if ndim == 3:
        if 0 in candidates:
            return 0
        if 2 in candidates:
            return 2

    if ndim == 4:
        if 1 in candidates:
            return 1
        if 0 in candidates:
            return 0
        if 3 in candidates:
            return 3

    return candidates[0]


def main():
    args = parse_args()
    data_dir = args.data_dir
    max_files = args.max_files

    if not data_dir.exists():
        print(f"目录不存在: {data_dir}")
        return

    files = find_tiff_files(data_dir)
    if not files:
        print(f"在 {data_dir} 下没有找到 tif/tiff 文件。")
        return

    print(f"共找到 {len(files)} 个 TIFF 文件")
    print(f"检查前 {min(max_files, len(files))} 个文件\n")

    for i, path in enumerate(files[:max_files], start=1):
        print("=" * 80)
        print(f"[{i}] 文件: {path}")

        try:
            arr = tifffile.imread(path)
            arr = np.asarray(arr)

            gmin, gmax, gmean, gstd = safe_stats(arr)
            channel_axis = guess_channel_axis(arr.shape)

            print(f"shape: {arr.shape}")
            print(f"dtype: {arr.dtype}")
            print(f"ndim:  {arr.ndim}")
            print(f"global min/max/mean/std: {gmin:.4f} / {gmax:.4f} / {gmean:.4f} / {gstd:.4f}")
            print(f"guess channel axis: {channel_axis}")

            if channel_axis is not None:
                arr_c_first = np.moveaxis(arr, channel_axis, 0)
                print(f"guessed num_channels: {arr_c_first.shape[0]}")

                for c in range(arr_c_first.shape[0]):
                    cmin, cmax, cmean, cstd = safe_stats(arr_c_first[c])
                    print(
                        f"  channel {c:>3}: "
                        f"shape={arr_c_first[c].shape}, "
                        f"min={cmin:.4f}, max={cmax:.4f}, "
                        f"mean={cmean:.4f}, std={cstd:.4f}"
                    )
            else:
                print("无法自动判断 channel axis，请手动检查。")

        except Exception as e:
            print(f"读取失败: {e}")


if __name__ == "__main__":
    main()