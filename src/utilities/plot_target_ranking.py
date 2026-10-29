#!/usr/bin/env python3
"""Plot all targets ranked by cell-level Pearson r.

Reads <runs-dir>/*/summary.json (written by evaluate.py) and draws one bar per
target: the Pearson r of per-cell nuclear-median intensity, predicted vs.
measured, across the test cells. Only per-target aggregates are plotted - no
images.

Usage:
  python src/utilities/plot_target_ranking.py --runs-dir local_runs/all_targets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = Path("local_runs/all_targets")
DEFAULT_OUTPUT = Path("docs/figures/target_ranking_nuc_median_pearson.png")
CORRELATION_KEY = "nuc_median"
THRESHOLD = 0.80

# Light chart surface, ink, and two categorical slots (validated for CVD separation).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
PROTEIN_COLOR = "#2a78d6"
DNA_COLOR = "#eb6834"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot targets ranked by nuclear-median Pearson r.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory with one <target>_<jobid>/summary.json per target. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path. Default: %(default)s",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_rows(runs_dir: Path) -> tuple[list[tuple[str, float]], int]:
    rows = []
    cell_counts = set()
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        corr = summary["cell_correlations"][CORRELATION_KEY]
        rows.append((summary["target_name"], float(corr["pearson"])))
        cell_counts.add(int(corr["n"]))
    if not rows:
        raise FileNotFoundError(f"No */summary.json found under {runs_dir}")
    if len(cell_counts) != 1:
        raise ValueError(f"Targets were evaluated on different cell counts: {sorted(cell_counts)}")
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows, cell_counts.pop()


def plot(rows: list[tuple[str, float]], n_cells: int, output_path: Path) -> None:
    names = [name for name, _ in rows]
    values = [value for _, value in rows]
    is_dna = [name.startswith("DNA_") for name in names]
    n_above = sum(value > THRESHOLD for value in values)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.bar(
        range(len(values)),
        values,
        width=0.72,
        color=[DNA_COLOR if dna else PROTEIN_COLOR for dna in is_dna],
        zorder=3,
    )
    ax.axhline(THRESHOLD, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
    ax.text(
        len(values) - 0.4,
        THRESHOLD + 0.012,
        f"r = {THRESHOLD:.2f}  ({n_above} of {len(values)} targets above)",
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
        zorder=5,
    )

    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=90, fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=9, colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_ylabel("Pearson r (predicted vs. measured)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "Cell-level accuracy per target: nuclear-median intensity, "
        f"test set ({n_cells} cells)",
        loc="left",
        fontsize=12,
        color=INK,
        pad=12,
    )
    ax.legend(
        handles=[
            Patch(color=PROTEIN_COLOR, label=f"Protein targets ({len(values) - sum(is_dna)})"),
            Patch(color=DNA_COLOR, label=f"DNA-stain channels ({sum(is_dna)})"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    runs_dir = resolve_repo_path(args.runs_dir)
    output_path = resolve_repo_path(args.output)
    rows, n_cells = load_rows(runs_dir)
    plot(rows, n_cells, output_path)
    print(f"Plotted {len(rows)} targets ({n_cells} test cells each) to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
