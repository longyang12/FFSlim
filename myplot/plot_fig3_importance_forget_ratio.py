import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot_style import FORGET_COLORS, apply_eurosys_style


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig3.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def short_method_name(full_name: str) -> str:
    if full_name.endswith("DFR"):
        return "DFR"
    if full_name.endswith("SFR"):
        return "SFR"
    if full_name.endswith("FFR"):
        return "FFR"
    return full_name


def _split_cols(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    cols = [c.strip() for c in re.split(r"\t+|\s{2,}", line) if c.strip()]
    if len(cols) <= 1:
        cols = line.split()
    return cols


def _to_float_percent(value: str) -> float:
    return float(value.strip().replace("%", ""))


def parse_fig3(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    current_fmt = None
    methods: List[str] = []
    idx = 0

    while idx < len(lines):
        s = lines[idx].strip()
        sec = re.match(r"^#+\s*(FFSlim|TDP)\s*#+$", s)
        if sec:
            current_fmt = sec.group(1)
            data[current_fmt] = {}
            idx += 1
            continue

        if current_fmt is None:
            idx += 1
            continue

        if s.startswith("Workload"):
            cols = _split_cols(lines[idx])
            methods = cols[1:]
            idx += 1
            continue

        if "Important forget ratio" in s or not s:
            idx += 1
            continue

        cols = _split_cols(lines[idx])
        if methods and len(cols) >= 1 + len(methods):
            workload = cols[0]
            values = cols[1 : 1 + len(methods)]
            data[current_fmt][workload] = {
                short_method_name(m): _to_float_percent(v) for m, v in zip(methods, values)
            }

        idx += 1

    for fmt in ["FFSlim", "TDP"]:
        if fmt not in data:
            raise ValueError(f"Missing section: {fmt}")
        if not data[fmt]:
            raise ValueError(f"No rows found in section: {fmt}")

    return data


def plot_fig3(data: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_eurosys_style()

    formats = ["FFSlim", "TDP"]
    workloads = ["Flickr30k", "F30-E", "MSCOCO", "Clotho", "MSRVTT", "VQA2", "GQA"]

    methods = ["DFR", "SFR", "FFR"]

    colors = FORGET_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    x = np.arange(len(workloads), dtype=float)
    bar_width = 0.23
    offsets = np.linspace(-bar_width, bar_width, len(methods))

    ymax = 0.0
    for fmt in formats:
        for wl in workloads:
            ymax = max(ymax, max(data[fmt][wl][m] for m in methods))

    for ax, fmt in zip(axes, formats):
        for k, method in enumerate(methods):
            ys = [data[fmt][wl][method] for wl in workloads]
            ax.bar(
                x + offsets[k],
                ys,
                width=bar_width,
                color=colors.get(method, "#999999"),
                edgecolor="#1F1F1F",
                linewidth=0.55,
                alpha=0.9,
                label=method,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(workloads, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax * 1.16)
        ax.set_xlabel("")
        ax.set_title(f"{fmt} Format")

    axes[0].set_ylabel("Important Forget Ratio (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0.01, 0.08, 1.0, 0.94])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "fig3_importance_forget_ratio.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig3_importance_forget_ratio.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    data = parse_fig3(INPUT_FILE)
    plot_fig3(data)


if __name__ == "__main__":
    main()
