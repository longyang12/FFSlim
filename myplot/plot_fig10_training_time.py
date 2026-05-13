import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_eurosys_style, method_color, method_marker


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig10.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _split_columns(line: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.strip()) if p.strip()]
    return parts if parts else [p for p in line.split() if p]


def _to_float(value: str) -> float:
    return float(value.strip().replace(",", ""))


def _format_workload_label(workload: str) -> str:
    mapping = {
        "BLIP-Base": r"$\mathrm{BLIP}_{\mathrm{Base}}$",
        "BLIP-Large": r"$\mathrm{BLIP}_{\mathrm{Large}}$",
        "BLIP2-OPT2.7": r"$\mathrm{BLIP2}_{\mathrm{OPT2.7B}}$",
        "BLIP2-OPT6.7": r"$\mathrm{BLIP2}_{\mathrm{OPT6.7B}}$",
    }
    return mapping.get(workload, workload)


def parse_fig10(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data: Dict[str, Dict[str, Dict[str, float]]] = {"FFSlim": {}, "TDP": {}}

    current_base = ""
    methods: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#") and "FFSlim" in line:
            current_base = "FFSlim"
            methods = []
            continue

        if line.startswith("#") and "TDP" in line:
            current_base = "TDP"
            methods = []
            continue

        if not current_base:
            continue

        if "Training time" in line:
            continue

        if line.startswith("Workload"):
            cols = _split_columns(line)
            methods = cols[1:]
            continue

        if not methods:
            continue

        cols = _split_columns(line)
        if len(cols) != len(methods) + 1:
            continue

        workload = cols[0]
        values = [_to_float(v) for v in cols[1:]]
        data[current_base][workload] = {m: v for m, v in zip(methods, values)}

    for base in ["FFSlim", "TDP"]:
        if not data[base]:
            raise ValueError(f"Missing section: {base}")

    return data


def plot_fig10(data: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_eurosys_style()

    methods_order = ["FFSlim", "TDP", "LRU", "Quiver", "CoorDL", "EMD-Cache"]
    panel_order = ["FFSlim", "TDP"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    legend_handles = {}

    for idx, base in enumerate(panel_order):
        ax = axes[idx]
        workloads = list(data[base].keys())
        x = np.arange(len(workloads))

        methods = [m for m in methods_order if m in next(iter(data[base].values())).keys()]
        width = 0.13
        offset = -((len(methods) - 1) * width / 2.0)

        for i, method in enumerate(methods):
            y = np.array([data[base][w][method] for w in workloads], dtype=float)
            ours = method == "EMD-Cache"
            bars = ax.bar(
                x + offset + i * width,
                y,
                width=width,
                color=method_color(method),
                edgecolor="#1F1F1F",
                linewidth=0.9 if ours else 0.75,
                alpha=0.98 if ours else 0.9,
                zorder=3,
                label=method,
            )

            if method not in legend_handles:
                legend_handles[method] = bars[0]

        ax.set_title(f"Base {base}")
        ax.set_xticks(x)
        ax.set_xticklabels([_format_workload_label(w) for w in workloads], rotation=0, ha="center")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_ylabel("Training Time (s)")

    all_values = []
    for base in panel_order:
        for workload in data[base].values():
            all_values.extend(workload.values())
    ymax = max(all_values) if all_values else 1.0
    for ax in axes:
        ax.set_ylim(0, ymax * 1.12)

    legend_order = [m for m in methods_order if m in legend_handles]
    handles = [legend_handles[m] for m in legend_order]
    fig.legend(
        handles,
        legend_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )

    fig.tight_layout(rect=[0.0, 0.085, 1.0, 0.98])

    png_path = os.path.join(OUTPUT_DIR, "fig10_training_time.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig10_training_time.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    data = parse_fig10(INPUT_FILE)
    plot_fig10(data)


if __name__ == "__main__":
    main()
