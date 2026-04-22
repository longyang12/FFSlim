import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_eurosys_style, is_dgim, is_ours, method_color, method_marker


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig9.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _to_float(value: str) -> float:
    return float(value.strip().replace("%", "").replace(",", ""))


def _split_columns(line: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.strip()) if p.strip()]
    return parts if parts else [p for p in line.split() if p]


def parse_fig9(path: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "FFSlim": {},
        "TDP": {},
    }

    current_base = ""
    current_dataset = ""
    methods: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#") and "FFSlim" in line:
            current_base = "FFSlim"
            current_dataset = ""
            methods = []
            continue

        if line.startswith("#") and "TDP" in line:
            current_base = "TDP"
            current_dataset = ""
            methods = []
            continue

        if "Derived CC3M" in line:
            current_dataset = "CC3M"
            methods = []
            continue

        if "Derived GQA" in line:
            current_dataset = "GQA"
            methods = []
            continue

        if not current_base or not current_dataset:
            continue

        if line.lower().startswith("ratio mean"):
            cols = _split_columns(line)
            methods = cols[1:]
            data[current_base][current_dataset] = {"ratio": [], **{m: [] for m in methods}}
            continue

        if not methods:
            continue

        cols = _split_columns(line)
        if len(cols) != len(methods) + 1:
            continue

        ratio = _to_float(cols[0])
        values = [_to_float(v) for v in cols[1:]]

        panel = data[current_base][current_dataset]
        panel["ratio"].append(ratio)
        for m, v in zip(methods, values):
            panel[m].append(v)

    for base in ["FFSlim", "TDP"]:
        for dataset in ["CC3M", "GQA"]:
            if dataset not in data[base]:
                raise ValueError(f"Missing dataset {dataset} in section {base}")

    return data


def plot_fig9(data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> None:
    apply_eurosys_style()

    panel_order = [
        ("FFSlim", "CC3M"),
        ("FFSlim", "GQA"),
        ("TDP", "CC3M"),
        ("TDP", "GQA"),
    ]

    titles = {
        ("FFSlim", "CC3M"): "Base FFSlim | CC3M Variant",
        ("FFSlim", "GQA"): "Base FFSlim | GQA Variant",
        ("TDP", "CC3M"): "Base TDP | CC3M Variant",
        ("TDP", "GQA"): "Base TDP | GQA Variant",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.4), sharey=True)

    legend_handles = {}

    for idx, (base, dataset) in enumerate(panel_order):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

        panel = data[base][dataset]
        x = np.array(panel["ratio"], dtype=float)

        methods = [m for m in panel.keys() if m != "ratio"]

        for method in methods:
            y = np.array(panel[method], dtype=float)
            dgim_method = is_dgim(method)
            ours_method = is_ours(method)
            marker_size = 8.1 if ours_method else (7.6 if dgim_method else 6.5)

            line, = ax.plot(
                x,
                y,
                color=method_color(method),
                linestyle="-",
                linewidth=2.25 if ours_method else 1.4,
                marker=method_marker(method),
                markersize=marker_size,
                markerfacecolor=method_color(method),
                markeredgecolor="#1F1F1F",
                markeredgewidth=1.0 if ours_method else 0.9,
                alpha=1.0,
                zorder=4 if (dgim_method or ours_method) else 3,
                label=method,
            )

            if method not in legend_handles:
                legend_handles[method] = line

        ax.set_title(titles[(base, dataset)])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
        ax.set_axisbelow(True)

        xmin, xmax = x.min(), x.max()
        pad = (xmax - xmin) * 0.04 if xmax > xmin else 0.5
        ax.set_xlim(xmin - pad, xmax + pad)

        if c == 0:
            ax.set_ylabel("Hit Rate (%)")
        if r == 1:
            ax.set_xlabel("Media/Text Ratio")

    all_values = []
    for base in data.values():
        for panel in base.values():
            for k, vals in panel.items():
                if k != "ratio":
                    all_values.extend(vals)
    ymax = max(all_values) if all_values else 100.0
    for ax in axes.ravel():
        ax.set_ylim(0, max(75.0, ymax * 1.08))

    legend_order = [
        "LRU",
        "Quiver",
        "CoorDL",
        "DGIM-LRU",
        "DGIM-Quiver",
        "DGIM-CoorDL",
        "EMD-Cache",
    ]
    labels = [m for m in legend_order if m in legend_handles]
    handles = [legend_handles[m] for m in labels]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.0,
    )

    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.98])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "fig9_ratio_hit_rate_impact.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig9_ratio_hit_rate_impact.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    data = parse_fig9(INPUT_FILE)
    plot_fig9(data)


if __name__ == "__main__":
    main()
