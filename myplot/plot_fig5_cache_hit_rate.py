import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_eurosys_style, method_color, method_marker


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig5.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _to_float_percent(value: str) -> float:
    return float(value.strip().replace("%", "").replace(",", ""))


def parse_fig5(path: str) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {}
    current_format = None
    methods: List[str] = []

    row_re = re.compile(
        r"^([A-Za-z0-9_\-]+)\s+([0-9]+%)\s+([0-9.]+%)\s+([0-9.]+%)\s+([0-9.]+%)\s+([0-9.]+%)\s*$"
    )

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        sec = re.match(r"^#+\s*(FFSlim|TDP)\s*#+$", s)
        if sec:
            current_format = sec.group(1)
            data.setdefault(current_format, {})
            continue

        if current_format is None:
            continue

        if s.startswith("Workload"):
            parts = [p for p in re.split(r"\t+|\s{2,}", s) if p]
            if len(parts) <= 1:
                parts = s.split()
            methods = [m.strip() for m in parts[1:]]
            continue

        if "Cache hit rate" in s:
            continue

        m = row_re.match(s)
        if not m:
            continue

        workload = m.group(1)
        cap = _to_float_percent(m.group(2))
        values = [_to_float_percent(m.group(3)), _to_float_percent(m.group(4)), _to_float_percent(m.group(5)), _to_float_percent(m.group(6))]

        workload_dict = data[current_format].setdefault(workload, {k: [] for k in methods})
        for method, value in zip(methods, values):
            workload_dict[method].append((cap, value))

    for fmt in ["FFSlim", "TDP"]:
        if fmt not in data:
            raise ValueError(f"Missing section: {fmt}")
        for workload in ["Flickr30k", "GQA"]:
            if workload not in data[fmt]:
                raise ValueError(f"Missing workload {workload} in {fmt}")

    return data


def plot_fig5(data: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]) -> None:
    apply_eurosys_style()

    formats = ["FFSlim", "TDP"]
    workloads = ["Flickr30k", "GQA"]
    methods = ["LRU", "Quiver", "CoorDL", "EMD-Cache"]

    colors = {m: method_color(m) for m in methods}
    line_styles = {
        "LRU": "-",
        "Quiver": "-",
        "CoorDL": "-",
        "EMD-Cache": "-",
    }
    markers = {
        "LRU": "o",
        "Quiver": "s",
        "CoorDL": "^",
        "EMD-Cache": "*",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.0), sharex=True, sharey=True)
    x_ticks = np.arange(10, 101, 10)

    legend_handles = []
    legend_labels = []

    for i, fmt in enumerate(formats):
        for j, workload in enumerate(workloads):
            ax = axes[i, j]
            panel_data = data[fmt][workload]

            ax.text(
                0.02,
                0.95,
                f"{fmt} | {workload}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.9, "pad": 2.0},
            )

            for idx, method in enumerate(methods):
                pairs = sorted(panel_data[method], key=lambda p: p[0])
                xs = np.array([p[0] for p in pairs], dtype=float)
                ys = np.array([p[1] for p in pairs], dtype=float)

                is_ours = method == "EMD-Cache"
                line, = ax.plot(
                    xs,
                    ys,
                    color=colors[method],
                    linestyle=line_styles[method],
                    linewidth=2.35 if is_ours else 1.45,
                    marker=method_marker(method),
                        markersize=8.2 if is_ours else 5.6,
                    markerfacecolor="white" if not is_ours else colors[method],
                    markeredgewidth=1.05 if is_ours else 0.95,
                    alpha=1.0 if is_ours else 0.9,
                    zorder=4 if is_ours else 3,
                    label=method,
                )

                if i == 0 and j == 0:
                    legend_handles.append(line)
                    legend_labels.append("EMD-Cache" if is_ours else method)

            ax.set_xlim(8, 102)
            ax.set_xticks(x_ticks)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
            ax.set_axisbelow(True)

            if j == 0:
                ax.set_ylabel("Hit Rate (%)")
            if i == 1:
                ax.set_xlabel("Cache Capacity (%)")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.0,
    )

    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.98])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "fig5_cache_hit_rate_capacity.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig5_cache_hit_rate_capacity.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    data = parse_fig5(INPUT_FILE)
    plot_fig5(data)


if __name__ == "__main__":
    main()
