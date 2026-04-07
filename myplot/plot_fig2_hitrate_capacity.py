import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig2.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _to_float_percent(s: str) -> float:
    return float(s.strip().replace("%", ""))


def parse_fig2(path: str) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {}
    current_format = None
    methods = ["LRU", "Quiver", "CoorDL"]

    row_re = re.compile(
        r"^([A-Za-z0-9_\-]+)\s+([0-9]+%)\s+([0-9.]+%)\s+([0-9.]+%)\s+([0-9.]+%)\s*$"
    )

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        section = re.match(r"^#+\s*(FFSlim|TDP)\s*#+$", s)
        if section:
            current_format = section.group(1)
            data.setdefault(current_format, {})
            continue

        if current_format is None:
            continue

        if s.startswith("Workload") or "Cache hit rate" in s:
            continue

        m = row_re.match(s)
        if not m:
            continue

        workload = m.group(1)
        cap = _to_float_percent(m.group(2))
        vals = [_to_float_percent(m.group(3)), _to_float_percent(m.group(4)), _to_float_percent(m.group(5))]

        workload_dict = data[current_format].setdefault(workload, {k: [] for k in methods})
        for method, value in zip(methods, vals):
            workload_dict[method].append((cap, value))

    for fmt in ["FFSlim", "TDP"]:
        if fmt not in data:
            raise ValueError(f"Missing section: {fmt}")
        for workload in ["Flickr30k", "GQA"]:
            if workload not in data[fmt]:
                raise ValueError(f"Missing workload {workload} in {fmt}")

    return data


def linear_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, float]:
    coef = np.polyfit(xs, ys, deg=1)
    fit = np.polyval(coef, xs)
    y_mean = ys.mean()
    ss_tot = np.sum((ys - y_mean) ** 2)
    ss_res = np.sum((ys - fit) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return fit, r2


def plot_fig2(data: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.8,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    formats = ["FFSlim", "TDP"]
    workloads = ["Flickr30k", "GQA"]
    methods = ["LRU", "Quiver", "CoorDL"]

    colors = {
        "LRU": "#4C78A8",
        "Quiver": "#F58518",
        "CoorDL": "#54A24B",
    }
    markers = {
        "LRU": "o",
        "Quiver": "s",
        "CoorDL": "^",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.3), sharex=True)

    legend_handles = []
    legend_labels = []

    for i, fmt in enumerate(formats):
        for j, workload in enumerate(workloads):
            ax = axes[i, j]

            ymax = 0.0
            r2_values = []
            for method in methods:
                pairs = sorted(data[fmt][workload][method], key=lambda x: x[0])
                xs = np.array([p[0] for p in pairs], dtype=float)
                ys = np.array([p[1] for p in pairs], dtype=float)
                ymax = max(ymax, ys.max())

                (line,) = ax.plot(
                    xs,
                    ys,
                    color=colors[method],
                    marker=markers[method],
                    markersize=4.2,
                    linewidth=1.35,
                    markerfacecolor="white",
                    markeredgewidth=0.95,
                    label=method,
                    zorder=3,
                )

                fit, r2 = linear_fit(xs, ys)
                r2_values.append(r2)
                ax.plot(
                    xs,
                    fit,
                    color=colors[method],
                    linestyle="--",
                    linewidth=0.95,
                    alpha=0.65,
                    zorder=2,
                )

                if i == 0 and j == 0:
                    legend_handles.append(line)
                    legend_labels.append(method)

            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_xlim(8, 102)
            ax.set_xticks(np.arange(10, 101, 10))
            ax.set_ylim(0, ymax * 1.15)
            ax.text(
                0.02,
                0.95,
                f"Avg. R^2={np.mean(r2_values):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.8, "pad": 2.0},
            )
            ax.text(
                0.98,
                0.95,
                f"{fmt} | {workload}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.8, "pad": 2.0},
            )
            if j == 0:
                ax.set_ylabel("Hit Rate (%)")
            if i == 1:
                ax.set_xlabel("Cache Capacity (%)")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=[0.03, 0.07, 1.0, 0.96])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "fig2_hitrate_capacity_linear.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig2_hitrate_capacity_linear.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    parsed = parse_fig2(INPUT_FILE)
    plot_fig2(parsed)


if __name__ == "__main__":
    main()
