import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig7.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _split_cols(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    cols = [x.strip() for x in re.split(r"\t+|\s{2,}", line) if x.strip()]
    if len(cols) <= 1:
        cols = line.split()
    return cols


def _to_float(value: str) -> float:
    v = value.strip().replace(",", "")
    if v.endswith("%"):
        return float(v[:-1])
    return float(v)


def _parse_table(lines: List[str], header_idx: int) -> Tuple[List[str], Dict[str, Dict[str, float]], int]:
    headers = _split_cols(lines[header_idx])
    if len(headers) < 2 or headers[0].lower() != "workload":
        raise ValueError(f"Invalid table header at line: {lines[header_idx]}")

    methods = headers[1:]
    table: Dict[str, Dict[str, float]] = {}
    i = header_idx + 1

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        if s.startswith("#") or "Samples per second" in s or "Cache hit rate" in s:
            break

        cols = _split_cols(raw)
        if len(cols) < 1 + len(methods):
            break

        workload = cols[0]
        values = cols[1 : 1 + len(methods)]
        table[workload] = {m: _to_float(v) for m, v in zip(methods, values)}
        i += 1

    return methods, table, i


def parse_fig7_ffslim(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    result: Dict[str, object] = {"methods": [], "performance": {}, "hit_rate": {}}
    in_ffslim = False
    i = 0

    while i < len(lines):
        s = lines[i].strip()

        if re.match(r"^#+\s*FFSlim\s*#+$", s):
            in_ffslim = True
            i += 1
            continue
        if in_ffslim and re.match(r"^#+\s*TDP\s*#+$", s):
            break

        if in_ffslim and "Samples per second" in s:
            while i < len(lines) and "Workload" not in lines[i]:
                i += 1
            methods, perf_table, i = _parse_table(lines, i)
            result["methods"] = methods
            result["performance"] = perf_table
            continue

        if in_ffslim and "Cache hit rate" in s:
            while i < len(lines) and "Workload" not in lines[i]:
                i += 1
            methods, hit_table, i = _parse_table(lines, i)
            if not result["methods"]:
                result["methods"] = methods
            result["hit_rate"] = hit_table
            continue

        i += 1

    if not result["performance"] or not result["hit_rate"]:
        raise ValueError("Failed to parse FFSlim section in fig7.txt")

    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _speedup_stats(perf: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    workloads = list(perf.keys())
    stats = {
        "LRU": np.mean([perf[w]["DGIM-LRU"] / perf[w]["LRU"] for w in workloads]),
        "Quiver": np.mean([perf[w]["DGIM-Quiver"] / perf[w]["Quiver"] for w in workloads]),
        "CoorDL": np.mean([perf[w]["DGIM-CoorDL"] / perf[w]["CoorDL"] for w in workloads]),
    }
    return {k: float(v) for k, v in stats.items()}


def plot_fig7_ffslim(payload: Dict[str, object]) -> None:
    _style()

    methods: List[str] = payload["methods"]  # type: ignore[assignment]
    perf: Dict[str, Dict[str, float]] = payload["performance"]  # type: ignore[assignment]
    hit: Dict[str, Dict[str, float]] = payload["hit_rate"]  # type: ignore[assignment]

    workloads = list(perf.keys())
    x = np.arange(len(workloads), dtype=float)

    # Muted traditional methods, vivid DGIM family for emphasis.
    colors = {
        "FFSlim": "#9E9E9E",
        "LRU": "#A0A0A0",
        "Quiver": "#B0B0B0",
        "CoorDL": "#8F8F8F",
        "DGIM-LRU": "#4C78A8",
        "DGIM-Quiver": "#F58518",
        "DGIM-CoorDL": "#54A24B",
        "EMD-Cache": "#D62728",
    }
    markers = {
        "FFSlim": "o",
        "LRU": "o",
        "Quiver": "s",
        "CoorDL": "^",
        "DGIM-LRU": "D",
        "DGIM-Quiver": "P",
        "DGIM-CoorDL": "X",
        "EMD-Cache": "*",
    }
    hatches = {
        "FFSlim": "",
        "LRU": "",
        "Quiver": "",
        "CoorDL": "",
        "DGIM-LRU": "//",
        "DGIM-Quiver": "\\",
        "DGIM-CoorDL": "xx",
        "EMD-Cache": "",
    }

    fig, ax1 = plt.subplots(figsize=(12.4, 4.8))
    ax2 = ax1.twinx()

    width = 0.10
    offsets = np.linspace(-3.5 * width, 3.5 * width, len(methods))
    line_offsets = np.linspace(-0.12, 0.12, len(methods))

    bar_handles = []
    line_handles = []

    for idx, method in enumerate(methods):
        y_perf = [perf[w][method] for w in workloads]
        y_hit = [hit[w][method] for w in workloads]

        is_dgim = method.startswith("DGIM-")
        is_ours = method == "EMD-Cache"

        bar = ax1.bar(
            x + offsets[idx],
            y_perf,
            width=width,
            color=colors[method],
            edgecolor="#222222",
            linewidth=0.48 if not (is_dgim or is_ours) else 0.72,
            alpha=0.85 if not (is_dgim or is_ours) else 0.96,
            hatch=hatches[method],
            zorder=2,
        )
        bar_handles.append(bar)

        line, = ax2.plot(
            x + line_offsets[idx],
            y_hit,
            color=colors[method],
            linewidth=1.15 if not (is_dgim or is_ours) else 1.8,
            linestyle="-" if method in ["DGIM-LRU", "DGIM-Quiver", "DGIM-CoorDL", "EMD-Cache"] else "--",
            marker=markers[method],
            markersize=3.6 if not (is_dgim or is_ours) else (5.0 if not is_ours else 6.6),
            markerfacecolor="white" if not is_ours else colors[method],
            markeredgewidth=0.9 if not is_ours else 1.0,
            zorder=3,
        )
        line_handles.append(line)

    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, rotation=18, ha="right")
    ax1.set_xlabel("Workload")
    ax1.set_ylabel("Throughput (samples/s)")
    ax2.set_ylabel("Hit Rate (%)")

    pmax = max(max(perf[w].values()) for w in workloads)
    hmax = max(max(hit[w].values()) for w in workloads)
    ax1.set_ylim(0, pmax * 1.15)
    ax2.set_ylim(0, max(10.0, hmax * 1.12))

    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
    ax1.set_axisbelow(True)

    speedups = _speedup_stats(perf)
    note = (
        f"DGIM speedup over traditional (avg throughput): "
        f"LRU {speedups['LRU']:.2f}x, Quiver {speedups['Quiver']:.2f}x, CoorDL {speedups['CoorDL']:.2f}x"
    )
    ax1.text(
        0.01,
        0.98,
        note,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88, "pad": 2.2},
    )

    ax1.set_title("FFSlim: Throughput and Hit Rate with DGIM-Enhanced Cache Policies")

    method_handles = [h[0] for h in bar_handles]
    legend_handles = method_handles + line_handles
    legend_labels = methods + [f"{m} Hit" for m in methods]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=8,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.85,
    )

    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.98])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "fig7_ffslim_dgim_dual_axis.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig7_ffslim_dgim_dual_axis.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    payload = parse_fig7_ffslim(INPUT_FILE)
    plot_fig7_ffslim(payload)


if __name__ == "__main__":
    main()
