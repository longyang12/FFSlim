import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_eurosys_style, is_dgim, is_ours, method_color, method_marker


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


def _speedup_stats(perf: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    workloads = list(perf.keys())
    stats = {
        "LRU": np.mean([perf[w]["DGIM-LRU"] / perf[w]["LRU"] for w in workloads]),
        "Quiver": np.mean([perf[w]["DGIM-Quiver"] / perf[w]["Quiver"] for w in workloads]),
        "CoorDL": np.mean([perf[w]["DGIM-CoorDL"] / perf[w]["CoorDL"] for w in workloads]),
    }
    return {k: float(v) for k, v in stats.items()}


def plot_fig7_ffslim(payload: Dict[str, object]) -> None:
    apply_eurosys_style()

    methods: List[str] = payload["methods"]  # type: ignore[assignment]
    perf: Dict[str, Dict[str, float]] = payload["performance"]  # type: ignore[assignment]
    hit: Dict[str, Dict[str, float]] = payload["hit_rate"]  # type: ignore[assignment]

    workloads = list(perf.keys())
    x = np.arange(len(workloads), dtype=float)

    colors = {m: method_color(m) for m in methods}
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
    dot_offsets = offsets.copy()

    bar_handles = []
    dot_handles = []

    for idx, method in enumerate(methods):
        y_perf = [perf[w][method] for w in workloads]
        y_hit = [hit[w][method] for w in workloads]

        dgim_method = is_dgim(method)
        ours_method = is_ours(method)

        bar = ax1.bar(
            x + offsets[idx],
            y_perf,
            width=width,
            color=colors[method],
            edgecolor="#222222",
            linewidth=0.48 if not (dgim_method or ours_method) else 0.72,
            alpha=0.84 if not (dgim_method or ours_method) else 0.96,
            hatch=hatches[method],
            zorder=2,
        )
        bar_handles.append(bar)

        dots = ax2.scatter(
            x + dot_offsets[idx],
            y_hit,
            color=colors[method],
            marker=method_marker(method),
            s=42 if not (dgim_method or ours_method) else (58 if dgim_method else 66),
            linewidths=0.9 if not ours_method else 1.0,
            edgecolors="#1F1F1F",
            zorder=3,
        )
        dot_handles.append(dots)

    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, rotation=0, ha="center")
    ax1.set_xlabel("")
    ax1.set_ylabel("Throughput (samples/s)")
    ax2.set_ylabel("Hit Rate (%)")

    pmax = max(max(perf[w].values()) for w in workloads)
    hmax = max(max(hit[w].values()) for w in workloads)
    ax1.set_ylim(0, pmax * 1.15)
    ax2.set_ylim(0, max(10.0, hmax * 1.12))

    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.38)
    ax1.set_axisbelow(True)

    ax1.set_title("FFSlim:DGIM Acceleration for Traditional Cache Policies")

    method_handles = [h[0] for h in bar_handles]
    legend_handles = method_handles + dot_handles
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

    fig.tight_layout(rect=[0.0, 0.09, 1.0, 0.98])
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
