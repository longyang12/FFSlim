import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/fig1.txt"
OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/myplot/output"


def _split_cols(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []

    # Prefer tabs / aligned spacing, then fallback to generic split.
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line) if p.strip()]
    if len(parts) <= 1:
        parts = line.split()
    return parts


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
        row = {}
        for method, value in zip(methods, values):
            row[method] = _to_float(value)
        table[workload] = row
        i += 1

    return methods, table, i


def parse_fig1(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    result: Dict[str, Dict[str, object]] = {}
    i = 0
    current_format = None

    while i < len(lines):
        s = lines[i].strip()

        m = re.match(r"^#+\s*(FFSlim|TDP)\s*#+$", s)
        if m:
            current_format = m.group(1)
            result[current_format] = {
                "methods": [],
                "performance": {},
                "hit_rate": {},
            }
            i += 1
            continue

        if current_format and "Samples per second" in s:
            while i < len(lines) and "Workload" not in lines[i]:
                i += 1
            methods, perf_table, i = _parse_table(lines, i)
            result[current_format]["methods"] = methods
            result[current_format]["performance"] = perf_table
            continue

        if current_format and "Cache hit rate" in s:
            while i < len(lines) and "Workload" not in lines[i]:
                i += 1
            methods, hit_table, i = _parse_table(lines, i)
            if not result[current_format]["methods"]:
                result[current_format]["methods"] = methods
            result[current_format]["hit_rate"] = hit_table
            continue

        i += 1

    for fmt in ["FFSlim", "TDP"]:
        if fmt not in result:
            raise ValueError(f"Missing section: {fmt}")
        if not result[fmt]["performance"]:
            raise ValueError(f"Missing performance table for: {fmt}")
        if not result[fmt]["hit_rate"]:
            raise ValueError(f"Missing hit-rate table for: {fmt}")

    return result


def _eurosys_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_dual_axis_for_format(fmt: str, payload: Dict[str, object], output_dir: str) -> None:
    methods: List[str] = payload["methods"]  # type: ignore[assignment]
    perf: Dict[str, Dict[str, float]] = payload["performance"]  # type: ignore[assignment]
    hit: Dict[str, Dict[str, float]] = payload["hit_rate"]  # type: ignore[assignment]

    workloads = list(perf.keys())
    x = np.arange(len(workloads), dtype=float)

    colors = {
        methods[0]: "#4C78A8",
        "LRU": "#F58518",
        "Quiver": "#54A24B",
        "CoorDL": "#E45756",
    }

    fig, ax1 = plt.subplots(figsize=(10.2, 4.0))
    ax2 = ax1.twinx()

    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    line_offsets = np.linspace(-0.09, 0.09, len(methods))

    bar_handles = []
    line_handles = []

    line_styles = {
        methods[0]: "-",
        "LRU": "--",
        "Quiver": "-.",
        "CoorDL": ":",
    }
    line_markers = {
        methods[0]: "o",
        "LRU": "s",
        "Quiver": "^",
        "CoorDL": "D",
    }

    for idx, method in enumerate(methods):
        y_perf = [perf[w][method] for w in workloads]
        y_hit = [hit[w][method] for w in workloads]

        bar = ax1.bar(
            x + offsets[idx],
            y_perf,
            width=width,
            color=colors.get(method, "#999999"),
            edgecolor="#1F1F1F",
            linewidth=0.5,
            alpha=0.85,
            label=f"{method} Throughput",
            zorder=2,
        )
        bar_handles.append(bar)

        # Keep hit-rate curves on workload centers to avoid zig-zag visual clutter.
        (line,) = ax2.plot(
            x + line_offsets[idx],
            y_hit,
            color=colors.get(method, "#999999"),
            marker=line_markers.get(method, "o"),
            markersize=4.6,
            linewidth=1.5,
            linestyle=line_styles.get(method, "-"),
            label=f"{method} Hit Rate",
            zorder=3,
        )
        line.set_markerfacecolor("white")
        line.set_markeredgewidth(1.0)
        line_handles.append(line)

    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, rotation=20, ha="right")
    ax1.set_ylabel("Throughput (samples/s)")
    ax2.set_ylabel("Hit Rate (%)")
    ax1.set_xlabel("Workload")

    ax1.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.42)
    ax1.set_axisbelow(True)

    pmax = max(max(perf[w].values()) for w in workloads)
    hmax = max(max(hit[w].values()) for w in workloads)
    ax1.set_ylim(0, pmax * 1.20)
    ax2.set_ylim(0, max(10.0, hmax * 1.25))

    ax1.set_title(f"{fmt}: Throughput and Cache Hit Rate")

    method_handles = [bh[0] for bh in bar_handles]
    legend_handles = method_handles + line_handles
    legend_labels = methods + [f"{m} Hit" for m in methods]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.1,
    )

    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"fig1_{fmt.lower()}_dual_axis.png")
    pdf_path = os.path.join(output_dir, f"fig1_{fmt.lower()}_dual_axis.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    _eurosys_style()
    parsed = parse_fig1(INPUT_FILE)

    for fmt in ["FFSlim", "TDP"]:
        plot_dual_axis_for_format(fmt, parsed[fmt], OUTPUT_DIR)


if __name__ == "__main__":
    main()
