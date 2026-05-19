import re
from pathlib import Path

import matplotlib.pyplot as plt

FIG12_PATH = Path(__file__).with_name("fig12.txt")
OUTPUT_PATH = Path(__file__).with_name("fig12_trends.png")

FORMATS = ["TDP", "WebDataset", "TFRecord", "FFRecord"]
FORMAT_KEYS = ["tdp", "webdataset", "tfrecord", "ffrecord"]
TIME_LINES = [3600, 86400, 604800, 2592000]
TIME_LABELS = ["1h", "1d", "1w", "1m"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

STYLE = {
    "tdp": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "webdataset": {"color": "#d62728", "marker": "s", "linestyle": "--"},
    "tfrecord": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "ffrecord": {"color": "#ff7f0e", "marker": "D", "linestyle": ":"},
}


def _parse_section(lines, title_prefix):
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(title_prefix):
            start = i
            break
    if start is None:
        raise ValueError(f"Section not found: {title_prefix}")

    header_idx = start + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    header = re.split(r"\s+", lines[header_idx].strip())
    if header[0] != "Workload":
        raise ValueError(f"Unexpected header in {title_prefix}: {header}")

    data = {}
    row_idx = header_idx + 1
    while row_idx < len(lines):
        row = lines[row_idx].strip()
        if not row:
            break
        parts = re.split(r"\s+", row)
        if len(parts) < 1 + len(FORMATS):
            break
        workload = parts[0]
        values = parts[1:1 + len(FORMATS)]
        data[workload] = [float(v.replace(",", "")) for v in values]
        row_idx += 1
    return data


def _scale_labels(workloads, sizes_gb):
    labels = []
    for w, size in zip(workloads, sizes_gb):
        display_w = w.replace("MSCOCO-", "")
        if size >= 1024 * 1024:
            size_label = "(PB scale)"
        elif size >= 1024:
            size_label = "(TB scale)"
        else:
            size_label = "(GB scale)"
        labels.append(f"{display_w}\n{size_label}")
    return labels


def _plot(ax, workloads, data, sizes_gb, title):
    for fmt, key in zip(FORMATS, FORMAT_KEYS):
        values = [data[w][FORMAT_KEYS.index(key)] for w in workloads]
        style = STYLE[key]
        ax.plot(
            workloads,
            values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=fmt,
        )

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Time (s, log scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for t, label in zip(TIME_LINES, TIME_LABELS):
        ax.axhline(t, color="gray", linestyle=":", linewidth=1)
        ax.text(
            1.01,
            t,
            label,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color="gray",
        )

    ax.set_xticks(workloads)
    ax.set_xticklabels(_scale_labels(workloads, sizes_gb))


def main():
    lines = FIG12_PATH.read_text(encoding="utf-8").splitlines()

    f2b_time = _parse_section(lines, "采样模式切换时间(F2B)-64进程估计")
    b2f_time = _parse_section(lines, "采样模式切换时间(B2F)-64进程估计")
    b_size = _parse_section(lines, "B模式下数据集大小(GB)-64进程估计")
    f_size = _parse_section(lines, "F模式下数据集大小(GB)-64进程估计")

    workloads = list(f2b_time.keys())
    b_sizes = [b_size[w][0] for w in workloads]
    f_sizes = [f_size[w][0] for w in workloads]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _plot(axes[0], workloads, f2b_time, f_sizes, "F2B Switch Time (64 proc)")
    _plot(axes[1], workloads, b2f_time, b_sizes, "B2F Switch Time (64 proc)")

    axes[0].legend(loc="upper left")
    fig.suptitle("Switch Time vs Dataset Scale (64 processes)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
