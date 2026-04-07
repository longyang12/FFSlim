import os
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/analyze/output"
SAVE_PATH = os.path.join(OUTPUT_DIR, "eurosys_8datasets_3metrics_violin.pdf")
SAVE_PATH_PNG = os.path.join(OUTPUT_DIR, "eurosys_8datasets_3metrics_violin.png")

# Use Clotho_E as Clotho and remove original Clotho.
DATASETS = [
    ("cc3m", "CC3M"),
    ("flickr30k", "Flickr30k"),
    ("flickr30k_entities", "F30-E"),
    ("mscoco", "MSCOCO"),
    ("Clotho_E", "Clotho"),
    ("MSRVTT", "MSRVTT"),
    ("vqa2", "VQA2"),
    ("gqa", "GQA"),
]


def read_numbers(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [float(x) for x in text.split()]


def build_metric_arrays():
    sample_size_arrays = []
    qty_ratio_arrays = []
    size_ratio_arrays = []
    labels = []

    for key, display in DATASETS:
        sample_path = os.path.join(OUTPUT_DIR, f"{key}_sample_size_nums.txt")
        text_path = os.path.join(OUTPUT_DIR, f"{key}_text_size_nums.txt")
        qty_path = os.path.join(OUTPUT_DIR, f"{key}_ratio_nums.txt")

        if not (os.path.exists(sample_path) and os.path.exists(text_path) and os.path.exists(qty_path)):
            print(f"skip {display}: missing cache files")
            continue

        sample_sizes = read_numbers(sample_path)
        text_sizes = read_numbers(text_path)
        qty_ratios = read_numbers(qty_path)

        if not sample_sizes or not text_sizes or not qty_ratios:
            print(f"skip {display}: empty cache files")
            continue

        n = min(len(sample_sizes), len(text_sizes))
        sample_sizes = np.array(sample_sizes[:n], dtype=float)
        text_sizes = np.array(text_sizes[:n], dtype=float)
        qty_ratios = np.array(qty_ratios, dtype=float)

        valid = text_sizes > 0
        size_ratios = (sample_sizes[valid] - text_sizes[valid]) / text_sizes[valid]

        sample_size_arrays.append(sample_sizes / 1024.0)  # KB
        qty_ratio_arrays.append(qty_ratios)
        size_ratio_arrays.append(size_ratios)
        labels.append(display)

    return labels, sample_size_arrays, qty_ratio_arrays, size_ratio_arrays


def draw_metric_violin(ax, data_arrays, labels, title, ylabel, log_scale=False, highlight_label=None):
    positions = np.arange(1, len(data_arrays) + 1)

    vp = ax.violinplot(
        data_arrays,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # EuroSys-like clean style: subtle fill + black outlines.
    for idx, body in enumerate(vp["bodies"]):
        if highlight_label is not None and labels[idx] == highlight_label:
            body.set_facecolor("#D95F02")
            body.set_alpha(0.85)
        else:
            body.set_facecolor("#90AFC5")
            body.set_alpha(0.75)
        body.set_edgecolor("#1F1F1F")
        body.set_linewidth(0.8)

    means = [np.mean(v) for v in data_arrays]
    medians = [np.median(v) for v in data_arrays]

    # Median: horizontal thick black segment per violin.
    for i, (x, y) in enumerate(zip(positions, medians)):
        med_color = "#A63D00" if (highlight_label is not None and labels[i] == highlight_label) else "#111111"
        ax.hlines(y, x - 0.24, x + 0.24, colors=med_color, linewidth=2.0, zorder=4)

    # Mean: white-filled black-edged circle.
    for i, (x, y) in enumerate(zip(positions, means)):
        edge_color = "#A63D00" if (highlight_label is not None and labels[i] == highlight_label) else "#111111"
        ax.scatter(
            x,
            y,
            marker="o",
            s=26,
            facecolor="white",
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=5,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


def add_qty_zoom_inset(fig, data_arrays, labels):
    positions = np.arange(1, len(data_arrays) + 1)
    # Place inset at the center of the whole figure.
    axins = fig.add_axes([0.41, 0.35, 0.18, 0.22])

    vp = axins.violinplot(
        data_arrays,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    means = [np.mean(v) for v in data_arrays]
    medians = [np.median(v) for v in data_arrays]

    for idx, body in enumerate(vp["bodies"]):
        if labels[idx] == "VQA2":
            body.set_facecolor("#D95F02")
            body.set_alpha(0.85)
        else:
            body.set_facecolor("#B7CBDD")
            body.set_alpha(0.70)
        body.set_edgecolor("#1F1F1F")
        body.set_linewidth(0.6)

    for i, (x, y) in enumerate(zip(positions, medians)):
        med_color = "#A63D00" if labels[i] == "VQA2" else "#111111"
        axins.hlines(y, x - 0.20, x + 0.20, colors=med_color, linewidth=1.4, zorder=4)

    for i, (x, y) in enumerate(zip(positions, means)):
        edge_color = "#A63D00" if labels[i] == "VQA2" else "#111111"
        axins.scatter(x, y, marker="o", s=12, facecolor="white", edgecolor=edge_color, linewidth=0.8, zorder=5)

    # Zoom into the range where VQA2 and fixed one-to-many datasets are distinguishable.
    axins.set_ylim(0, 25)
    axins.set_xticks(positions)
    axins.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    axins.tick_params(axis="y", labelsize=6)
    axins.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    axins.set_title("Zoom (0-25)", fontsize=7, pad=2)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_linewidth(0.7)


def plot_all():
    labels, sample_size, qty_ratio, size_ratio = build_metric_arrays()
    if not labels:
        raise RuntimeError("No valid datasets found. Please generate cache files first.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=False)

    draw_metric_violin(
        axes[0],
        sample_size,
        labels,
        "Sample Size",
        "KB (log scale)",
        log_scale=True,
    )
    draw_metric_violin(
        axes[1],
        qty_ratio,
        labels,
        "Media-Text Quantity Ratio",
        "ratio",
        log_scale=False,
        highlight_label="VQA2",
    )
    add_qty_zoom_inset(fig, qty_ratio, labels)
    draw_metric_violin(
        axes[2],
        size_ratio,
        labels,
        "Media-Text Size Ratio",
        "ratio (log scale)",
        log_scale=True,
    )

    # Reserve top margin for legend and finalize subplot positions first.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    # Shared legend for mean/median semantics, anchored above the middle subplot.
    median_proxy = plt.Line2D([0], [0], color="#111111", lw=2.0)
    mean_proxy = plt.Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#111111", markersize=5
    )
    mid_box = axes[1].get_position()
    legend_x = (mid_box.x0 + mid_box.x1) / 2
    legend_y = min(mid_box.y1 + 0.03, 0.98)
    fig.legend(
        [median_proxy, mean_proxy],
        ["Median", "Mean"],
        loc="lower center",
        bbox_to_anchor=(legend_x, legend_y),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SAVE_PATH_PNG, dpi=300, bbox_inches="tight")
    print(f"saved: {SAVE_PATH}")
    print(f"saved: {SAVE_PATH_PNG}")


if __name__ == "__main__":
    plot_all()
