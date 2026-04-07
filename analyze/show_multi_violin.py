import os
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = "/home/llm/experiment/dataset_Analysis_Modeling/analyze/output"
DATASETS = [
    "cc3m",
    "flickr30k",
    "flickr30k_entities",
    "mscoco",
    "vqa2",
    "gqa",
    "MSRVTT",
    "Clotho",
    "Clotho_E",
]


def read_numbers(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    return [float(x) for x in content.split()]


def build_violin_data(dataset):
    sample_size_path = os.path.join(OUTPUT_DIR, f"{dataset}_sample_size_nums.txt")
    text_size_path = os.path.join(OUTPUT_DIR, f"{dataset}_text_size_nums.txt")
    ratio_path = os.path.join(OUTPUT_DIR, f"{dataset}_ratio_nums.txt")

    sample_sizes = read_numbers(sample_size_path)
    text_sizes = read_numbers(text_size_path)
    quantity_ratios = read_numbers(ratio_path)

    if not sample_sizes or not text_sizes or not quantity_ratios:
        raise ValueError(f"missing or empty data file for dataset: {dataset}")

    n = min(len(sample_sizes), len(text_sizes))
    sample_sizes = sample_sizes[:n]
    text_sizes = text_sizes[:n]

    # size ratio is media/text where media = sample - text
    size_ratios = [
        (sample_sizes[i] - text_sizes[i]) / text_sizes[i]
        for i in range(n)
        if text_sizes[i] > 0
    ]

    if not size_ratios:
        raise ValueError(f"no valid size ratio values for dataset: {dataset}")

    # Use log10 scale to put three heterogeneous metrics in one violin chart.
    sample_size_kb_log10 = np.log10(np.array(sample_sizes) / 1024.0 + 1e-12)
    quantity_ratio_log10 = np.log10(np.array(quantity_ratios) + 1e-12)
    size_ratio_log10 = np.log10(np.array(size_ratios) + 1e-12)

    return [sample_size_kb_log10, quantity_ratio_log10, size_ratio_log10]


def plot_multi_violin(dataset):
    violin_data = build_violin_data(dataset)

    fig, ax = plt.subplots(figsize=(9, 6))
    vp = ax.violinplot(
        violin_data,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    for body in vp["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_edgecolor("black")
        body.set_alpha(0.65)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([
        "Sample Size (KB)",
        "Quantity Ratio",
        "Size Ratio",
    ])
    ax.set_ylabel("log10(value)")
    ax.set_title(f"{dataset} multi-violin: sample size / quantity ratio / size ratio")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(OUTPUT_DIR, f"{dataset}_multi_violin.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    for dataset_name in DATASETS:
        try:
            plot_multi_violin(dataset_name)
        except Exception as e:
            print(f"skip {dataset_name}: {e}")
