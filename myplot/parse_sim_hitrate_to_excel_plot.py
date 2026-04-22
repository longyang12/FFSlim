import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOG_PATH = Path("/home/llm/experiment/dataset_Analysis_Modeling/output_sim.log")
MAPPING_PATH = Path("/mnt/datasets/generate_datasets/gqa_gamma11_mapping.csv")
OUT_DIR = Path("/home/llm/experiment/dataset_Analysis_Modeling/myplot/output")
EXCEL_PATH = OUT_DIR / "gqa_gamma11_hitrate_tables.xlsx"
PLOT_PATH = OUT_DIR / "gqa_gamma11_hitrate_curves.png"

FFSLIM_ORDER = [
    "lru",
    "quiver",
    "coordl",
    "dgim_lru",
    "dgim_quiver",
    "dgim_coordl",
    "dgim_iacma",
    "dgim_iacma_admission",
    "dgim_iacma_with_memory",
    "dgim_iacma_with_memory_admission",
]

TDP_ORDER = [
    "lru",
    "quiver",
    "coordl",
    "iacma",
    "iacma_admission",
    "iacma_with_memory",
    "iacma_with_memory_admission",
]

OVERLAP_EPS = 0.003


SIM_HEADER_RE = re.compile(
    r"^\[SIM\]\[(FFSlim|TDP)\]\s+dataset=([^,]+),.*cache=([^,]+),"
)
HIT_RE = re.compile(r"cache hit rate:([0-9.]+)")


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    current = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = SIM_HEADER_RE.match(line)
            if m:
                current = {
                    "format": m.group(1),
                    "dataset": m.group(2),
                    "algorithm": m.group(3),
                }
                continue

            if current is not None:
                hm = HIT_RE.search(line)
                if hm:
                    row = dict(current)
                    row["hit_rate"] = float(hm.group(1))
                    rows.append(row)
                    current = None

    if not rows:
        raise ValueError(f"No hit-rate records found in {path}")

    return pd.DataFrame(rows)


def load_mapping(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    for col in ["ffslim_output_root", "tdp_output_root"]:
        m[col] = m[col].astype(str).str.rstrip("/")
    m["ffslim_dataset"] = m["ffslim_output_root"].apply(lambda x: Path(x).name)
    m["tdp_dataset"] = m["tdp_output_root"].apply(lambda x: Path(x).name)
    return m[["factor", "mean", "std", "ffslim_dataset", "tdp_dataset"]]


def attach_mean_std(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    ff_map = mapping[["factor", "mean", "std", "ffslim_dataset"]].rename(
        columns={"ffslim_dataset": "dataset"}
    )
    tdp_map = mapping[["factor", "mean", "std", "tdp_dataset"]].rename(
        columns={"tdp_dataset": "dataset"}
    )

    ff = df[df["format"] == "FFSlim"].merge(ff_map, on="dataset", how="left")
    tdp = df[df["format"] == "TDP"].merge(tdp_map, on="dataset", how="left")

    out = pd.concat([ff, tdp], ignore_index=True)
    missing = out[out["mean"].isna()]
    if not missing.empty:
        miss_names = ", ".join(sorted(missing["dataset"].unique()))
        raise ValueError(f"Cannot map mean/std for datasets: {miss_names}")

    out["factor"] = out["factor"].astype(float)
    out["mean"] = out["mean"].astype(float)
    out["std"] = out["std"].astype(float)
    return out


def build_tables(df: pd.DataFrame):
    tables = {}
    for fmt in ["FFSlim", "TDP"]:
        sub = df[df["format"] == fmt].copy()
        algo_order = FFSLIM_ORDER if fmt == "FFSlim" else TDP_ORDER
        sub["algorithm"] = pd.Categorical(sub["algorithm"], categories=algo_order, ordered=True)
        sub = sub.sort_values(["mean", "algorithm"])
        pivot = sub.pivot_table(
            index=["factor", "mean", "std"],
            columns="algorithm",
            values="hit_rate",
            observed=False,
        )
        pivot = pivot.reindex(columns=algo_order)
        pivot = pivot.sort_index(level="mean")
        tables[fmt] = pivot
    return tables


def _overlap_groups(vals: pd.Series, eps: float):
    pairs = []
    items = [(k, v) for k, v in vals.items() if pd.notna(v)]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_name, a_val = items[i]
            b_name, b_val = items[j]
            if abs(float(a_val) - float(b_val)) <= eps:
                pairs.append((a_name, b_name, float((a_val + b_val) / 2.0)))
    return pairs


def plot_curves(tables, plot_path: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    overlap_notes = []

    for ax, fmt in zip(axes, ["FFSlim", "TDP"]):
        tab = tables[fmt].reset_index().sort_values("mean")
        x = tab["mean"].values
        for col in (FFSLIM_ORDER if fmt == "FFSlim" else TDP_ORDER):
            if col in {"factor", "mean", "std"}:
                continue
            if col not in tab.columns:
                continue
            y = tab[col].values
            ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.8, label=col)

        # Mark overlap points where different algorithms have nearly identical hit rates.
        algo_cols = [c for c in (FFSLIM_ORDER if fmt == "FFSlim" else TDP_ORDER) if c in tab.columns]
        for _, row in tab.iterrows():
            vals = row[algo_cols]
            pairs = _overlap_groups(vals, OVERLAP_EPS)
            if not pairs:
                continue
            x0 = float(row["mean"])
            y0 = pairs[0][2]
            ax.scatter([x0], [y0], s=28, facecolors="none", edgecolors="black", linewidths=0.9, zorder=6)
            note = "; ".join([f"{a}={b}" for a, b, _ in pairs])
            overlap_notes.append({"format": fmt, "mean": x0, "pairs": note})

        ax.set_title(f"{fmt} Hit Rate vs Mean Ratio")
        ax.set_xlabel("Ratio Mean")
        ax.grid(alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Hit Rate")
    axes[1].legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    return pd.DataFrame(overlap_notes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse cache simulation log and export hit-rate tables/curves.")
    parser.add_argument("--log", type=Path, default=LOG_PATH, help="Path to output_sim.log")
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH, help="Path to gamma mapping csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for excel and plot outputs")
    parser.add_argument("--excel-name", type=str, default=EXCEL_PATH.name, help="Excel output filename")
    parser.add_argument("--plot-name", type=str, default=PLOT_PATH.name, help="Plot output filename")
    return parser


def main():
    args = build_parser().parse_args()

    out_dir = args.out_dir
    excel_path = out_dir / args.excel_name
    plot_path = out_dir / args.plot_name

    out_dir.mkdir(parents=True, exist_ok=True)

    log_df = parse_log(args.log)
    mapping_df = load_mapping(args.mapping)
    full_df = attach_mean_std(log_df, mapping_df)
    full_df = full_df.sort_values(["format", "mean", "algorithm"])

    tables = build_tables(full_df)

    overlap_df = plot_curves(tables, plot_path)

    with pd.ExcelWriter(excel_path) as writer:
        full_df.to_excel(writer, sheet_name="raw_records", index=False)
        tables["FFSlim"].to_excel(writer, sheet_name="FFSlim_hit_rate")
        tables["TDP"].to_excel(writer, sheet_name="TDP_hit_rate")
        if not overlap_df.empty:
            overlap_df.to_excel(writer, sheet_name="overlap_marks", index=False)

    print(f"Saved excel: {excel_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
