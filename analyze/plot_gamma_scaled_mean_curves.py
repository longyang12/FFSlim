import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma


def load_numbers(file_path: Path) -> np.ndarray:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {file_path}")
    values = [float(x) for x in text.split()]
    return np.asarray(values, dtype=np.float64)


def parse_best_gamma(summary_file: Path):
    text = summary_file.read_text(encoding="utf-8")
    match = re.search(r"gamma: params=\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", text)
    if not match:
        raise ValueError(f"Cannot find gamma params in summary: {summary_file}")
    shape = float(match.group(1))
    loc = float(match.group(2))
    scale = float(match.group(3))
    return shape, loc, scale


def parse_scales(scale_text: str):
    return [float(x.strip()) for x in scale_text.split(",") if x.strip()]


def apply_eurosys_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.7,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gamma curves with scaled mean while keeping shape consistent."
    )
    parser.add_argument("--input", required=True, help="Path to ratio nums txt")
    parser.add_argument("--summary", required=True, help="Path to fit summary txt that includes best gamma params")
    parser.add_argument("--output", required=True, help="Path to output png")
    parser.add_argument("--stats-out", required=True, help="Path to output curve stats txt")
    parser.add_argument(
        "--factors",
        default="0.08,0.24,0.40,0.56,0.72,1.0,1.2,1.4,1.6,1.8,2.0",
        help="Comma-separated multipliers applied to gamma scale (default: mean starts near 1, left 5 + baseline + right 5)",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins")
    args = parser.parse_args()

    sample = load_numbers(Path(args.input))
    base_k, base_loc, base_theta = parse_best_gamma(Path(args.summary))
    factors = parse_scales(args.factors)

    x_min = max(0.0, np.percentile(sample, 0.1) - 1.0)
    x_max = np.percentile(sample, 99.9) + 1.0
    x = np.linspace(x_min, x_max, 1400)

    apply_eurosys_style()

    plt.figure(figsize=(7.2, 4.6))
    plt.hist(
        sample,
        bins=args.bins,
        density=True,
        alpha=0.52,
        color="#A7C4D8",
        edgecolor="#5C6670",
        linewidth=0.25,
        label="Real histogram",
    )

    cmap = plt.get_cmap("tab10")
    lines = []

    for i, f in enumerate(factors):
        theta = base_theta * f
        y = gamma.pdf(x, a=base_k, loc=base_loc, scale=theta)
        mean = base_k * theta + base_loc
        std = np.sqrt(base_k) * theta
        cv = std / mean if mean != 0 else np.nan

        label = f"factor={f:g}, mean={mean:.2f}, std={std:.2f}"
        lw = 2.3 if abs(f - 1.0) < 1e-12 else 1.55
        color = "#C44E52" if abs(f - 1.0) < 1e-12 else cmap(i % 10)
        plt.plot(x, y, linewidth=lw, color=color, label=label)

        lines.append(
            f"factor={f:g}, shape={base_k:.12f}, loc={base_loc:.6f}, scale={theta:.12f}, "
            f"mean={mean:.6f}, std={std:.6f}, cv={cv:.6f}"
        )

    plt.title("Gamma Curves with Scaled Mean (Fixed Shape)")
    plt.xlabel("Ratio value")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--")
    plt.legend(frameon=True, framealpha=0.96, edgecolor="#BBBBBB", ncol=1)
    plt.tight_layout()

    output_path = Path(args.output)
    stats_path = Path(args.stats_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()

    header = [
        f"base_gamma_shape={base_k:.12f}",
        f"base_gamma_loc={base_loc:.6f}",
        f"base_gamma_scale={base_theta:.12f}",
        "",
        "[scaled_curves]",
    ]
    stats_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

    print(f"Saved plot: {output_path}")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
