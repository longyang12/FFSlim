import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - handled at runtime
    stats = None


def load_numbers(file_path: Path) -> np.ndarray:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {file_path}")

    nums = []
    for token in text.split():
        try:
            nums.append(float(token))
        except ValueError:
            continue

    if not nums:
        raise ValueError(f"No valid numeric values found in: {file_path}")

    return np.asarray(nums, dtype=np.float64)


def apply_eurosys_style() -> None:
    # Use a compact, publication-friendly style close to systems-paper figures.
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
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
        }
    )


def fit_distributions(sample: np.ndarray):
    if stats is None:
        return []

    result = []

    # Normal distribution
    norm_params = stats.norm.fit(sample)
    norm_ll = np.sum(stats.norm.logpdf(sample, *norm_params))
    result.append(("normal", norm_params, 2, norm_ll, stats.norm))

    positive = sample[sample > 0]
    if positive.size > 3:
        # Lognormal: fix location at 0 for positive ratios
        lognorm_params = stats.lognorm.fit(positive, floc=0)
        lognorm_ll = np.sum(stats.lognorm.logpdf(positive, *lognorm_params))
        result.append(("lognormal", lognorm_params, 2, lognorm_ll, stats.lognorm))

        # Gamma: fix location at 0 for positive ratios
        gamma_params = stats.gamma.fit(positive, floc=0)
        gamma_ll = np.sum(stats.gamma.logpdf(positive, *gamma_params))
        result.append(("gamma", gamma_params, 2, gamma_ll, stats.gamma))

    return result


def aic(k: int, log_likelihood: float) -> float:
    return 2 * k - 2 * log_likelihood


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ratio distribution and fit curves.")
    parser.add_argument("--input", required=True, help="Path to txt file containing space-separated numbers")
    parser.add_argument("--output", required=True, help="Path to output figure (.png)")
    parser.add_argument("--summary", required=True, help="Path to output fit summary (.txt)")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    sample = load_numbers(input_path)

    x_min = max(0.0, np.percentile(sample, 0.1) - 1.0)
    x_max = np.percentile(sample, 99.8) + 1.0
    x = np.linspace(x_min, x_max, 1000)

    apply_eurosys_style()

    plt.figure(figsize=(7.0, 4.2))
    plt.hist(
        sample,
        bins=args.bins,
        density=True,
        alpha=0.55,
        color="#9DBCD4",
        edgecolor="#5C6670",
        linewidth=0.3,
        label="Histogram",
    )

    fit_info = fit_distributions(sample)
    fit_info_sorted = sorted(fit_info, key=lambda item: aic(item[2], item[3]))

    colors = {
        "normal": "#C44E52",
        "lognormal": "#DD8452",
        "gamma": "#4C72B0",
    }

    summary_lines = []
    summary_lines.append(f"input={input_path}")
    summary_lines.append(f"count={sample.size}")
    summary_lines.append(f"min={sample.min():.6f}")
    summary_lines.append(f"max={sample.max():.6f}")
    summary_lines.append(f"mean={sample.mean():.6f}")
    summary_lines.append(f"std={sample.std(ddof=1):.6f}")
    summary_lines.append(f"p50={np.percentile(sample, 50):.6f}")
    summary_lines.append(f"p90={np.percentile(sample, 90):.6f}")
    summary_lines.append(f"p99={np.percentile(sample, 99):.6f}")

    if fit_info_sorted:
        summary_lines.append("\n[fitted_distributions]")
        for name, params, k, log_likelihood, dist_obj in fit_info_sorted:
            current_aic = aic(k, log_likelihood)
            y = dist_obj.pdf(x, *params)
            plt.plot(x, y, linewidth=2.0, color=colors.get(name, "black"), label=f"{name} (AIC={current_aic:.1f})")
            summary_lines.append(
                f"{name}: params={tuple(float(p) for p in params)}, logL={log_likelihood:.6f}, AIC={current_aic:.6f}"
            )

        best_name = fit_info_sorted[0][0]
        summary_lines.append(f"best_by_aic={best_name}")
    else:
        summary_lines.append("\n[fitted_distributions]")
        summary_lines.append("scipy_not_available_or_data_not_suitable")

    plt.xlabel("Ratio value")
    plt.ylabel("Density")
    plt.title("Ratio Distribution with Parametric Fits")
    plt.grid(True, linestyle="--")
    plt.legend(frameon=True, framealpha=0.95, edgecolor="#BBBBBB")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved figure: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
