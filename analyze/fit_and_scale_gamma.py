import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
    from scipy.stats import gamma as gamma_dist
except Exception:  # pragma: no cover - handled at runtime
    stats = None
    gamma_dist = None


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
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.7,
        }
    )


def fit_distributions(sample: np.ndarray):
    if stats is None:
        return []

    result = []

    norm_params = stats.norm.fit(sample)
    norm_ll = np.sum(stats.norm.logpdf(sample, *norm_params))
    result.append(("normal", norm_params, 2, norm_ll, stats.norm))

    positive = sample[sample > 0]
    if positive.size > 3:
        lognorm_params = stats.lognorm.fit(positive, floc=0)
        lognorm_ll = np.sum(stats.lognorm.logpdf(positive, *lognorm_params))
        result.append(("lognormal", lognorm_params, 2, lognorm_ll, stats.lognorm))

        gamma_params = stats.gamma.fit(positive, floc=0)
        gamma_ll = np.sum(stats.gamma.logpdf(positive, *gamma_params))
        result.append(("gamma", gamma_params, 2, gamma_ll, stats.gamma))

    return result


def aic(k: int, log_likelihood: float) -> float:
    return 2 * k - 2 * log_likelihood


def parse_factors(scale_text: str):
    return [float(x.strip()) for x in scale_text.split(",") if x.strip()]


def select_gamma_fit(fit_info_sorted):
    for name, params, k, log_likelihood, dist_obj in fit_info_sorted:
        if name == "gamma":
            return name, params, k, log_likelihood, dist_obj
    raise ValueError("Gamma fit is not available; cannot scale gamma curves.")


def build_figure(sample: np.ndarray, fit_info_sorted, gamma_item, factors, bins: int):
    apply_eurosys_style()

    x_min = max(0.0, np.percentile(sample, 0.1) - 1.0)
    x_max = np.percentile(sample, 99.9) + 1.0
    x = np.linspace(x_min, x_max, 1400)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.12, wspace=0.16)

    # Left panel: fit distributions
    ax = axes[0]
    ax.hist(
        sample,
        bins=bins,
        density=True,
        alpha=0.52,
        color="#A7C4D8",
        edgecolor="#5C6670",
        linewidth=0.25,
        label="Histogram",
    )

    colors = {
        "normal": "#C44E52",
        "lognormal": "#DD8452",
        "gamma": "#4C72B0",
    }

    for name, params, k, log_likelihood, dist_obj in fit_info_sorted:
        current_aic = aic(k, log_likelihood)
        y = dist_obj.pdf(x, *params)
        ax.plot(x, y, linewidth=2.0, color=colors.get(name, "black"), label=f"{name} (AIC={current_aic:.1f})")

    ax.set_title("(a) Distribution Fitting")
    ax.set_xlabel("Ratio value")
    ax.set_ylabel("Density")
    ax.yaxis.labelpad = 10
    ax.grid(True, linestyle="--")
    ax.legend(frameon=True, framealpha=0.96, edgecolor="#BBBBBB")

    # Right panel: scaled gamma curves using the fitted gamma parameters.
    ax = axes[1]
    base_k, base_loc, base_theta = gamma_item[1]
    ax.hist(
        sample,
        bins=bins,
        density=True,
        alpha=0.40,
        color="#D9D9D9",
        edgecolor="#5C6670",
        linewidth=0.25,
        label="Histogram",
    )

    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(factors):
        theta = base_theta * f
        y = gamma_dist.pdf(x, a=base_k, loc=base_loc, scale=theta)
        mean = base_k * theta + base_loc
        std = np.sqrt(base_k) * theta
        label = f"factor={f:g}, mean={mean:.2f}, std={std:.2f}"
        lw = 2.3 if abs(f - 1.0) < 1e-12 else 1.55
        color = "#C44E52" if abs(f - 1.0) < 1e-12 else cmap(i % 10)
        ax.plot(x, y, linewidth=lw, color=color, label=label)

    ax.set_title("(b) Gamma Scaling")
    ax.set_xlabel("Ratio value")
    ax.set_ylabel("Density")
    ax.yaxis.labelpad = 16
    ax.grid(True, linestyle="--")
    ax.legend(frameon=True, framealpha=0.96, edgecolor="#BBBBBB", ncol=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit distributions first, then scale gamma and plot in one figure.")
    parser.add_argument("--input", required=True, help="Path to txt file containing space-separated numbers")
    parser.add_argument("--output", required=True, help="Path to output figure (.png or .pdf)")
    parser.add_argument("--summary", required=True, help="Path to output combined summary (.txt)")
    parser.add_argument(
        "--factors",
        default="0.08,0.24,0.40,0.56,0.72,1.0,1.2,1.4,1.6,1.8,2.0",
        help="Comma-separated multipliers applied to gamma scale",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    factors = parse_factors(args.factors)

    sample = load_numbers(input_path)
    fit_info = fit_distributions(sample)
    fit_info_sorted = sorted(fit_info, key=lambda item: aic(item[2], item[3]))
    gamma_item = select_gamma_fit(fit_info_sorted)

    fig = build_figure(sample, fit_info_sorted, gamma_item, factors, args.bins)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_path if output_path.suffix.lower() == ".png" else output_path.with_suffix(".png")
    pdf_path = output_path if output_path.suffix.lower() == ".pdf" else output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=340)
    fig.savefig(pdf_path)
    plt.close(fig)

    base_k, base_loc, base_theta = gamma_item[1]
    summary_lines = [
        f"input={input_path}",
        f"count={sample.size}",
        f"min={sample.min():.6f}",
        f"max={sample.max():.6f}",
        f"mean={sample.mean():.6f}",
        f"std={sample.std(ddof=1):.6f}",
        f"p50={np.percentile(sample, 50):.6f}",
        f"p90={np.percentile(sample, 90):.6f}",
        f"p99={np.percentile(sample, 99):.6f}",
        "",
        "[fitted_distributions]",
    ]

    for name, params, k, log_likelihood, dist_obj in fit_info_sorted:
        current_aic = aic(k, log_likelihood)
        summary_lines.append(
            f"{name}: params={tuple(float(p) for p in params)}, logL={log_likelihood:.6f}, AIC={current_aic:.6f}"
        )

    summary_lines.extend(
        [
            f"best_by_aic={fit_info_sorted[0][0]}",
            f"selected_gamma_shape={base_k:.12f}",
            f"selected_gamma_loc={base_loc:.6f}",
            f"selected_gamma_scale={base_theta:.12f}",
            "",
            "[scaled_curves]",
        ]
    )

    for f in factors:
        theta = base_theta * f
        mean = base_k * theta + base_loc
        std = np.sqrt(base_k) * theta
        cv = std / mean if mean != 0 else np.nan
        summary_lines.append(
            f"factor={f:g}, shape={base_k:.12f}, loc={base_loc:.6f}, scale={theta:.12f}, mean={mean:.6f}, std={std:.6f}, cv={cv:.6f}"
        )

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()