import argparse
import csv
import os
from datetime import datetime

# /home/llm/experiment/dataset_Analysis_Modeling/.env/bin/python analyze/plot_monitor.py --input /home/llm/experiment/dataset_Analysis_Modeling
# /monitor


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def read_monitor_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ts_ms": _safe_int(row.get("ts_ms")),
                    "rss_kb": _safe_int(row.get("rss_kb")),
                    "anon_kb": _safe_int(row.get("anon_kb")),
                    "file_kb": _safe_int(row.get("file_kb")),
                    "cpu_pct": _safe_float(row.get("cpu_pct")),
                    "read_bps": _safe_int(row.get("read_bps")),
                    "write_bps": _safe_int(row.get("write_bps")),
                }
            )
    return rows


def summarize(rows):
    if not rows:
        return {}
    rss_peak = max(rows, key=lambda r: r["rss_kb"])
    anon_peak = max(rows, key=lambda r: r["anon_kb"])
    file_peak = max(rows, key=lambda r: r["file_kb"])
    cpu_peak = max(rows, key=lambda r: r["cpu_pct"])
    read_peak = max(rows, key=lambda r: r["read_bps"])
    write_peak = max(rows, key=lambda r: r["write_bps"])

    return {
        "count": len(rows),
        "rss_peak_kb": rss_peak["rss_kb"],
        "rss_peak_ts": rss_peak["ts_ms"],
        "anon_peak_kb": anon_peak["anon_kb"],
        "anon_peak_ts": anon_peak["ts_ms"],
        "file_peak_kb": file_peak["file_kb"],
        "file_peak_ts": file_peak["ts_ms"],
        "cpu_peak_pct": cpu_peak["cpu_pct"],
        "cpu_peak_ts": cpu_peak["ts_ms"],
        "read_peak_bps": read_peak["read_bps"],
        "read_peak_ts": read_peak["ts_ms"],
        "write_peak_bps": write_peak["write_bps"],
        "write_peak_ts": write_peak["ts_ms"],
    }


def ts_to_iso(ts_ms):
    if not ts_ms:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000.0).isoformat(timespec="seconds")


def plot(rows, out_dir, title_prefix="monitor"):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to plot") from exc

    ts = [r["ts_ms"] for r in rows]
    t0 = ts[0] if ts else 0
    tsec = [(t - t0) / 1000.0 for t in ts]

    rss_mb = [r["rss_kb"] / 1024.0 for r in rows]
    anon_mb = [r["anon_kb"] / 1024.0 for r in rows]
    file_mb = [r["file_kb"] / 1024.0 for r in rows]
    cpu = [r["cpu_pct"] for r in rows]
    read_mb = [r["read_bps"] / (1024.0 * 1024.0) for r in rows]
    write_mb = [r["write_bps"] / (1024.0 * 1024.0) for r in rows]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(tsec, rss_mb, color="#1f77b4", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("RSS (MB)")
    plt.title(f"{title_prefix} - RSS")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(tsec, anon_mb, color="#17becf", linewidth=1, label="anon")
    plt.plot(tsec, file_mb, color="#9467bd", linewidth=1, label="file")
    plt.xlabel("time (s)")
    plt.ylabel("Memory (MB)")
    plt.title(f"{title_prefix} - anon/file")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "anon_file.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(tsec, cpu, color="#ff7f0e", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("CPU (%)")
    plt.title(f"{title_prefix} - CPU")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cpu.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(tsec, read_mb, color="#2ca02c", linewidth=1, label="read")
    plt.plot(tsec, write_mb, color="#d62728", linewidth=1, label="write")
    plt.xlabel("time (s)")
    plt.ylabel("MB/s")
    plt.title(f"{title_prefix} - IO bandwidth")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "io.png"), dpi=150)
    plt.close()


def _iter_inputs(input_path):
    if os.path.isdir(input_path):
        for name in sorted(os.listdir(input_path)):
            if name.endswith(".csv"):
                yield os.path.join(input_path, name)
    else:
        yield input_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to monitor CSV or directory")
    parser.add_argument(
        "--outdir",
        default="/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/monitor_plots",
    )
    parser.add_argument("--title", default="monitor")
    parser.add_argument(
        "--summary-xlsx",
        default=None,
        help="Write per-file peaks to an Excel file (default: <outdir>/monitor_summary.xlsx)",
    )
    args = parser.parse_args()

    summaries = []
    for path in _iter_inputs(args.input):
        rows = read_monitor_csv(path)
        if not rows:
            print(f"skip empty: {path}")
            continue

        title = f"{args.title}_{os.path.splitext(os.path.basename(path))[0]}"
        out_dir = os.path.join(args.outdir, os.path.splitext(os.path.basename(path))[0])

        summary = summarize(rows)
        print(f"file {path}")
        print("samples", summary.get("count", 0))
        print("rss_peak_kb", summary.get("rss_peak_kb", 0), ts_to_iso(summary.get("rss_peak_ts")))
        print("anon_peak_kb", summary.get("anon_peak_kb", 0), ts_to_iso(summary.get("anon_peak_ts")))
        print("file_peak_kb", summary.get("file_peak_kb", 0), ts_to_iso(summary.get("file_peak_ts")))
        print("cpu_peak_pct", summary.get("cpu_peak_pct", 0.0), ts_to_iso(summary.get("cpu_peak_ts")))
        print("read_peak_bps", summary.get("read_peak_bps", 0), ts_to_iso(summary.get("read_peak_ts")))
        print("write_peak_bps", summary.get("write_peak_bps", 0), ts_to_iso(summary.get("write_peak_ts")))

        summaries.append(
            {
                "file": os.path.basename(path),
                "samples": summary.get("count", 0),
                "anon_peak_kb": summary.get("anon_peak_kb", 0),
                "file_peak_kb": summary.get("file_peak_kb", 0),
                "cpu_peak_pct": summary.get("cpu_peak_pct", 0.0),
                "read_peak_bps": summary.get("read_peak_bps", 0),
            }
        )

        plot(rows, out_dir, title)

    if summaries:
        out_path = args.summary_xlsx
        if out_path is None:
            out_path = os.path.join(args.outdir, "monitor_summary.xlsx")
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("pandas is required to write Excel summaries") from exc

        df = pd.DataFrame(
            summaries,
            columns=[
                "file",
                "samples",
                "anon_peak_kb",
                "file_peak_kb",
                "cpu_peak_pct",
                "read_peak_bps",
            ],
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_excel(out_path, index=False)
        print(f"summary_xlsx {out_path}")


if __name__ == "__main__":
    main()
