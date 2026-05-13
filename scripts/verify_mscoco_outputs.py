import argparse
import json
import os
import pickle
from collections import defaultdict


def _count_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_tdp_rows(csv_path):
    import csv

    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
    return count


def _compute_expected(input_ffr):
    from ffrecord import FileReader

    reader = FileReader(input_ffr)
    expected_b = reader.n
    expected_f = 0
    bad = 0
    for i in range(reader.n):
        samples = reader.read([i])
        for sample_bytes in samples:
            try:
                obj = pickle.loads(sample_bytes)
            except Exception:
                bad += 1
                continue
            captions = obj.get("captions") or obj.get(b"captions") or []
            if isinstance(captions, (bytes, str)):
                expected_f += 1
            else:
                expected_f += len(captions)
    return expected_b, expected_f, bad


def _load_counts(dataset_name, outdir):
    counts = defaultdict(dict)
    for variant in ["B", "F"]:
        # webdataset
        wd_dir = os.path.join(outdir, f"{dataset_name}_webdataset_{variant}")
        wd_index = os.path.join(wd_dir, "index.jsonl")
        counts[f"webdataset_{variant}"] = {
            "exists": os.path.exists(wd_index),
            "count": _count_lines(wd_index) if os.path.exists(wd_index) else None,
            "path": wd_index,
        }
        # tfrecord
        tf_base = f"{dataset_name}_tfrecord_{variant}"
        tf_dir = os.path.join(outdir, tf_base)
        tf_index = os.path.join(tf_dir, tf_base + ".tfrecord.index.jsonl")
        counts[f"tfrecord_{variant}"] = {
            "exists": os.path.exists(tf_index),
            "count": _count_lines(tf_index) if os.path.exists(tf_index) else None,
            "path": tf_index,
        }
        # ffrecord
        ffr_base = f"{dataset_name}_ffrecord_{variant}"
        ffr_dir = os.path.join(outdir, ffr_base)
        ffr_path = os.path.join(ffr_dir, ffr_base + ".ffr")
        if os.path.exists(ffr_path):
            try:
                from ffrecord import FileReader

                reader = FileReader(ffr_path)
                ffr_count = reader.n
            except Exception as exc:
                ffr_count = None
                counts[f"ffrecord_{variant}"] = {
                    "exists": True,
                    "count": None,
                    "path": ffr_path,
                    "error": str(exc),
                }
            else:
                counts[f"ffrecord_{variant}"] = {
                    "exists": True,
                    "count": ffr_count,
                    "path": ffr_path,
                }
        else:
            counts[f"ffrecord_{variant}"] = {
                "exists": False,
                "count": None,
                "path": ffr_path,
            }
        # tdp
        tdp_dir = os.path.join(outdir, f"{dataset_name}_tdp_{variant}")
        tdp_csv = os.path.join(tdp_dir, f"{dataset_name}_tdp_{variant}_TDP.csv")
        counts[f"tdp_{variant}"] = {
            "exists": os.path.exists(tdp_csv),
            "count": _count_tdp_rows(tdp_csv) if os.path.exists(tdp_csv) else None,
            "path": tdp_csv,
        }
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input FFSlim .ffr")
    parser.add_argument("--outdir", required=True, help="Output root directory")
    parser.add_argument("--dataset-name", default="mscoco")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    expected_b, expected_f, bad = _compute_expected(args.input)
    counts = _load_counts(args.dataset_name, args.outdir)

    summary = {
        "dataset": args.dataset_name,
        "expected": {"B": expected_b, "F": expected_f, "bad_samples": bad},
        "counts": counts,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(f"dataset={args.dataset_name}")
        print(f"expected_B={expected_b}, expected_F={expected_f}, bad_samples={bad}")
        for key in sorted(counts.keys()):
            info = counts[key]
            print(f"{key}: exists={info['exists']} count={info['count']} path={info['path']}")

    mismatches = []
    for key, info in counts.items():
        if not info.get("exists"):
            mismatches.append((key, "missing"))
            continue
        if key.endswith("_B"):
            if info.get("count") != expected_b:
                mismatches.append((key, info.get("count")))
        if key.endswith("_F"):
            if info.get("count") != expected_f:
                mismatches.append((key, info.get("count")))

    if mismatches:
        print("MISMATCH:")
        for key, value in mismatches:
            print(f"  {key}: {value}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
