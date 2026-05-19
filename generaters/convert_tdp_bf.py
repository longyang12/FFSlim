import argparse
import json
import os
import time

import pandas as pd


def _find_tdp_csv(input_dir, input_csv):
    if input_csv:
        return input_csv
    matches = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith("_TDP.csv")
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one *_TDP.csv in input dir")
    return matches[0]


def _default_output_csv(output_dir):
    base = os.path.basename(os.path.normpath(output_dir))
    return os.path.join(output_dir, base + "_TDP.csv")


def _ensure_image_dir(input_dir, output_dir, link_images):
    input_img = os.path.join(input_dir, "img")
    output_img = os.path.join(output_dir, "img")
    if os.path.exists(output_img):
        return
    if link_images:
        try:
            os.symlink(input_img, output_img)
            return
        except Exception:
            pass
    raise RuntimeError("Output img dir missing; create link or copy images")


def _write_log(log_path, dataset, data_format, mode, stats, elapsed):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    line = (
        f"{dataset} {data_format} {mode} "
        f"{stats['input_records']} {stats['output_records']} "
        f"{stats['image_count']} {elapsed:.4f}\n"
    )
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(line)


def b2f(input_dir, output_dir, input_csv=None, output_csv=None, link_images=True):
    os.makedirs(output_dir, exist_ok=True)
    input_csv = _find_tdp_csv(input_dir, input_csv)
    output_csv = output_csv or _default_output_csv(output_dir)

    df = pd.read_csv(input_csv)
    rows = []
    input_records = len(df)
    input_captions = 0

    for _, r in df.iterrows():
        captions = json.loads(r["image_caption"])
        input_captions += len(captions)
        for cap in captions:
            rows.append({"image_path": r["image_path"], "image_caption": cap})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    _ensure_image_dir(input_dir, output_dir, link_images)

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": len(out_df),
        "output_captions": len(out_df),
        "image_count": out_df["image_path"].nunique(),
    }


def f2b(input_dir, output_dir, input_csv=None, output_csv=None, link_images=True):
    os.makedirs(output_dir, exist_ok=True)
    input_csv = _find_tdp_csv(input_dir, input_csv)
    output_csv = output_csv or _default_output_csv(output_dir)

    df = pd.read_csv(input_csv)
    grouped = {}
    input_records = len(df)
    input_captions = len(df)

    for _, r in df.iterrows():
        key = r["image_path"]
        grouped.setdefault(key, []).append(r["image_caption"])

    rows = []
    for key in grouped:
        rows.append({
            "image_path": key,
            "image_caption": json.dumps(grouped[key], ensure_ascii=True),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    _ensure_image_dir(input_dir, output_dir, link_images)

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": len(out_df),
        "output_captions": sum(len(grouped[k]) for k in grouped),
        "image_count": len(grouped),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["B2F", "F2B"])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--input-csv")
    p.add_argument("--output-csv")
    p.add_argument("--dataset", default="unknown")
    p.add_argument("--format", dest="data_format", default="tdp")
    p.add_argument("--log-file")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--link-images", action="store_true", default=True)
    group.add_argument("--no-link-images", action="store_true", default=False)
    args = p.parse_args()

    link_images = args.link_images and not args.no_link_images

    start = time.perf_counter()
    if args.mode == "B2F":
        stats = b2f(args.input_dir, args.output_dir, args.input_csv, args.output_csv, link_images)
    else:
        stats = f2b(args.input_dir, args.output_dir, args.input_csv, args.output_csv, link_images)
    elapsed = time.perf_counter() - start

    stats_line = (
        f"mode={args.mode} "
        f"input_records={stats['input_records']} "
        f"input_captions={stats['input_captions']} "
        f"output_records={stats['output_records']} "
        f"output_captions={stats['output_captions']} "
        f"images={stats['image_count']} "
        f"elapsed_sec={elapsed:.4f}"
    )
    print(stats_line)
    _write_log(args.log_file, args.dataset, args.data_format, args.mode, stats, elapsed)


if __name__ == "__main__":
    main()
