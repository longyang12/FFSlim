"""Small runner to convert the MSCoco FFSlim to various formats.

Usage examples:
python run_convert_mscoco.py --input /mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr --outdir /mnt/datasets/mscoco_converted --formats webdataset,tfrecord,ffrecord,tdp

"""
# /home/llm/experiment/dataset_Analysis_Modeling/.env/bin/python /home/llm/experiment/dataset_Analysis_Modeling/generaters/run_convert_mscoco.py \
#   --input /mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr \
#   --outdir /mnt/datasets/mscoco_converted \
#   --dataset-name mscoco \
#   --formats webdataset,tfrecord,ffrecord,tdp \
#   --variants B,F


import os
import argparse
import sys


def _ensure_repo_on_path():
    # Allow running the script directly via absolute path.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _output_base(dataset_name, fmt, variant):
    return f"{dataset_name}_{fmt}_{variant}"


def _should_skip_dir(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def main():
    _ensure_repo_on_path()
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dataset-name", default="mscoco")
    p.add_argument("--formats", default="webdataset,tfrecord,ffrecord,tdp")
    p.add_argument("--variants", default="B,F")
    p.add_argument("--shard-size", type=int, default=1000)
    args = p.parse_args()

    ensure_dir(args.outdir)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]

    input_ffr = args.input

    for variant in variants:
        per_caption = variant == "F"
        if "webdataset" in formats:
            from generaters.FFSlim2WebDataset import ffslim_to_webdataset

            out = os.path.join(args.outdir, _output_base(args.dataset_name, "webdataset", variant))
            if _should_skip_dir(out):
                print("Skipping existing webdataset ->", out)
            else:
                print("Creating webdataset ->", out)
                ffslim_to_webdataset(
                    input_ffr,
                    out,
                    shard_size=args.shard_size,
                    per_caption=per_caption,
                )

        if "tfrecord" in formats:
            from generaters.FFSlim2TFRecord import ffslim_to_tfrecord

            out_dir = os.path.join(args.outdir, _output_base(args.dataset_name, "tfrecord", variant))
            out = os.path.join(out_dir, _output_base(args.dataset_name, "tfrecord", variant) + ".tfrecord")
            if _should_skip_dir(out_dir):
                print("Skipping existing TFRecord ->", out)
            else:
                print("Creating TFRecord ->", out)
                ensure_dir(out_dir)
                ffslim_to_tfrecord(input_ffr, out, per_caption=per_caption)

        if "ffrecord" in formats:
            from generaters.FFSlim2FFRecord import ffslim_to_ffrecord

            out_dir = os.path.join(args.outdir, _output_base(args.dataset_name, "ffrecord", variant))
            out = os.path.join(out_dir, _output_base(args.dataset_name, "ffrecord", variant) + ".ffr")
            if _should_skip_dir(out_dir):
                print("Skipping existing FFRecord ->", out)
            else:
                print("Creating FFRecord ->", out)
                ensure_dir(out_dir)
                ffslim_to_ffrecord(input_ffr, out, per_caption=per_caption)

        if "tdp" in formats:
            from generaters.FFSlim2TDP import ffslim_to_tdp

            out = os.path.join(args.outdir, _output_base(args.dataset_name, "tdp", variant))
            if _should_skip_dir(out):
                print("Skipping existing TDP ->", out)
            else:
                print("Creating TDP ->", out)
                ffslim_to_tdp(
                    input_ffr,
                    out,
                    _output_base(args.dataset_name, "tdp", variant),
                    per_caption=per_caption,
                )


if __name__ == "__main__":
    main()