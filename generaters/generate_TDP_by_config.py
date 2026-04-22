"""Generate a synthetic TDP dataset from a statistics config.

Generated TDP layout:
- <output_root>/img/<id>.jpg
- <output_root>/<csv_file>

CSV columns are compatible with existing loaders:
- image_path
- image_caption
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import string
from collections import defaultdict
from typing import Any, Dict
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path.lower())
    with open(config_path, "r", encoding="utf-8") as f:
        if ext in {".json"}:
            return json.load(f)

        if ext in {".yml", ".yaml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read YAML configs") from exc

            return yaml.safe_load(f)

    raise ValueError("Unsupported config format. Use .json, .yml, or .yaml")


def sample_positive_int(rng: random.Random, mean: float, std: float, minimum: int = 1) -> int:
    if std <= 0:
        return max(minimum, int(round(mean)))
    return max(minimum, int(round(rng.gauss(mean, std))))


def sample_caption_count(
    rng: random.Random,
    np_rng: np.random.Generator,
    mean: float,
    std: float,
    ratio_model: str,
    ratio_support_shift: int,
    ratio_nb_r: float,
    ratio_nb_p: float,
    ratio_gamma_shape: float,
    ratio_gamma_scale: float,
) -> int:
    if ratio_model == "zt_nbinom":
        if ratio_nb_r <= 0 or not (0 < ratio_nb_p < 1):
            raise ValueError("zt_nbinom requires ratio_nb_r > 0 and 0 < ratio_nb_p < 1")
        for _ in range(10000):
            k = int(np_rng.negative_binomial(ratio_nb_r, ratio_nb_p))
            if k > 0:
                return max(1, k + ratio_support_shift)
        return max(1, 1 + ratio_support_shift)

    if ratio_model == "gamma":
        if ratio_gamma_shape <= 0 or ratio_gamma_scale <= 0:
            raise ValueError("gamma requires ratio_gamma_shape > 0 and ratio_gamma_scale > 0")
        sampled = float(np_rng.gamma(shape=ratio_gamma_shape, scale=ratio_gamma_scale))
        return max(1, int(round(sampled)) + ratio_support_shift)

    return sample_positive_int(rng, mean, std)


def build_text(rng: random.Random, target_length: int) -> str:
    if target_length <= 0:
        return ""

    letters = string.ascii_lowercase
    chunks = []
    remaining = target_length

    while remaining > 0:
        chunk_length = min(remaining, rng.randint(1, 12))
        chunks.append("".join(rng.choice(letters) for _ in range(chunk_length)))
        remaining -= chunk_length
        if remaining > 0:
            chunks.append(" ")
            remaining -= 1

    text = "".join(chunks)
    if len(text) < target_length:
        text += "a" * (target_length - len(text))
    return text[:target_length]


def safe_pstdev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None, required: bool = False) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]

    if required:
        raise KeyError(f"Missing required config key: one of {keys}")

    return default


def generate_tdp_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    media_num = int(get_config_value(config, "media_num", "sample_num", required=True))
    avg_quantity_ratio = float(get_config_value(config, "avg_quantity_ratio", required=True))
    std_quantity_ratio = float(get_config_value(config, "std_quantity_ratio", default=0.0))
    avg_media_size = float(get_config_value(config, "avg_media_size", required=True))
    std_media_size = float(get_config_value(config, "std_media_size", default=0.0))
    avg_text_size = float(get_config_value(config, "avg_text_size", required=True))
    std_text_size = float(get_config_value(config, "std_text_size", default=0.0))
    ratio_model = str(get_config_value(config, "ratio_model", default="gaussian"))
    ratio_support_shift = int(get_config_value(config, "ratio_support_shift", default=0))
    ratio_nb_r = float(get_config_value(config, "ratio_nb_r", default=0.0))
    ratio_nb_p = float(get_config_value(config, "ratio_nb_p", default=0.0))
    ratio_gamma_shape = float(get_config_value(config, "ratio_gamma_shape", default=0.0))
    ratio_gamma_scale = float(get_config_value(config, "ratio_gamma_scale", default=0.0))

    output_root = str(get_config_value(config, "output_root", required=True))
    dataset_name = str(get_config_value(config, "dataset_name", default="generated_tdp"))
    manifest_file = str(get_config_value(config, "manifest_file", default="generated_manifest.json"))
    ratio_nums_file = str(get_config_value(config, "ratio_nums_file", default="generated_ratio_nums.txt"))
    seed = int(get_config_value(config, "seed", default=42))

    default_csv = f"{os.path.basename(output_root.rstrip('/'))}.csv"
    csv_file = str(get_config_value(config, "csv_file", default=default_csv))

    if media_num <= 0:
        raise ValueError("media_num must be greater than 0")
    if avg_quantity_ratio <= 0:
        raise ValueError("avg_quantity_ratio must be greater than 0")
    if avg_media_size <= 0:
        raise ValueError("avg_media_size must be greater than 0")
    if avg_text_size <= 0:
        raise ValueError("avg_text_size must be greater than 0")
    if std_quantity_ratio < 0 or std_media_size < 0 or std_text_size < 0:
        raise ValueError("Standard deviations must be non-negative")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    os.makedirs(output_root, exist_ok=True)
    img_dir = os.path.join(output_root, "img")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(output_root, csv_file)
    ratio_nums_path = os.path.join(output_root, ratio_nums_file)

    media_sizes = []
    caption_counts = []
    text_sizes = []
    sample_sizes = []

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_fp:
        writer = csv.writer(csv_fp)
        writer.writerow(["image_path", "image_caption"])

        for media_id in range(media_num):
            caption_count = sample_caption_count(
                rng=rng,
                np_rng=np_rng,
                mean=avg_quantity_ratio,
                std=std_quantity_ratio,
                ratio_model=ratio_model,
                ratio_support_shift=ratio_support_shift,
                ratio_nb_r=ratio_nb_r,
                ratio_nb_p=ratio_nb_p,
                ratio_gamma_shape=ratio_gamma_shape,
                ratio_gamma_scale=ratio_gamma_scale,
            )
            media_size = sample_positive_int(rng, avg_media_size, std_media_size)

            image_name = f"{media_id}.jpg"
            image_path = os.path.join(img_dir, image_name)
            with open(image_path, "wb") as f:
                f.write(os.urandom(media_size))

            media_sizes.append(media_size)
            caption_counts.append(caption_count)

            for _ in range(caption_count):
                text_size = sample_positive_int(rng, avg_text_size, std_text_size)
                caption = build_text(rng, text_size)
                writer.writerow([image_name, caption])

                text_sizes.append(text_size)
                sample_sizes.append(media_size + text_size)

    realized_stats = {
        "sample_num": len(sample_sizes),
        "media_num": len(media_sizes),
        "avg_quantity_ratio": round(sum(caption_counts) / len(caption_counts), 4),
        "std_quantity_ratio": round(safe_pstdev(caption_counts), 4),
        "avg_media_size": round(sum(media_sizes) / len(media_sizes), 4),
        "std_media_size": round(safe_pstdev(media_sizes), 4),
        "avg_text_size": round(sum(text_sizes) / len(text_sizes), 4) if text_sizes else 0.0,
        "std_text_size": round(safe_pstdev(text_sizes), 4),
        "avg_sample_size": round(sum(sample_sizes) / len(sample_sizes), 4) if sample_sizes else 0.0,
        "std_sample_size": round(safe_pstdev(sample_sizes), 4),
    }

    target_stats = {
        "media_num": media_num,
        "avg_quantity_ratio": avg_quantity_ratio,
        "std_quantity_ratio": std_quantity_ratio,
        "avg_media_size": avg_media_size,
        "std_media_size": std_media_size,
        "avg_text_size": avg_text_size,
        "std_text_size": std_text_size,
    }

    manifest = {
        "dataset_name": dataset_name,
        "seed": seed,
        "ratio_model": ratio_model,
        "ratio_support_shift": ratio_support_shift,
        "ratio_nb_r": ratio_nb_r,
        "ratio_nb_p": ratio_nb_p,
        "ratio_gamma_shape": ratio_gamma_shape,
        "ratio_gamma_scale": ratio_gamma_scale,
        "csv_file": csv_path,
        "img_dir": img_dir,
        "ratio_nums_file": ratio_nums_path,
        "target_stats": target_stats,
        "realized_stats": realized_stats,
    }

    with open(ratio_nums_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, caption_counts)))

    manifest_path = os.path.join(output_root, manifest_file)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"TDP dataset written: {output_root}")
    print(f"CSV written: {csv_path}")
    print(f"Ratio nums written: {ratio_nums_path}")
    print(f"Manifest written: {manifest_path}")
    print(
        f"media_num={realized_stats['media_num']}, sample_num={realized_stats['sample_num']}, "
        f"avg_quantity_ratio={realized_stats['avg_quantity_ratio']}, std_quantity_ratio={realized_stats['std_quantity_ratio']}, "
        f"avg_media_size={realized_stats['avg_media_size']}, std_media_size={realized_stats['std_media_size']}, "
        f"avg_text_size={realized_stats['avg_text_size']}, std_text_size={realized_stats['std_text_size']}"
    )

    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a synthetic TDP dataset from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON/YAML config")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config = load_config(args.config)
    generate_tdp_dataset(config)
