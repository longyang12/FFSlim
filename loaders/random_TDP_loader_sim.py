import argparse
import json
import os
import random
import time

import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cache.tdp_lru import LRUCache
from cache.tdp_quiver import Quiver
from cache.tdp_coordl import CoorDL
from cache.tdp_iacma import IACMACache
from cache.tdp_iacma_with_memory import IACMACacheWithMemory
from cache.tdp_iacma_with_heap import IACMACacheWithHeap
from cache.tdp_iacma_admission import IACMAAdmission
from cache.tdp_iacma_with_memory_admission import IACMACacheWithMemoryAdmission


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SIM_LOG_PATH = os.path.join(BASE_DIR, 'output_sim.log')
SIM_OBJECT_LOG_DIR = os.path.join(BASE_DIR, 'object_importance_log')


def load_manifest(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_cache(cache_algorithm: str, num: int, images):
    if cache_algorithm == 'lru':
        return LRUCache(num)
    if cache_algorithm == 'quiver':
        return Quiver(num)
    if cache_algorithm == 'coordl':
        return CoorDL(num)
    if cache_algorithm == 'iacma':
        return IACMACache(num, images)
    if cache_algorithm == 'iacma_with_memory':
        return IACMACacheWithMemory(num, images)
    if cache_algorithm == 'iacma_with_heap':
        return IACMACacheWithHeap(num, images)
    if cache_algorithm == 'iacma_admission':
        return IACMAAdmission(num, images)
    if cache_algorithm == 'iacma_with_memory_admission':
        return IACMACacheWithMemoryAdmission(num, images)
    raise ValueError(f'Unknown cache algorithm: {cache_algorithm}')


def resolve_tdp_paths(manifest):
    csv_path = manifest.get('csv_file')
    img_dir = manifest.get('img_dir')
    root_dir = manifest.get('root_dir')

    if not csv_path and root_dir:
        csv_path = os.path.join(root_dir, f"{os.path.basename(root_dir)}.csv")
    if not img_dir and root_dir:
        img_dir = os.path.join(root_dir, 'img')

    if not csv_path or not img_dir:
        raise ValueError('Manifest missing csv_file/img_dir (or root_dir fallback) for TDP')
    return csv_path, img_dir


def process_samples(args):
    manifest = load_manifest(args.manifest)
    csv_path, img_dir = resolve_tdp_paths(manifest)

    df = pd.read_csv(csv_path)
    images = df['image_path']
    captions = df['image_caption']

    sample_num = manifest.get('sample_num')
    if sample_num is None:
        sample_num = manifest.get('realized_stats', {}).get('sample_num', len(df))
    sample_num = min(int(sample_num), len(df))

    request_count = min(sample_num, args.request_count) if args.request_count > 0 else sample_num

    sample_ids = list(range(sample_num))
    random.seed(args.shuffle_seed)
    random.shuffle(sample_ids)
    sample_ids = sample_ids[:request_count]

    cache = None
    hits = 0
    cache_get_calls = 0
    cache_put_calls = 0
    cache_get_overhead = 0.0
    cache_put_overhead = 0.0

    if args.num > 0:
        cache = build_cache(args.cache_algorithm, args.num, images)

    process_begin = time.perf_counter()

    for req_id, sid in enumerate(sample_ids):
        image_key = images[sid].split('.')[0]

        if cache is not None:
            cache_get_start = time.perf_counter()
            img_data = cache.get(image_key, req_id)
            cache_get_overhead += time.perf_counter() - cache_get_start
            cache_get_calls += 1
            if img_data != -1:
                hits += 1
                _ = (img_data, captions[sid])
                continue

        img_path = os.path.join(img_dir, images[sid])
        with open(img_path, 'rb') as f:
            img_data = f.read()

        if cache is not None:
            cache_put_start = time.perf_counter()
            cache.put(image_key, img_data, req_id)
            cache_put_overhead += time.perf_counter() - cache_put_start
            cache_put_calls += 1

        _ = (img_data, captions[sid])

    process_total = time.perf_counter() - process_begin

    os.makedirs(SIM_OBJECT_LOG_DIR, exist_ok=True)
    with open(SIM_LOG_PATH, 'a+', encoding='utf-8') as f:
        f.write(
            f"[SIM][TDP] dataset={manifest.get('dataset_name','unknown')},requests={request_count},"
            f"cache={args.cache_algorithm},cache_size={args.num},manifest={args.manifest}\n"
        )
        f.write(f"sample_num={sample_num},csv={csv_path},img_dir={img_dir}\n")
        if cache is not None and request_count > 0:
            hit_rate = hits / request_count
            cache_overhead_total = cache_get_overhead + cache_put_overhead
            avg_get_us = (cache_get_overhead / cache_get_calls * 1e6) if cache_get_calls else 0.0
            avg_put_us = (cache_put_overhead / cache_put_calls * 1e6) if cache_put_calls else 0.0
            f.write(
                f"cache hits:{hits},requests:{request_count},cache hit rate:{hit_rate}\n"
                f"cache mgmt overhead:get_calls={cache_get_calls},put_calls={cache_put_calls},"
                f"get_total={cache_get_overhead:.6f}s,put_total={cache_put_overhead:.6f}s,"
                f"avg_get={avg_get_us:.2f}us,avg_put={avg_put_us:.2f}us,"
                f"cache_total={cache_overhead_total:.6f}s,process_total={process_total:.6f}s,"
                f"cache_overhead_ratio={(cache_overhead_total / process_total) if process_total > 0 else 0:.6f}\n"
            )

    if cache is not None and args.cache_algorithm in [
        'iacma',
        'iacma_with_memory',
        'iacma_admission',
        'iacma_with_memory_admission',
    ]:
        log_name = f"simulated_{manifest.get('dataset_name', 'unknown')}_{args.num}_{args.cache_algorithm}_object_importance.log"
        obj_log_path = os.path.join(SIM_OBJECT_LOG_DIR, log_name)
        with open(obj_log_path, 'w', encoding='utf-8') as f:
            for key, score_shifts in cache.get_object_importance().items():
                f.write(f"key:{key},score_shifts:{score_shifts}\n")


def build_parser():
    parser = argparse.ArgumentParser(description='Random read benchmark for simulated TDP datasets')
    parser.add_argument('--manifest', type=str, required=True, help='Path to simulated dataset manifest JSON')
    parser.add_argument('--num', type=int, default=0, help='Cache size')
    parser.add_argument('--cache-algorithm', type=str, default='lru', help='Cache algorithm')
    parser.add_argument('--request-count', type=int, default=0, help='Request count (0 means all samples)')
    parser.add_argument('--shuffle-seed', type=int, default=123, help='Shuffle seed')
    parser.add_argument('--memory', type=int, default=0, help='Reserved for cgroup compatibility')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    process_samples(args)
