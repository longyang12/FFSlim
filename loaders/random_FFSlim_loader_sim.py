import argparse
import json
import os
import pickle
import random
import time
from typing import Dict, Tuple

from ffrecord import FileReader

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cache.ffslim_lru import LRUCache
from cache.ffslim_quiver import Quiver
from cache.ffslim_coordl import CoorDL
from cache.ffslim_dgim_lru import DGIMLRUCache
from cache.ffslim_dgim_quiver import DGIMQuiver
from cache.ffslim_dgim_coordl import DGIMCoorDL
from cache.ffslim_dgim_iacma import DGIMIACMACache
from cache.ffslim_dgim_iacma_with_memory import DGIMIACMACacheWithMemory
from cache.ffslim_dgim_iacma_admission import DGIMIACMAAdmissionCache
from cache.ffslim_dgim_iacma_with_memory_admission import DGIMIACMAAdmissionCacheWithMemory


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SIM_LOG_PATH = os.path.join(BASE_DIR, 'output_sim.log')
SIM_OBJECT_LOG_DIR = os.path.join(BASE_DIR, 'object_importance_log')


def load_manifest(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_sample_num(manifest: Dict, hash_index: Dict) -> int:
    if 'sample_num' in manifest:
        return int(manifest['sample_num'])
    realized = manifest.get('realized_stats', {})
    if 'sample_num' in realized:
        return int(realized['sample_num'])
    if hash_index:
        return len(hash_index)
    raise ValueError('Cannot resolve sample_num from manifest or hash index')


def build_cache(cache_algorithm: str, num: int):
    if cache_algorithm == 'lru':
        return LRUCache(num)
    if cache_algorithm == 'quiver':
        return Quiver(num)
    if cache_algorithm == 'coordl':
        return CoorDL(num)
    if cache_algorithm == 'dgim_lru':
        return DGIMLRUCache(num)
    if cache_algorithm == 'dgim_quiver':
        return DGIMQuiver(num)
    if cache_algorithm == 'dgim_coordl':
        return DGIMCoorDL(num)
    if cache_algorithm == 'dgim_iacma':
        return DGIMIACMACache(num)
    if cache_algorithm == 'dgim_iacma_with_memory':
        return DGIMIACMACacheWithMemory(num)
    if cache_algorithm == 'dgim_iacma_admission':
        return DGIMIACMAAdmissionCache(num)
    if cache_algorithm == 'dgim_iacma_with_memory_admission':
        return DGIMIACMAAdmissionCacheWithMemory(num)
    raise ValueError(f'Unknown cache algorithm: {cache_algorithm}')


def resolve_paths(manifest: Dict) -> Tuple[str, str]:
    ffr_path = manifest.get('output_file') or manifest.get('data_file')
    hash_index_path = manifest.get('hash_index_file')
    if not ffr_path:
        raise ValueError('Manifest missing output_file/data_file for FFSlim')
    if not hash_index_path:
        raise ValueError('Manifest missing hash_index_file for hash_index mode')
    return ffr_path, hash_index_path


def process_samples(args):
    manifest = load_manifest(args.manifest)
    ffr_path, hash_index_path = resolve_paths(manifest)

    with open(hash_index_path, 'rb') as f:
        hash_index = pickle.load(f)['hash_index']

    sample_num = resolve_sample_num(manifest, hash_index)
    request_count = min(sample_num, args.request_count) if args.request_count > 0 else sample_num

    sample_ids = list(range(sample_num))
    random.seed(args.shuffle_seed)
    random.shuffle(sample_ids)
    sample_ids = sample_ids[:request_count]

    reader = FileReader(ffr_path)

    cache = None
    hits = 0
    cache_get_calls = 0
    cache_put_calls = 0
    cache_get_overhead = 0.0
    cache_put_overhead = 0.0

    if args.num > 0:
        cache = build_cache(args.cache_algorithm, args.num)

    index_time = 0.0
    process_begin = time.perf_counter()

    for req_id, sid in enumerate(sample_ids):
        if cache is not None:
            cache_get_start = time.perf_counter()
            result = cache.get(sid, req_id)
            cache_get_overhead += time.perf_counter() - cache_get_start
            cache_get_calls += 1
            if result != -1:
                hits += 1
                continue

        if args.index_structure == 'hash_index':
            idx_start = time.perf_counter()
            media_id, caption_id = hash_index[sid]
            index_time += time.perf_counter() - idx_start
        elif args.index_structure == 'adaptive_index':
            adaptive_x = args.adaptive_x
            if adaptive_x <= 0:
                adaptive_x = int(manifest.get('adaptive_x', 0))
            if adaptive_x <= 0:
                raise ValueError('adaptive_index requires --adaptive-x or manifest adaptive_x > 0')
            idx_start = time.perf_counter()
            media_id, caption_id = sid // adaptive_x, sid % adaptive_x
            index_time += time.perf_counter() - idx_start
        else:
            raise ValueError(f'Unknown index_structure: {args.index_structure}')

        datasets = reader.read([media_id])
        sample_obj = pickle.loads(datasets[0])
        _ = [sample_obj['image'], sample_obj['captions'][caption_id]]

        if cache is not None:
            cache_put_start = time.perf_counter()
            cache.put(sid - caption_id, sample_obj, req_id)
            cache_put_overhead += time.perf_counter() - cache_put_start
            cache_put_calls += 1

    process_total = time.perf_counter() - process_begin

    os.makedirs(SIM_OBJECT_LOG_DIR, exist_ok=True)
    with open(SIM_LOG_PATH, 'a+', encoding='utf-8') as f:
        f.write(
            f"[SIM][FFSlim] dataset={manifest.get('dataset_name','unknown')},requests={request_count},"
            f"cache={args.cache_algorithm},cache_size={args.num},index={args.index_structure},"
            f"manifest={args.manifest}\n"
        )
        f.write(f"reader.n={reader.n},sample_num={sample_num},index_time={index_time:.6f}s\n")
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
        'dgim_iacma',
        'dgim_iacma_with_memory',
        'dgim_iacma_admission',
        'dgim_iacma_with_memory_admission',
    ]:
        log_name = f"simulated_{manifest.get('dataset_name', 'unknown')}_{args.num}_{args.cache_algorithm}_object_importance.log"
        obj_log_path = os.path.join(SIM_OBJECT_LOG_DIR, log_name)
        with open(obj_log_path, 'w', encoding='utf-8') as f:
            for key, score_shifts in cache.get_object_importance().items():
                f.write(f"key:{key},score_shifts:{score_shifts}\n")


def build_parser():
    parser = argparse.ArgumentParser(description='Random read benchmark for simulated FFSlim datasets')
    parser.add_argument('--manifest', type=str, required=True, help='Path to simulated dataset manifest JSON')
    parser.add_argument('--num', type=int, default=0, help='Cache size')
    parser.add_argument('--cache-algorithm', type=str, default='lru', help='Cache algorithm')
    parser.add_argument('--index-structure', type=str, default='hash_index', choices=['hash_index', 'adaptive_index'])
    parser.add_argument('--adaptive-x', type=int, default=0, help='Adaptive index X override')
    parser.add_argument('--request-count', type=int, default=0, help='Request count (0 means all samples)')
    parser.add_argument('--shuffle-seed', type=int, default=123, help='Shuffle seed')
    parser.add_argument('--memory', type=int, default=0, help='Reserved for cgroup compatibility')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    process_samples(args)
