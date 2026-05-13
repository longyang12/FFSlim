import argparse
import os
import pickle
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ffrecord import FileReader
from cache.ffrecord_lru import LRUCache

VARIANT = "B"


def _get_anon_kb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def process_samples(sample_ids, reader, num):
    cache = LRUCache(num) if num != 0 else None
    hits = 0
    count = 0

    for sid in sample_ids:
        if cache is not None:
            count += 1
            cached = cache.get(sid)
            if cached != -1:
                hits += 1
                continue
        datasets = reader.read([sid])
        for sample in datasets:
            result = pickle.loads(sample)
            if cache is not None:
                cache.put(sid, result)

    if cache is not None:
        with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
            f.write(f"cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("num", type=int)
    parser.add_argument("memory", type=int)
    parser.add_argument("--root", default="/mnt/datasets/mscoco_converted")
    args = parser.parse_args()

    base_name = f"{args.dataset}_ffrecord_{VARIANT}"
    base_dir = os.path.join(args.root, base_name)
    ffr_path = os.path.join(base_dir, base_name + ".ffr")

    anon_before = _get_anon_kb()
    reader = FileReader(ffr_path)
    anon_after = _get_anon_kb()
    
    sample_ids = list(range(reader.n))
    random.seed(123)
    random.shuffle(sample_ids)

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"dataset:{args.dataset},format:ffrecord_{VARIANT},samples:{reader.n},num:{args.num},memory_size:{args.memory}\n"
        )
        f.write(
            "index_anon_kb:before={0},after={1},delta={2}\n".format(
                anon_before, anon_after, anon_after - anon_before
            )
        )

    start = time.time()
    process_samples(sample_ids, reader, args.num)
    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"FFRecord random read耗时 {VARIANT} at {args.num},memory_size:{args.memory}:{time.time() - start:.4f}秒\n"
        )


if __name__ == "__main__":
    main()
