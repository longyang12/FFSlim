import argparse
import json
import os
import random
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache.tdp_lru import LRUCache

VARIANT = "B"


_POSIX_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
_HAS_FADVISE = hasattr(os, "posix_fadvise")


def _drop_file_cache(fd, offset=0, length=0):
    if not _HAS_FADVISE:
        return
    try:
        os.posix_fadvise(fd, offset, length, _POSIX_FADV_DONTNEED)
    except Exception:
        pass


def _get_anon_kb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def process_samples(sample_ids, base_dir, df, num):
    cache = LRUCache(num) if num != 0 else None
    hits = 0
    count = 0

    anon_before = _get_anon_kb()
    images = df["image_path"].tolist()
    captions = df["image_caption"].tolist()
    anon_after = _get_anon_kb()
    for sid in sample_ids:
        key = os.path.splitext(images[sid])[0]
        if cache is not None:
            count += 1
            cached = cache.get(key, sid)
            if cached != -1:
                hits += 1
                continue
        img_path = os.path.join(base_dir, "img", images[sid])
        with open(img_path, "rb") as f:
            img_data = f.read()
            _drop_file_cache(f.fileno())
        try:
            caption = json.loads(captions[sid])
        except Exception:
            caption = captions[sid]
        result = {"image": img_data, "caption": caption}
        if cache is not None:
            cache.put(key, result, sid)

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

    base_name = f"{args.dataset}_tdp_{VARIANT}"
    base_dir = os.path.join(args.root, base_name)
    csv_path = os.path.join(base_dir, base_name + "_TDP.csv")

    df = pd.read_csv(csv_path)
    sample_ids = list(range(len(df)))
    random.seed(123)
    random.shuffle(sample_ids)

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"dataset:{args.dataset},format:tdp_{VARIANT},samples:{len(df)},num:{args.num},memory_size:{args.memory}\n"
        )

    start = time.time()
    process_samples(sample_ids, base_dir, df, args.num)
    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"TDP random read耗时 {VARIANT} at {args.num},memory_size:{args.memory}:{time.time() - start:.4f}秒\n"
        )


if __name__ == "__main__":
    main()
