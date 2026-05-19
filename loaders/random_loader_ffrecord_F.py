import argparse
import os
import pickle
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ffrecord import FileReader
from cache.ffrecord_lru import LRUCache

VARIANT = "F"


_POSIX_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
_HAS_FADVISE = hasattr(os, "posix_fadvise")


def _drop_file_cache(fd, offset=0, length=0):
    if not _HAS_FADVISE or fd is None:
        return
    try:
        os.posix_fadvise(fd, offset, length, _POSIX_FADV_DONTNEED)
    except Exception:
        pass


def _open_side_fd(ffr_path):
    try:
        return os.open(ffr_path, os.O_RDONLY)
    except Exception:
        return None



def process_samples(sample_ids, reader, side_fd, num):
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
        _drop_file_cache(side_fd)
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

    reader = FileReader(ffr_path)
    side_fd = _open_side_fd(ffr_path)


    sample_ids = list(range(reader.n))
    random.seed(123)
    random.shuffle(sample_ids)

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"dataset:{args.dataset},format:ffrecord_{VARIANT},samples:{reader.n},num:{args.num},memory_size:{args.memory}\n"
        )
    start = time.time()
    try:
        process_samples(sample_ids, reader, side_fd, args.num)
    finally:
        if side_fd is not None:
            try:
                os.close(side_fd)
            except Exception:
                pass

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"FFRecord random read耗时 {VARIANT} at {args.num},memory_size:{args.memory}:{time.time() - start:.4f}秒\n"
        )


if __name__ == "__main__":
    main()
