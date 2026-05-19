import argparse
import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache.ffrecord_lru import LRUCache

VARIANT = "F"


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


def _load_index(index_path):
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _read_member(f, member):
    f.seek(member["data_offset"])
    return f.read(member["length"])


def _read_sample(base_dir, entry):
    shard_path = os.path.join(base_dir, entry["shard"])
    result = {}
    with open(shard_path, "rb") as f:
        for member in entry["members"]:
            raw = _read_member(f, member)
            name = member["name"]
            if name.endswith(".jpg"):
                result["image"] = raw
            elif name.endswith(".json"):
                try:
                    result["meta"] = json.loads(raw.decode("utf-8"))
                except Exception:
                    result["meta_raw"] = raw
            else:
                result[name] = raw
        _drop_file_cache(f.fileno())
    return result


def process_samples(sample_ids, base_dir, entries, num):
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
        entry = entries[sid]
        result = _read_sample(base_dir, entry)
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

    base_dir = os.path.join(args.root, f"{args.dataset}_webdataset_{VARIANT}")
    index_path = os.path.join(base_dir, "index.jsonl")
    entries = _load_index(index_path)

    sample_ids = list(range(len(entries)))
    random.seed(123)
    random.shuffle(sample_ids)

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"dataset:{args.dataset},format:webdataset_{VARIANT},samples:{len(entries)},num:{args.num},memory_size:{args.memory}\n"
        )

    start = time.time()
    process_samples(sample_ids, base_dir, entries, args.num)
    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"WebDataset random read耗时 {VARIANT} at {args.num},memory_size:{args.memory}:{time.time() - start:.4f}秒\n"
        )


if __name__ == "__main__":
    main()
