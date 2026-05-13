import argparse
import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def _load_index(index_path):
    anon_before = _get_anon_kb()
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    anon_after = _get_anon_kb()
    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            "index_anon_kb:before={0},after={1},delta={2}\n".format(
                anon_before, anon_after, anon_after - anon_before
            )
        )
    return entries


def _read_record(f, offset):
    f.seek(offset)
    length_bytes = f.read(8)
    if len(length_bytes) != 8:
        return None
    length = int.from_bytes(length_bytes, "little")
    f.read(4)
    data = f.read(length)
    f.read(4)
    return data


def _decode_example(data):
    try:
        import tensorflow as tf
    except Exception:
        return {"raw": data}

    example = tf.train.Example.FromString(data)
    feat = example.features.feature
    image = None
    caption = None
    if "image/encoded" in feat and feat["image/encoded"].bytes_list.value:
        image = feat["image/encoded"].bytes_list.value[0]
    if "image/caption" in feat and feat["image/caption"].bytes_list.value:
        caption = feat["image/caption"].bytes_list.value[0].decode("utf-8", "ignore")
    if "image/captions" in feat and feat["image/captions"].bytes_list.value:
        caption = feat["image/captions"].bytes_list.value[0].decode("utf-8", "ignore")
    return {"image": image, "caption": caption}


def process_samples(sample_ids, tfrecord_path, entries, num):
    cache = LRUCache(num) if num != 0 else None
    hits = 0
    count = 0

    with open(tfrecord_path, "rb") as f:
        for sid in sample_ids:
            if cache is not None:
                count += 1
                cached = cache.get(sid)
                if cached != -1:
                    hits += 1
                    continue
            entry = entries[sid]
            data = _read_record(f, entry["offset"])
            if data is None:
                continue
            result = _decode_example(data)
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

    base_name = f"{args.dataset}_tfrecord_{VARIANT}"
    base_dir = os.path.join(args.root, base_name)
    tfrecord_path = os.path.join(base_dir, base_name + ".tfrecord")
    index_path = tfrecord_path + ".index.jsonl"

    entries = _load_index(index_path)

    sample_ids = list(range(len(entries)))
    random.seed(123)
    random.shuffle(sample_ids)

    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"dataset:{args.dataset},format:tfrecord_{VARIANT},samples:{len(entries)},num:{args.num},memory_size:{args.memory}\n"
        )

    start = time.time()
    process_samples(sample_ids, tfrecord_path, entries, args.num)
    with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log", "a+") as f:
        f.write(
            f"TFRecord random read耗时 {VARIANT} at {args.num},memory_size:{args.memory}:{time.time() - start:.4f}秒\n"
        )


if __name__ == "__main__":
    main()
