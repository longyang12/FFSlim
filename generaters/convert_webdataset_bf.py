import argparse
import hashlib
import io
import json
import os
import tarfile
import time


def image_key(image_bytes):
    return hashlib.sha1(image_bytes).hexdigest()


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


def iter_samples_from_tar(tar_path):
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        by_key = {}
        for member in members:
            name = member.name
            base, ext = os.path.splitext(name)
            if ext not in [".jpg", ".json"]:
                continue
            fp = tar.extractfile(member)
            if fp is None:
                continue
            data = fp.read()
            entry = by_key.setdefault(base, {})
            entry[ext] = data
        for key, entry in by_key.items():
            yield key, entry.get(".jpg"), entry.get(".json")


def write_sharded_webdataset(output_dir, samples, shard_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    item_in_shard = 0
    tar = None
    shard_name = None
    shard_entries = []
    index_path = os.path.join(output_dir, "index.jsonl")
    index_fp = open(index_path, "w", encoding="utf-8")

    def open_shard(idx):
        nonlocal shard_name
        shard_name = f"shard-{idx:05d}.tar"
        path = os.path.join(output_dir, shard_name)
        return tarfile.open(path, "w")

    def flush_index():
        if not shard_entries:
            return
        shard_path = os.path.join(output_dir, shard_name)
        member_map = {}
        with tarfile.open(shard_path, "r") as tar_r:
            for member in tar_r.getmembers():
                member_map[member.name] = {
                    "header_offset": member.offset,
                    "data_offset": member.offset_data,
                    "length": member.size,
                }
        for entry in shard_entries:
            members = []
            for name in entry["members"]:
                info = member_map.get(name)
                if info:
                    members.append({"name": name, **info})
            record = {
                "sample_id": entry["sample_id"],
                "shard": shard_name,
                "members": members,
            }
            index_fp.write(json.dumps(record, ensure_ascii=True) + "\n")
        shard_entries.clear()

    def close_shard():
        nonlocal tar, shard_idx, item_in_shard
        if tar is None:
            return
        tar.close()
        tar = None
        flush_index()
        shard_idx += 1
        item_in_shard = 0

    def write_sample(sample_key, image_bytes, caption_payload):
        nonlocal tar, item_in_shard
        members = []
        if image_bytes:
            img_name = sample_key + ".jpg"
            ti = tarfile.TarInfo(img_name)
            ti.size = len(image_bytes)
            tar.addfile(ti, io.BytesIO(image_bytes))
            members.append(img_name)
        meta = {"txt": caption_payload}
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        meta_name = sample_key + ".json"
        ti = tarfile.TarInfo(meta_name)
        ti.size = len(meta_bytes)
        tar.addfile(ti, io.BytesIO(meta_bytes))
        members.append(meta_name)
        shard_entries.append({"sample_id": sample_key, "members": members})
        item_in_shard += 1
        if item_in_shard >= shard_size:
            close_shard()
            tar = open_shard(shard_idx)

    tar = open_shard(shard_idx)
    for sample_key, image_bytes, caption_payload in samples:
        write_sample(sample_key, image_bytes, caption_payload)

    if tar is not None:
        close_shard()
    index_fp.close()


def b2f(input_dir, output_dir, shard_size=1000):
    input_records = 0
    input_captions = 0
    output_records = 0
    output_captions = 0

    def sample_iter():
        nonlocal input_records, input_captions, output_records, output_captions
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith(".tar"):
                continue
            tar_path = os.path.join(input_dir, name)
            for key, img_bytes, meta_bytes in iter_samples_from_tar(tar_path):
                if meta_bytes is None:
                    continue
                input_records += 1
                meta = json.loads(meta_bytes.decode("utf-8", errors="ignore"))
                txt = meta.get("txt", [])
                if isinstance(txt, list):
                    input_captions += len(txt)
                    for idx, cap in enumerate(txt):
                        sample_key = f"{key}_{idx:03d}"
                        output_records += 1
                        output_captions += 1
                        yield sample_key, img_bytes, cap
                else:
                    input_captions += 1
                    sample_key = f"{key}_000"
                    output_records += 1
                    output_captions += 1
                    yield sample_key, img_bytes, txt

    write_sharded_webdataset(output_dir, sample_iter(), shard_size=shard_size)

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": output_records,
        "output_captions": output_captions,
        "image_count": output_records,
    }


def f2b(input_dir, output_dir, shard_size=1000):
    grouped = {}
    input_records = 0
    input_captions = 0

    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".tar"):
            continue
        tar_path = os.path.join(input_dir, name)
        for key, img_bytes, meta_bytes in iter_samples_from_tar(tar_path):
            if img_bytes is None or meta_bytes is None:
                continue
            input_records += 1
            meta = json.loads(meta_bytes.decode("utf-8", errors="ignore"))
            txt = meta.get("txt", "")
            key_hash = image_key(img_bytes)
            if key_hash not in grouped:
                grouped[key_hash] = {"image": img_bytes, "captions": []}
            if isinstance(txt, list):
                grouped[key_hash]["captions"].extend(txt)
                input_captions += len(txt)
            else:
                grouped[key_hash]["captions"].append(txt)
                input_captions += 1

    output_records = 0
    output_captions = 0

    def sample_iter():
        nonlocal output_records, output_captions
        idx = 0
        for key in grouped:
            sample_key = f"{idx:08d}"
            payload = grouped[key]
            output_records += 1
            output_captions += len(payload["captions"])
            yield sample_key, payload["image"], payload["captions"]
            idx += 1

    write_sharded_webdataset(output_dir, sample_iter(), shard_size=shard_size)

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": output_records,
        "output_captions": output_captions,
        "image_count": output_records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["B2F", "F2B"])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--dataset", default="unknown")
    p.add_argument("--format", dest="data_format", default="webdataset")
    p.add_argument("--log-file")
    args = p.parse_args()

    start = time.perf_counter()
    if args.mode == "B2F":
        stats = b2f(args.input_dir, args.output_dir, args.shard_size)
    else:
        stats = f2b(args.input_dir, args.output_dir, args.shard_size)
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
