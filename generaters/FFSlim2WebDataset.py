import os
import io
import json
import tarfile
from ffrecord import FileReader


def ffslim_to_webdataset(
    input_ffr,
    output_dir,
    shard_size=1000,
    per_caption=False,
    index_path=None,
):
    os.makedirs(output_dir, exist_ok=True)
    reader = FileReader(input_ffr)
    shard_idx = 0
    item_in_shard = 0
    tar = None
    shard_name = None
    shard_entries = []
    if index_path is None:
        index_path = os.path.join(output_dir, "index.jsonl")
    index_fp = open(index_path, "w", encoding="utf-8")

    def open_shard(idx):
        nonlocal shard_name
        shard_name = f"shard-{idx:05d}.tar"
        path = os.path.join(output_dir, shard_name)
        return tarfile.open(path, "w")

    def flush_shard_index():
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
            for member_name in entry["members"]:
                info = member_map.get(member_name)
                if info is None:
                    continue
                members.append({"name": member_name, **info})
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
        flush_shard_index()
        shard_idx += 1
        item_in_shard = 0

    try:
        for i in range(reader.n):
            samples = reader.read([i])
            for sample_bytes in samples:
                obj = None
                try:
                    import pickle

                    obj = pickle.loads(sample_bytes)
                except Exception:
                    # if already stored as dict-like via ffrecord, try to treat as bytes
                    continue

                # create shard when needed
                if tar is None:
                    tar = open_shard(shard_idx)
                    item_in_shard = 0

                key_base = f"{i:08d}"
                img_bytes = obj.get("image") or obj.get(b"image")
                captions = obj.get("captions") or obj.get(b"captions") or []

                def write_sample(sample_key, caption_payload):
                    members = []
                    if img_bytes:
                        img_name = sample_key + ".jpg"
                        ti = tarfile.TarInfo(img_name)
                        ti.size = len(img_bytes)
                        tar.addfile(ti, io.BytesIO(img_bytes))
                        members.append(img_name)

                    meta = {"txt": caption_payload}
                    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
                    ti = tarfile.TarInfo(sample_key + ".json")
                    ti.size = len(meta_bytes)
                    tar.addfile(ti, io.BytesIO(meta_bytes))
                    members.append(sample_key + ".json")
                    return members

                if per_caption:
                    for cap_idx, cap in enumerate(captions):
                        sample_key = f"{key_base}_{cap_idx:03d}"
                        members = write_sample(sample_key, cap)
                        shard_entries.append(
                            {"sample_id": sample_key, "members": members}
                        )
                        item_in_shard += 1
                        if item_in_shard >= shard_size:
                            close_shard()
                            tar = open_shard(shard_idx)
                else:
                    members = write_sample(key_base, captions)
                    shard_entries.append(
                        {"sample_id": key_base, "members": members}
                    )
                    item_in_shard += 1
                if item_in_shard >= shard_size:
                    close_shard()

    finally:
        if tar is not None:
            close_shard()
        index_fp.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--per-caption", action="store_true")
    p.add_argument("--index-path")
    args = p.parse_args()
    ffslim_to_webdataset(
        args.input,
        args.output,
        shard_size=args.shard_size,
        per_caption=args.per_caption,
        index_path=args.index_path,
    )
