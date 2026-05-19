import argparse
import hashlib
import os
import pickle
import time

from ffrecord import FileReader, FileWriter


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


def _normalize_captions(captions):
    if captions is None:
        return []
    if isinstance(captions, (bytes, str)):
        return [captions]
    return list(captions)


def b2f(input_ffr, output_ffr):
    reader = FileReader(input_ffr)
    total_records = 0
    input_records = reader.n
    input_captions = 0

    for i in range(reader.n):
        samples = reader.read([i])
        for sample_bytes in samples:
            try:
                obj = pickle.loads(sample_bytes)
            except Exception:
                continue
            captions = _normalize_captions(obj.get("captions") or obj.get(b"captions"))
            input_captions += len(captions)
            total_records += len(captions)

    os.makedirs(os.path.dirname(output_ffr) or ".", exist_ok=True)
    output_records = 0
    output_captions = 0
    image_count = 0

    with FileWriter(output_ffr, total_records) as writer:
        for i in range(reader.n):
            samples = reader.read([i])
            for sample_bytes in samples:
                try:
                    obj = pickle.loads(sample_bytes)
                except Exception:
                    continue
                image = obj.get("image") or obj.get(b"image")
                captions = _normalize_captions(obj.get("captions") or obj.get(b"captions"))
                image_count += 1
                for cap in captions:
                    per_obj = {"image": image, "captions": [cap]}
                    writer.write_one(pickle.dumps(per_obj))
                    output_records += 1
                    output_captions += 1

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": output_records,
        "output_captions": output_captions,
        "image_count": image_count,
    }


def f2b(input_ffr, output_ffr):
    reader = FileReader(input_ffr)
    grouped = {}
    input_records = reader.n
    input_captions = 0

    for i in range(reader.n):
        samples = reader.read([i])
        for sample_bytes in samples:
            try:
                obj = pickle.loads(sample_bytes)
            except Exception:
                continue
            image = obj.get("image") or obj.get(b"image")
            captions = _normalize_captions(obj.get("captions") or obj.get(b"captions"))
            input_captions += len(captions)
            if image is None:
                continue
            key = image_key(image)
            if key not in grouped:
                grouped[key] = {"image": image, "captions": []}
            grouped[key]["captions"].extend(captions)

    os.makedirs(os.path.dirname(output_ffr) or ".", exist_ok=True)
    output_records = 0
    output_captions = 0

    with FileWriter(output_ffr, len(grouped)) as writer:
        for key in grouped:
            payload = {"image": grouped[key]["image"], "captions": grouped[key]["captions"]}
            writer.write_one(pickle.dumps(payload))
            output_records += 1
            output_captions += len(grouped[key]["captions"])

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
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="unknown")
    p.add_argument("--format", dest="data_format", default="ffrecord")
    p.add_argument("--log-file")
    args = p.parse_args()

    start = time.perf_counter()
    if args.mode == "B2F":
        stats = b2f(args.input, args.output)
    else:
        stats = f2b(args.input, args.output)
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
