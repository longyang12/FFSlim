import os
import pickle
from ffrecord import FileReader, FileWriter


def _count_records(reader, per_caption):
    if not per_caption:
        return reader.n

    total = 0
    for i in range(reader.n):
        samples = reader.read([i])
        for sample_bytes in samples:
            try:
                obj = pickle.loads(sample_bytes)
            except Exception:
                continue
            captions = obj.get("captions") or obj.get(b"captions") or []
            if isinstance(captions, (bytes, str)):
                captions = [captions]
            total += len(captions)
    return total


def ffslim_to_ffrecord(input_ffr, output_ffr, per_caption=False):
    os.makedirs(os.path.dirname(output_ffr) or ".", exist_ok=True)
    reader = FileReader(input_ffr)
    total_records = _count_records(reader, per_caption)
    with FileWriter(output_ffr, total_records) as writer:
        for i in range(reader.n):
            samples = reader.read([i])
            for sample_bytes in samples:
                if not per_caption:
                    writer.write_one(sample_bytes)
                    continue

                try:
                    obj = pickle.loads(sample_bytes)
                except Exception:
                    continue

                image = obj.get("image") or obj.get(b"image")
                captions = obj.get("captions") or obj.get(b"captions") or []
                if isinstance(captions, (bytes, str)):
                    captions = [captions]
                for cap in captions:
                    per_obj = {"image": image, "captions": [cap]}
                    writer.write_one(pickle.dumps(per_obj))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-caption", action="store_true")
    args = p.parse_args()
    ffslim_to_ffrecord(args.input, args.output, per_caption=args.per_caption)
