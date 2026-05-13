import json
import os
import pickle
import struct
from ffrecord import FileReader


def _bytes_feature(value):
    try:
        from tensorflow.train import BytesList, Feature
    except Exception:
        return None

    if isinstance(value, str):
        value = value.encode("utf-8")
    return Feature(bytes_list=BytesList(value=[value]))


def _build_tfrecord_index(tfrecord_path, index_path):
    with open(tfrecord_path, "rb") as f, open(index_path, "w", encoding="utf-8") as idx:
        record_id = 0
        while True:
            record_offset = f.tell()
            length_bytes = f.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise RuntimeError("TFRecord index: truncated length header")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(f.read(4)) != 4:
                raise RuntimeError("TFRecord index: truncated length CRC")
            f.seek(length, os.SEEK_CUR)
            if len(f.read(4)) != 4:
                raise RuntimeError("TFRecord index: truncated data CRC")
            record = {
                "sample_id": record_id,
                "path": os.path.basename(tfrecord_path),
                "offset": record_offset,
                "length": int(length),
            }
            idx.write(json.dumps(record, ensure_ascii=True) + "\n")
            record_id += 1


def ffslim_to_tfrecord(
    input_ffr,
    output_tfrecord,
    per_caption=False,
    index_path=None,
):
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("tensorflow required for TFRecord conversion: pip install tensorflow") from e

    reader = FileReader(input_ffr)
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for i in range(reader.n):
            samples = reader.read([i])
            for sample_bytes in samples:
                try:
                    obj = pickle.loads(sample_bytes)
                except Exception:
                    continue

                image = obj.get("image") or obj.get(b"image")
                captions = obj.get("captions") or obj.get(b"captions") or []

                if per_caption:
                    for cap in captions:
                        feature = {
                            'image/encoded': _bytes_feature(image),
                            'image/caption': _bytes_feature(cap),
                            'image/id': _bytes_feature(str(i))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                else:
                    # join captions with special token
                    cap_joined = "\n".join(captions)
                    feature = {
                        'image/encoded': _bytes_feature(image),
                        'image/captions': _bytes_feature(cap_joined),
                        'image/id': _bytes_feature(str(i))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

    if index_path is None:
        index_path = output_tfrecord + ".index.jsonl"
    _build_tfrecord_index(output_tfrecord, index_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-caption", action="store_true")
    p.add_argument("--index-path")
    args = p.parse_args()
    ffslim_to_tfrecord(
        args.input,
        args.output,
        per_caption=args.per_caption,
        index_path=args.index_path,
    )
