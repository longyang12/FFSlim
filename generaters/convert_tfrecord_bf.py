import argparse
import hashlib
import os
import time

import tensorflow as tf
from tensorflow.train import Example
from tensorflow.train import BytesList, Feature


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


def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return Feature(bytes_list=BytesList(value=[value]))


def _get_feature_bytes(feature_map, name):
    if name not in feature_map:
        return None
    values = feature_map[name].bytes_list.value
    return values[0] if values else None


def _parse_example(raw_bytes):
    ex = Example()
    ex.ParseFromString(raw_bytes)
    feat = ex.features.feature
    return {
        "image": _get_feature_bytes(feat, "image/encoded"),
        "caption": _get_feature_bytes(feat, "image/caption"),
        "captions": _get_feature_bytes(feat, "image/captions"),
    }


def _write_f_record(writer, image_bytes, caption):
    feat = {
        "image/encoded": _bytes_feature(image_bytes),
        "image/caption": _bytes_feature(caption),
    }
    ex = Example(features=tf.train.Features(feature=feat))
    writer.write(ex.SerializeToString())


def _write_b_record(writer, image_bytes, captions_joined):
    feat = {
        "image/encoded": _bytes_feature(image_bytes),
        "image/captions": _bytes_feature(captions_joined),
    }
    ex = Example(features=tf.train.Features(feature=feat))
    writer.write(ex.SerializeToString())


def b2f(input_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds = tf.data.TFRecordDataset([input_path])
    input_records = 0
    input_captions = 0
    output_records = 0
    output_captions = 0

    with tf.io.TFRecordWriter(output_path) as writer:
        for raw in ds:
            input_records += 1
            obj = _parse_example(raw.numpy())
            image = obj["image"]
            captions = obj["captions"]
            if image is None or captions is None:
                continue
            caption_text = captions.decode("utf-8", errors="ignore")
            split_caps = [c for c in caption_text.split("\n") if c != ""]
            input_captions += len(split_caps)
            for cap in split_caps:
                _write_f_record(writer, image, cap)
                output_records += 1
                output_captions += 1

    return {
        "input_records": input_records,
        "input_captions": input_captions,
        "output_records": output_records,
        "output_captions": output_captions,
        "image_count": output_records,
    }


def f2b(input_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds = tf.data.TFRecordDataset([input_path])
    grouped = {}
    input_records = 0
    input_captions = 0

    for raw in ds:
        input_records += 1
        obj = _parse_example(raw.numpy())
        image = obj["image"]
        caption = obj["caption"]
        if image is None or caption is None:
            continue
        key = image_key(image)
        if key not in grouped:
            grouped[key] = {"image": image, "captions": []}
        grouped[key]["captions"].append(caption.decode("utf-8", errors="ignore"))
        input_captions += 1

    output_records = 0
    output_captions = 0

    with tf.io.TFRecordWriter(output_path) as writer:
        for key in grouped:
            joined = "\n".join(grouped[key]["captions"])
            _write_b_record(writer, grouped[key]["image"], joined)
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
    p.add_argument("--format", dest="data_format", default="tfrecord")
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
