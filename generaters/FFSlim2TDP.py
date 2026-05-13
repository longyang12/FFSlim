from ffrecord import FileReader
import pickle
import os
import json
import pandas as pd


def ffslim_to_tdp(input_ffr, output_dir, dataset_name, per_caption=False):
    os.makedirs(output_dir, exist_ok=True)

    images = []
    captions = []

    reader = FileReader(input_ffr)
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    for i in range(reader.n):
        datasets = reader.read([i])
        for sample in datasets:
            sample_obj = pickle.loads(sample)
            img_path = os.path.join(img_dir, f"{i}.jpg")
            with open(img_path, "wb") as f:
                f.write(sample_obj["image"])

            sample_captions = sample_obj.get("captions", [])
            if isinstance(sample_captions, (bytes, str)):
                if isinstance(sample_captions, bytes):
                    sample_captions = sample_captions.decode("utf-8", errors="ignore")
                sample_captions = [sample_captions]
            if per_caption:
                for caption in sample_captions:
                    if isinstance(caption, bytes):
                        caption = caption.decode("utf-8", errors="ignore")
                    images.append(f"{i}.jpg")
                    captions.append(caption)
            else:
                safe_captions = []
                for caption in sample_captions:
                    if isinstance(caption, bytes):
                        caption = caption.decode("utf-8", errors="ignore")
                    safe_captions.append(caption)
                images.append(f"{i}.jpg")
                captions.append(json.dumps(safe_captions, ensure_ascii=True))

    csv_path = os.path.join(output_dir, f"{dataset_name}_TDP.csv")
    df = pd.DataFrame({"image_path": images, "image_caption": captions})
    df.to_csv(csv_path, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["cc3m", "flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E"])
    parser.add_argument("--per-caption", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"
        output_dir = f"/mnt/datasets/{dataset}_TDP"
        ffslim_to_tdp(file_path, output_dir, dataset, per_caption=args.per_caption)
        print(f"{dataset} done!")


if __name__ == "__main__":
    main()