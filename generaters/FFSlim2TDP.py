from ffrecord import FileReader
import pickle
import os
import pandas as pd 


for dataset in ["cc3m","flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E"]:
    file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"
    output_dir = f"/mnt/datasets/{dataset}_TDP"

    # 如果output_dir已经存在则将其目录名修改为output_dir+"_backup"
    # if os.path.exists(output_dir):
    #     os.rename(output_dir, output_dir+"_backup")

    os.makedirs(output_dir, exist_ok=True)

    images = []
    captions = []

    reader = FileReader(file_path)
    for i in range(reader.n):
        datasets = reader.read([i])
        for sample in datasets:
            samples1 = pickle.loads(sample)
            # 将图片保存在output_dir下的img文件夹中
            img_dir = os.path.join(output_dir, "img")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{i}.jpg")
            with open(img_path, "wb") as f:
                f.write(samples1["image"])
            
            # 准备图像文件名和文本数据，图像文件名为"{i}.jpg"，文本数据为samples1["captions"]中的每个caption，保存到images和captions列表中
            for caption in samples1["captions"]:
                images.append(f"{i}.jpg")
                captions.append(caption)
            
    # 将图像文件名和文本数据保存到output_dir下的{dataset}_TDP.csv中
    csv_path = os.path.join(output_dir, f"{dataset}_TDP.csv")
    df = pd.DataFrame({"image_path": images, "image_caption": captions})
    df.to_csv(csv_path, index=False)

    print(f"{dataset} done!")