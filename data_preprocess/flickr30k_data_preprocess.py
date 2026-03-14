import pandas as pd
import ast
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

# 读取 JSON 文件
file_path = "datasets/flickr30k/flickr_annotations_30k.csv"  # 替换为你的文件路径
df = pd.read_csv(file_path)
print(df.tail())
print(df.describe())
print(df.columns)
captions = ast.literal_eval(df["raw"][0])
print(type(captions))
print(df["filename"][:5])
data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],"image_caption":[],"image_caption_size":[]}
images = [row["filename"] for idx, row in df.iterrows()]
captions = []
for idx, row in df.iterrows():
    captions.append(ast.literal_eval(row["raw"]))

image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k/flickr30k-images"

for i in range(31014):
    if i % 100 == 0:
        print(i/31014)
    for caption in captions[i]:
        image_path = images[i]
        image_full_path = os.path.join(image_dir, image_path)
        data["image_path"].append(image_path)
        with Image.open(image_full_path) as img:
            image_shape = (len(img.getbands()), img.height, img.width)
            tensor = transform(img)
            image_tensor_size = tensor.numel() * tensor.element_size()
        data["image_shape"].append(image_shape)
        data["image_tensor_size"].append(image_tensor_size)
        image_file_size = os.path.getsize(image_full_path)
        data["image_file_size"].append(image_file_size)
        data["compression_rate"].append(f"{image_tensor_size/image_file_size:0.2f}")
        image_caption = caption
        data["image_caption"].append(image_caption)
        image_caption_size = len(image_caption)
        data["image_caption_size"].append(image_caption_size)

df = pd.DataFrame(data)
df.to_csv("flickr30k_sample.csv",index=False)



