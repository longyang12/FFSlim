import pandas as pd
import json
import os
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

# 读取 JSON 文件
file_path = "dataset/annotations/captions_train2017.json"  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],"image_caption":[],"image_caption_size":[]}
images = json_data["images"]
annotations = json_data["annotations"]
image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/dataset/train2017/"

for i in range(1000):
    image_path = f"{annotations[i]["image_id"]:012d}.jpg"
    image_full_path = image_dir + image_path
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
    image_caption = annotations[i]["caption"]
    data["image_caption"].append(image_caption)
    image_caption_size = len(image_caption)
    data["image_caption_size"].append(image_caption_size)

df = pd.DataFrame(data)
df.to_csv("mscoco_sample.csv",index=False)



