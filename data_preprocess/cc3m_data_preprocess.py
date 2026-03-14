import pandas as pd
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

# 数据探索
txt_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/txt"
all_files = os.listdir(txt_dir)
print(len(all_files))
print(all_files[:2])

image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/cc3m-images"
data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],"image_caption":[],"image_caption_size":[]}
images = []
captions = []
for file in all_files:
    images.append(file.split(".")[0] + ".jpg")
    with open(os.path.join(txt_dir,file),"r") as f:
        captions.append(f.read())
print(images[:2])
print(captions[:2])


n = len(images)
for i in range(n):
    if i % 100 == 0:
        print(i/n)
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
    image_caption = captions[i]
    data["image_caption"].append(image_caption)
    image_caption_size = len(image_caption)
    data["image_caption_size"].append(image_caption_size)

df = pd.DataFrame(data)
df.to_csv("cc3m.csv",index=False)



