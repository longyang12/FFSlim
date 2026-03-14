import pandas as pd
import ast
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

# 写入图片
# for i in range(21):
#     file_path = f"/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/train_balanced-{i:05d}-of-00021.parquet"  # 替换为你的文件路径
#     df = pd.read_parquet(file_path, engine='pyarrow')
#     print(file_path)
#     for idx, row in df.iterrows():
#         image = row["image"]
#         file_path = os.path.join("/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/gqa-images",image["path"])
#         with open(file_path, 'wb') as f:
#             f.write(image["bytes"])

image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/gqa-images"
data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],"image_qa":[],"image_qa_size":[]}
images = []
qas = []
for i in range(21):
    file_path = f"/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/balanced_parquet/train_balanced-{i:05d}-of-00021.parquet"  # 替换为你的文件路径
    df = pd.read_parquet(file_path, engine='pyarrow')
    print(file_path)
    for idx, row in df.iterrows():
        images.append(row["image"]["path"])
        qas.append(row["qa"])

print(len(images))
for i in range(72140):
    if i % 100 == 0:
        print(i/72140)
    for qa in qas[i]:
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
        image_qa = str(qa)
        data["image_qa"].append(image_qa)
        image_qa_size = len(image_qa)
        data["image_qa_size"].append(image_qa_size)

df = pd.DataFrame(data)
df.to_csv("gqa.csv",index=False)



