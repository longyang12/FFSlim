import pandas as pd
import ast
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

# 数据探索
# file_path = f"/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/train-00000-of-00136.parquet"  # 替换为你的文件路径
# df = pd.read_parquet(file_path, engine='pyarrow')
# print(df.tail())
# print(df.columns)
# for i in range(10):
#     image = df["image"][i]
#     print(image.keys())
#     print(f"path:{image['path']},image_id:{df["image_id"][i]},question_id:{df["question_id"][i]},question_type:{df["question_type"][i]}")
#     print(f"multiple_choice_answer:{df["multiple_choice_answer"][i]},question:{df["question"][i]},answer_type:{df["answer_type"][i]}")
#     print(f"answers:{df["answers"][i]}")



# 写入图片
# for i in range(136):
#     file_path = f"/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/train-{i:05d}-of-00136.parquet"  # 替换为你的文件路径
#     df = pd.read_parquet(file_path, engine='pyarrow')
#     print(file_path)
#     for idx, row in df.iterrows():
#         image = row["image"]
#         file_path = os.path.join("/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/vqa-images",image["path"])
#         with open(file_path, 'wb') as f:
#             f.write(image["bytes"])

image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/vqa-images"
data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],"image_qa":[],"image_qa_size":[]}
images = []
qas = []
for i in range(136):
    file_path = f"/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/parquets/train-{i:05d}-of-00136.parquet"  # 替换为你的文件路径
    df = pd.read_parquet(file_path, engine='pyarrow')
    print(file_path)
    for idx, row in df.iterrows():
        images.append(row["image"]["path"])
        qa = {}
        qa["question_id"] = row["question_id"]
        qa["question_type"] = row["question_type"]
        qa["question"] = row["question"]
        qa["answer_type"] = row["answer_type"]
        qa["multiple_choice_answer"] = row["multiple_choice_answer"]
        qa["answers"] = list(row["answers"])
        qas.append(qa)

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
    image_qa = str(qas[i])
    data["image_qa"].append(image_qa)
    image_qa_size = len(image_qa)
    data["image_qa_size"].append(image_qa_size)

df = pd.DataFrame(data)
df.to_csv("vqa2.csv",index=False)



