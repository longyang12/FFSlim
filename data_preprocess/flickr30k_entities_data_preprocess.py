import pandas as pd
import ast
import os
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET


transform = transforms.Compose([
    transforms.ToTensor()  # 自动将 PIL.Image 转换为张量并归一化到 [0,1]
])

def parse_xml_to_dict(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    result = {}
    
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        
        # 处理带坐标框的情况
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            coords = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            result[name] = coords
        # 处理场景标记的情况
        else:
            scene = obj.find('scene').text.strip() if obj.find('scene') is not None else None
            result[name] = scene if scene is not None else "无场景信息"
    
    return result


def get_files_basic(dir_path):
    return [f for f in os.listdir(dir_path) 
            if os.path.isfile(os.path.join(dir_path, f))]

# 示例
image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/flickr30k-images"  # 替换为你的文件路径
caption_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/Sentences"
annotation_dir = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/Annotations"
images = get_files_basic(image_dir)
captions = get_files_basic(caption_dir)
annotations = get_files_basic(annotation_dir)
images.sort()
captions.sort()
annotations.sort()
print(images[:2])
print(captions[:2])
print(annotations[:2])
n = len(images)

captions1 = []
annotations1 = []

for i in range(n):
    if images[i].split(".")[0] != captions[i].split(".")[0] or images[i].split(".")[0] != annotations[i].split(".")[0]:
        print("what")
        print(images[i],captions[i],annotations[i])
    else:
        with open(os.path.join(caption_dir,captions[i])) as f:
            captions1.append([i.replace("\n","") for i in f.readlines()])
        filename = os.path.join(annotation_dir,annotations[i])
        annotations1.append(str(parse_xml_to_dict(filename)))

print(captions1[:2])
print(annotations1[:2])

data = {"image_path":[],"image_shape":[],"image_tensor_size":[],"image_file_size":[],"compression_rate":[],
        "image_caption":[],"image_caption_size":[],"image_annotation":[],"image_annotation_size":[],"text_size":[]}


for i in range(n):
    if i % 100 == 0:
        print(i/n)
    for caption in captions1[i]:
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
        image_annotation = annotations1[i]
        data["image_annotation"].append(image_annotation)
        image_annotation_size = len(image_annotation)
        data["image_annotation_size"].append(image_annotation_size)
        text_size = image_caption_size + image_annotation_size
        data["text_size"].append(text_size)

df = pd.DataFrame(data)
df.to_csv("flickr30k_entities.csv",index=False)



