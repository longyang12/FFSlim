import pandas as pd
import json
import os
from PIL import Image
from torchvision import transforms
import multiprocessing
from tqdm import tqdm

# 初始化全局变量
transform = transforms.Compose([transforms.ToTensor()])
image_dir = "/home/llm/experiment/dataset_Analysis_Modeling/dataset/train2017/"

def init_worker():
    """子进程初始化函数（解决多进程环境下的资源初始化问题）"""
    global transform
    transform = transforms.Compose([transforms.ToTensor()])

def process_annotation(args):
    """单个annotation处理函数（会被分发到不同进程）"""
    idx, annotation, image_path = args
    try:
        # 图像处理部分
        with Image.open(image_path) as img:
            image_shape = (len(img.getbands()), img.height, img.width)
            tensor = transform(img)
            image_tensor_size = tensor.numel() * tensor.element_size()
        
        # 文件元数据
        image_file_size = os.path.getsize(image_path)
        
        # 文本处理部分
        image_caption = annotation["caption"]
        image_caption_size = len(image_caption)
        
        return {
            "image_path": os.path.basename(image_path),
            "image_shape": image_shape,
            "image_tensor_size": image_tensor_size,
            "image_file_size": image_file_size,
            "image_caption": image_caption,
            "image_caption_size": image_caption_size
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    # 读取JSON数据
    with open("dataset/annotations/captions_train2017.json", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 准备任务参数
    tasks = []
    for idx, ann in enumerate(json_data["annotations"]):
        image_path = os.path.join(image_dir, f"{ann['image_id']:012d}.jpg")
        if os.path.exists(image_path):
            tasks.append((idx, ann, image_path))
    
    # 创建进程池（建议设置为CPU核心数的1.5-2倍[5,7](@ref)）
    num_workers = min(multiprocessing.cpu_count() * 2, 32)
    # num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker
    ) as pool:
        # 使用imap_unordered加速处理（允许乱序返回结果）
        results = []
        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for result in pool.imap_unordered(process_annotation, tasks, chunksize=100):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # 生成DataFrame
    df = pd.DataFrame(results)
    df.to_csv("mscoco.csv", index=False)

if __name__ == "__main__":
    main()