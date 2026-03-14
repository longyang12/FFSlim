import pandas as pd
import os
import shutil
from tqdm import tqdm

def split_coco_dataset(csv_path, image_src_dir, output_root, N=1000):
    """
    拆分COCO数据集到指定目录结构
    
    参数：
    csv_path : str       - 输入的CSV文件路径
    image_src_dir : str  - 原始图片所在目录
    output_root : str    - 输出根目录
    N : int              - 每个子目录最大文件数
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建进度条
    pbar = tqdm(total=len(df), desc="Processing files")
    start = time.time()
    for idx, row in df.iterrows():
        # 计算目录编号和文件编号
        dir_num = idx // N
        file_num = idx % N
        
        # 生成目录路径（6位补零）
        dir_path = os.path.join(output_root, f"{dir_num:06d}")
        os.makedirs(dir_path, exist_ok=True)
        
        # 构造目标文件名（6位补零）
        base_name = f"{file_num:06d}"
        
        # 处理图片文件
        src_image = os.path.join(image_src_dir, row['image_path'])
        dest_image = os.path.join(dir_path, f"{base_name}.jpg")

        # 读写图片
        with open(src_image, 'rb') as src_file:
            binary_data = src_file.read()  # 读取二进制数据[3,10](@ref)
        with open(dest_image, 'wb') as dest_file:
            dest_file.write(binary_data)   # 写入二进制数据[9,10](@ref)

        # shutil.copy2(src_image, dest_image)
        
        # 生成文本文件
        dest_text = os.path.join(dir_path, f"{base_name}.txt")
        with open(dest_text, 'w', encoding='utf-8') as f:
            f.write(row['image_caption'])
        
        pbar.update(1)
    
    pbar.close()
    print(f"数据集拆分完成，共处理 {len(df)} 条记录")
    print(f"Files写入耗时:{(time.time() - start)/2:.4f}秒") 

# 使用示例
if __name__ == "__main__":
    # 写入性能
    import argparse
    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('dataset', type=str, help='输入数据集名称')
    # 解析参数
    args = parser.parse_args()
    print(f"dataset:{args.dataset}")

    # flickr30k
    if args.dataset == "flickr30k":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/flickr30k_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k/flickr30k-images",
            "output_root": "/mnt/datasets/flickr30k_Files",
            "N": 1000000  # 可修改为10/100/1000/10000/100000/1000000
        }

    # flickr30k_entities
    if args.dataset == "flickr30k_entities":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/flickr30k_entities_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/flickr30k-images",
            "output_root": "/mnt/datasets/flickr30k_entities_Files",
            "N": 1000000  # 可修改为10/100/1000/10000/100000/1000000
        }

    # mscoco
    if args.dataset == "mscoco":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/mscoco_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/mscoco/train2017",
            "output_root": "/mnt/datasets/mscoco_Files",
            "N": 1000000  # 可修改为10/100/1000/10000/100000/1000000
        }

    # vqa2
    if args.dataset == "vqa2":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/vqa2_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/vqa-images",
            "output_root": "/mnt/datasets/vqa2_Files",
            "N": 1000000  # 可修改为10/100/1000/10000/100000/1000000
        }

    # gqa
    if args.dataset == "gqa":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/gqa_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/gqa-images",
            "output_root": "/mnt/datasets/gqa_Files",
            "N": 1000000  # 可修改为10/100/1000/10000/100000/1000000
        }

    # cc3m
    if args.dataset == "cc3m":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/cc3m_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/cc3m-images",
            "output_root": "/mnt/datasets/cc3m_Files",
            "N": 1000000
        }
    # MSRVTT
    if args.dataset == "MSRVTT":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/MSRVTT_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/MSRVTT/videos/all",
            "output_root": "/mnt/datasets/MSRVTT_Files",
            "N": 1000000
        }
    # Clotho
    if args.dataset == "Clotho":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/Clotho_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/Clotho/development+evaluation",
            "output_root": "/mnt/datasets/Clotho_Files",
            "N": 1000000
        }
    # Clotho_E
    if args.dataset == "Clotho_E":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/ExpandedClotho_sort.csv",
            "image_src_dir": "/mnt/datasets/ExpandedClotho/wavs",
            "output_root": "/mnt/datasets/Clotho_E_Files",
            "N": 1000000
        }      
        
    import time
    start = time.time()
    # 执行拆分
    split_coco_dataset(**config)
    print(f"Files总耗时:{time.time() - start:.4f}秒")

