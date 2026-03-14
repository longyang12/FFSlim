import pandas as pd
import os
import shutil
from tqdm import tqdm

def process_coco_dataset(csv_path, image_src_dir, output_dir, new_csv):
    """
    处理COCO数据集的核心函数
    
    参数：
    input_csv : str   - 原始CSV文件路径
    output_dir : str  - 图片输出目录
    new_csv : str     - 新CSV文件路径
    """
    # 创建目标目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始CSV文件（网页1、2推荐方法）
    df = pd.read_csv(csv_path)
    
    start = time.time()
    # 准备新数据集
    new_data = []
    
    hash_table = {}
    # 带进度条的遍历处理
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        src_path = os.path.join(image_src_dir, row['image_path'])
        
        try:
            # 提取文件名（网页9路径处理）
            filename = os.path.basename(src_path)
            dest_path = os.path.join(output_dir, filename)
            
            # 读写非重复图片
            key = src_path.split("/")[-1]
            if key not in hash_table:
                with open(src_path, 'rb') as src_file:
                    binary_data = src_file.read()  # 读取二进制数据[3,10](@ref)
                with open(dest_path, 'wb') as dest_file:
                    dest_file.write(binary_data)   # 写入二进制数据[9,10](@ref)
                hash_table[key] = 1
                
            # 复制文件（网页4推荐方法）
            # shutil.copy2(src_path, dest_path)
            
            # 记录新数据（网页6 CSV构建方法）
            new_data.append({
                'image_path': filename,
                'image_caption': row['image_caption']
            })
        except Exception as e:
            print(f"文件处理失败：{src_path}，错误：{str(e)}")
    
    # 生成新CSV（网页7推荐方法）
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_csv, index=False)

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
            "output_dir": "/mnt/datasets/flickr30k_TDP/img",
            "new_csv": "/mnt/datasets/flickr30k_TDP/flickr30k_TDP.csv"
        }

    # flickr30k_entities
    if args.dataset == "flickr30k_entities":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/flickr30k_entities_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/flickr30k-images",
            "output_dir": "/mnt/datasets/flickr30k_entities_TDP/img",
            "new_csv": "/mnt/datasets/flickr30k_entities_TDP/flickr30k_entities_TDP.csv"
        }

    # mscoco
    if args.dataset == "mscoco":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/mscoco_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/mscoco/train2017",
            "output_dir": "/mnt/datasets/mscoco_TDP/img",
            "new_csv": "/mnt/datasets/mscoco_TDP/mscoco_TDP.csv"
        }

    # vqa2
    if args.dataset == "vqa2":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/vqa2_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/vqa-images",
            "output_dir": "/mnt/datasets/vqa2_TDP/img",
            "new_csv": "/mnt/datasets/vqa2_TDP/vqa2_TDP.csv"
        }

    # gqa
    if args.dataset == "gqa":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/gqa_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/gqa-images",
            "output_dir": "/mnt/datasets/gqa_TDP/img",
            "new_csv": "/mnt/datasets/gqa_TDP/gqa_TDP.csv"
        }

    # cc3m
    if args.dataset == "cc3m":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/cc3m_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/cc3m-images",
            "output_dir": "/mnt/datasets/cc3m_TDP/img",
            "new_csv": "/mnt/datasets/cc3m_TDP/cc3m_TDP.csv"
        }

    # MSRVTT
    if args.dataset == "MSRVTT":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/MSRVTT_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/MSRVTT/videos/all",
            "output_dir": "/mnt/datasets/MSRVTT_TDP/img",
            "new_csv": "/mnt/datasets/MSRVTT_TDP/MSRVTT_TDP.csv"
        }
    # Clotho
    if args.dataset == "Clotho":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/Clotho_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/Clotho/development+evaluation",
            "output_dir": "/mnt/datasets/Clotho_TDP/img",
            "new_csv": "/mnt/datasets/Clotho_TDP/Clotho_TDP.csv"
        }
    # Clotho_E
    if args.dataset == "Clotho_E":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/ExpandedClotho_sort.csv",
            "image_src_dir": "/mnt/datasets/ExpandedClotho/wavs",
            "output_dir": "/mnt/datasets/Clotho_E_TDP/img",
            "new_csv": "/mnt/datasets/Clotho_E_TDP/Clotho_E_TDP.csv"
        }

    import time
    start = time.time()
    process_coco_dataset(**config)
    print(f"Files总耗时:{time.time() - start:.4f}秒")