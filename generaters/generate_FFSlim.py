import os
import pandas as pd
import pickle
import hashlib
from ffrecord import FileWriter

def process_dataset(csv_path, image_src_dir, output_root):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(len(df))
    print(len(df)/5)
    # 创建哈希字典存储结构
    hash_dict = {}  # {image_hash: {'bytes': b'', 'captions': []}}

    # 遍历数据集计算哈希
    for index, row in df.iterrows():
        img_path = row['image_path']
        caption = row['image_caption']
        
        try:
            # 读取图像字节
            with open(os.path.join(image_src_dir,img_path), 'rb') as f:
                img_bytes = f.read()

            img_hash = hashlib.sha256(img_bytes).hexdigest()

            # 更新哈希字典
            if img_hash in hash_dict:
                hash_dict[img_hash]['captions'].append(caption)
            else:
                hash_dict[img_hash] = {
                    'image': img_bytes,
                    'captions': [caption]
                }
        except Exception as e:
            print(f"处理文件 {img_path} 出错: {e}")
            continue

    start = time.time()
    # 准备FFRecord写入
    os.makedirs(output_root, exist_ok=True)
    output_file = os.path.join(output_root, 'mscoco_caption.ffr')
    total_samples = len(hash_dict)
    
    # 初始化FFRecord写入器
    writer = FileWriter(output_file, total_samples)
    
    # 序列化并写入数据
    for img_data in hash_dict.values():
        # 构建数据对象
        data = {
            'image': img_data['image'],
            'captions': img_data['captions']
        }
        
        # 序列化数据（使用协议5支持大文件）
        serialized = pickle.dumps(data, protocol=5)
        writer.write_one(serialized)
    
    writer.close()
    print(f"成功写入 {total_samples} 个去重样本到 {output_file}")
    print(f"Slim_FFRecord写入耗时:{time.time() - start:.4f}秒") 

if __name__ == "__main__":
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
            "output_root": "/mnt/datasets/flickr30k_FFSlim"
        }

    # flickr30k_entities
    if args.dataset == "flickr30k_entities":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/flickr30k_entities_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/flickr30k_entities/flickr30k-images",
            "output_root": "/mnt/datasets/flickr30k_entities_FFSlim"
        }

    # mscoco
    if args.dataset == "mscoco":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/mscoco_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/mscoco/train2017",
            "output_root": "/mnt/datasets/mscoco_FFSlim"
        }


    # vqa2
    if args.dataset == "vqa2":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/vqa2_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/VQA2/vqa-images",
            "output_root": "/mnt/datasets/vqa2_FFSlim"
        }

    # gqa
    if args.dataset == "gqa":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/gqa_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/GQA/gqa-images",
            "output_root": "/mnt/datasets/gqa_FFSlim"
        }

    # cc3m
    if args.dataset == "cc3m":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/cc3m_sort.csv",
            "image_src_dir": "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/cc3m-images",
            "output_root": "/mnt/datasets/cc3m_FFSlim"
        }

    # MSRVTT
    if args.dataset == "MSRVTT":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/MSRVTT_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/MSRVTT/videos/all",
            "output_root": "/mnt/datasets/MSRVTT_FFSlim"
        }
    # Clotho
    if args.dataset == "Clotho":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/Clotho_sort.csv",
            "image_src_dir": "/mnt/datasets/liyaqi/Clotho/development+evaluation",
            "output_root": "/mnt/datasets/Clotho_FFSlim",
        }   
    # Clotho
    if args.dataset == "Clotho_E":
        config = {
            "csv_path": "/home/llm/experiment/dataset_Analysis_Modeling/processed_datasets/ExpandedClotho_sort.csv",
            "image_src_dir": "/mnt/datasets/ExpandedClotho/wavs",
            "output_root": "/mnt/datasets/Clotho_E_FFSlim",
        }      
    import time
    start = time.time()
    # 执行拆分
    process_dataset(**config)
    print(f"Slim_FFRecord总耗时:{time.time() - start:.4f}秒") 
    