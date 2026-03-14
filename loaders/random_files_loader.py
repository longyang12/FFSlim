import os
from PIL import Image
import random
import mmap
from files_lru import LRUCache

BLOCK_SIZE = 4096

def aligned_alloc(size):
    return mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

def read_file_with_o_direct(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        file_size = os.fstat(fd).st_size
        buffer_size = (file_size + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        buf = aligned_alloc(buffer_size)

        total_read = 0
        while total_read < file_size:
            remaining = file_size - total_read
            to_read = min(remaining, BLOCK_SIZE * 1)  # 每次最多读 1 blocks

            # 确保 to_read 是 block size 的整数倍
            to_read = (to_read + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

            # 正确调用方式：只传一个整数，表示想读多少字节
            data = os.read(fd, to_read)

            if not data:
                break

            # 将数据复制到 mmap 缓冲区中
            buf[total_read:total_read + len(data)] = data
            total_read += len(data)

        return buf[:file_size]  # 返回实际文件大小的数据

    finally:
        os.close(fd)

def process_samples(sample_ids, base_dir, num, cache_algorithm):
    """
    根据样本ID数组处理对应的图像和文本文件
    返回结构：{id: (图像对象, 文本内容), ...}
    
    参数：
        sample_ids: 需要处理的样本编号列表（int列表）
        base_dir: 文件存储根目录（默认data）
    """

    # 构建缓存
    if num != 0:
        if cache_algorithm == "lru":
            cache = LRUCache(num)
        hits = 0
        count = 0

    for sid in sample_ids:
        if num != 0:
            count += 1
            result = cache.get(sid)
            # 缓存命中
            if result != -1:
                hits += 1
                continue
        # 缓存未命中
        # 生成6位补零文件名
        file_prefix = f"{sid:06d}"
        img_path = os.path.join(base_dir, f"{file_prefix}.jpg")
        txt_path = os.path.join(base_dir, f"{file_prefix}.txt")

        # 处理图像文件
        if os.path.exists(img_path):
            try:
                with open(img_path,"rb") as f:
                    img_data = f.read()
                # img_data = read_file_with_o_direct(img_path)
            except Exception as e:
                with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
                    f.write(f"图像文件 {file_prefix}.jpg 损坏: {str(e)}\n")
        else:
            with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
                f.write(f"图像文件 {file_prefix}.jpg 不存在\n")

        # 处理文本文件
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    txt_data = f.read()
                # txt_data = read_file_with_o_direct(img_path)
            except Exception as e:
                with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
                    f.write(f"文本文件 {file_prefix}.txt 读取失败: {str(e)}\n")
        else:
            with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
                f.write(f"文本文件 {file_prefix}.txt 不存在\n")

        result = {"image":img_data,"caption":txt_data}
        if num != 0:
            cache.put(sid, result)
            
    if num != 0:
        with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
            f.write(f"cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")

import argparse
# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('dataset', type=str, help='输入数据集名称')
parser.add_argument('num', type=int, help='输入缓存大小')
parser.add_argument('cache_algorithm', type=str, help='输入缓存算法')
parser.add_argument('memory', type=int, help='输入限制内存容量大小')

# 解析参数
args = parser.parse_args()
with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"dataset:{args.dataset},num:{args.num},cache_algorithm:{args.cache_algorithm},memory_size:{args.memory}\n")


# flickr30k
if args.dataset == "flickr30k":
    filename = "/mnt/datasets/flickr30k_Files/000000"
# flickr30k_entities
if args.dataset == "flickr30k_entities":
    filename = "/mnt/datasets/flickr30k_entities_Files/000000"
# mscoco
if args.dataset == "mscoco":
    filename = "/mnt/datasets/mscoco_Files/000000"
# vqa2
if args.dataset == "vqa2":
    filename = "/mnt/datasets/vqa2_Files/000000"
# gqa
if args.dataset == "gqa":
    filename = "/mnt/datasets/gqa_Files/000000"
# cc3m
if args.dataset == "cc3m":
    filename = "/mnt/datasets/cc3m_Files/000000"
# MSRVTT
if args.dataset == "MSRVTT":
    filename = "/mnt/datasets/MSRVTT_Files/000000"
# Clotho
if args.dataset == "Clotho":
    filename = "/mnt/datasets/Clotho_Files/000000"

samples = {"flickr30k":155060,"flickr30k_entities":158905,"mscoco":589785,"vqa2":443757,"gqa":943000,"cc3m":256877,"MSRVTT":200000-10,"Clotho":19690-5}

# 调用处理函数
sample_list = [i for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理
random.seed(123)
random.shuffle(sample_list)
# print(f"size={len(filename)}")

import time
start = time.time()
process_samples(sample_list, filename, args.num, args.cache_algorithm)
with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"Files random read耗时 at {args.num},{args.cache_algorithm},memory_size:{args.memory}:{time.time() - start:.4f}秒\n") 
    
