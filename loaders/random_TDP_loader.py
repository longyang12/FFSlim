import os
from PIL import Image
import random
import pandas as pd
import mmap
import sys
import argparse
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from cache.tdp_lru import LRUCache
from cache.tdp_quiver import Quiver
from cache.tdp_coordl import CoorDL
from cache.tdp_iacma import IACMACache
from cache.tdp_iacma_with_memory import IACMACacheWithMemory
from cache.tdp_iacma_with_heap import IACMACacheWithHeap
from cache.tdp_iacma_admission import IACMAAdmission
from cache.tdp_iacma_with_memory_admission import IACMACacheWithMemoryAdmission



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

def warm_up(cache, num, images, base_dir):
    for sid in range(num):
        if len(cache.cache) >= cache.capacity:
            break
        img_data = cache.get(images[sid].split('.')[0])
        # 缓存命中
        if img_data != -1:
            continue
        # 缓存未命中
        img_path = os.path.join(base_dir,"img/"+images[sid])
        # 处理图像文件
        if os.path.exists(img_path):
            try:
                with open(img_path,"rb") as f:
                    img_data = f.read()
            except Exception as e:
                print(f"图像文件 {images[sid]} 损坏: {str(e)}")
        else:
            print(f"图像文件 {images[sid]} 不存在")
        cache.warm_up_put(images[sid].split('.')[0],img_data)


def process_samples(sample_ids, base_dir, num, cache_algorithm):
    process_begin = time.perf_counter()
    csv_path = os.path.join(base_dir,base_dir.split("/")[-1]+".csv")
    df = pd.read_csv(csv_path)

    images = df["image_path"]
    captions = df["image_caption"]

    # nums = []
    # for index in range(len(images)):
    #     nums.append(len(images[index])+len(captions[index]))
    # print(f"size={sum(nums)}")

    # 构建缓存
    if num != 0:
        if cache_algorithm == "lru":
            cache = LRUCache(num)
        elif cache_algorithm == "quiver":
            cache = Quiver(num)
        elif cache_algorithm == "coordl":
            cache = CoorDL(num)   
        elif cache_algorithm == "iacma":
            cache = IACMACache(num, images)
        elif cache_algorithm == "iacma_with_memory":
            cache = IACMACacheWithMemory(num, images)
        elif cache_algorithm == "iacma_with_heap":
            cache = IACMACacheWithHeap(num, images)
        elif cache_algorithm == "iacma_admission":
            cache = IACMAAdmission(num, images)
        elif cache_algorithm == "iacma_with_memory_admission":
            cache = IACMACacheWithMemoryAdmission(num, images)
        else:
            print(f"未知的缓存算法: {cache_algorithm}")
            return
        cache_get_calls = 0
        cache_put_calls = 0
        cache_get_overhead = 0.0
        cache_put_overhead = 0.0

    hits = 0
    count = 0
    id = 0
    for sid in sample_ids:
        if num != 0:
            count += 1
            cache_get_start = time.perf_counter()
            img_data = cache.get(images[sid].split('.')[0], id)
            cache_get_overhead += time.perf_counter() - cache_get_start
            cache_get_calls += 1
            # 缓存命中
            if img_data != -1:
                hits += 1
                result = (img_data,captions[sid])
                continue
        # 缓存未命中
        img_path = os.path.join(base_dir,"img/"+images[sid])
        # 处理图像文件
        if os.path.exists(img_path):
            try:
                with open(img_path,"rb") as f:
                    img_data = f.read()
                # img_data = read_file_with_o_direct(img_path)
            except Exception as e:
                print(f"图像文件 {images[sid]} 损坏: {str(e)}")
        else:
            print(f"图像文件 {images[sid]} 不存在")
        if num != 0:
            cache_put_start = time.perf_counter()
            cache.put(images[sid].split('.')[0],img_data, id)
            cache_put_overhead += time.perf_counter() - cache_put_start
            cache_put_calls += 1
        txt_data = captions[sid]
        result = (img_data,txt_data)
        id += 1
        
    if num != 0:
        process_total = time.perf_counter() - process_begin
        cache_overhead_total = cache_get_overhead + cache_put_overhead
        avg_get_us = (cache_get_overhead / cache_get_calls * 1e6) if cache_get_calls else 0.0
        avg_put_us = (cache_put_overhead / cache_put_calls * 1e6) if cache_put_calls else 0.0
        overhead_ratio = (cache_overhead_total / process_total) if process_total > 0 else 0.0
        with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
            f.write(f"cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")
            f.write(
                "cache mgmt overhead: "
                f"get_calls={cache_get_calls},put_calls={cache_put_calls},"
                f"get_total={cache_get_overhead:.6f}s,put_total={cache_put_overhead:.6f}s,\n"
                f"avg_get={avg_get_us:.2f}us,avg_put={avg_put_us:.2f}us,"
                f"cache_total={cache_overhead_total:.6f}s,"
                f"process_total={process_total:.6f}s,"
                f"cache_overhead_ratio={overhead_ratio:.6f}\n"
            )
        # if args.cache_algorithm in ["iacma","iacma_with_memory","iacma_admission","iacma_with_memory_admission"]:
        #     # 输出对象重要性
        #     with open(f"/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/{args.dataset}_TDP_{args.num}_{args.cache_algorithm}_object_importance.log","w") as f:
        #         f.write(f"dataset:{args.dataset}_TDP,num:{args.num},cache_algorithm:{args.cache_algorithm},cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")
        #         for key, score_shifts in cache.get_object_importance().items():
        #             f.write(f"key:{key},score_shifts:{score_shifts}\n")

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
    filename = "/mnt/datasets/flickr30k_TDP"
# flickr30k_entities
if args.dataset == "flickr30k_entities":
    filename = "/mnt/datasets/flickr30k_entities_TDP"
# mscoco
if args.dataset == "mscoco":
    filename = "/mnt/datasets/mscoco_TDP"
# vqa2
if args.dataset == "vqa2":
    filename = "/mnt/datasets/vqa2_TDP"
# gqa
if args.dataset == "gqa":
    filename = "/mnt/datasets/gqa_TDP"
# cc3m
if args.dataset == "cc3m":
    filename = "/mnt/datasets/cc3m_TDP"
# MSRVTT
if args.dataset == "MSRVTT":
    filename = "/mnt/datasets/MSRVTT_TDP"
# Clotho
if args.dataset == "Clotho":
    filename = "/mnt/datasets/Clotho_TDP"
# Clotho_E
if args.dataset == "Clotho_E":
    filename = "/mnt/datasets/Clotho_E_TDP"

# mscoco_hdd
if args.dataset == "mscoco_hdd":
    filename = "/mnt/datasets_hdd/mscoco_TDP"


samples = {"cc3m":256878,"flickr30k":155065,"flickr30k_entities":158910,"mscoco":589790,"mscoco_hdd":589790,"vqa2":443757,"gqa":943000,"MSRVTT":200000,"Clotho":19690,"Clotho_E":98450}

# 调用处理函数
sample_list = [i for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理
random.seed(123)
random.shuffle(sample_list)

start = time.time()
processed = process_samples(sample_list, filename, args.num, args.cache_algorithm)
with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"TDP random read耗时 at {args.num},{args.cache_algorithm},memory_size:{args.memory}:{time.time() - start:.4f}秒\n") 

# test
# sudo python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py flickr30k  12405 iacma 0
