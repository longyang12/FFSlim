# 写入阶段
from ffrecord import FileReader
import pickle
import random
from cache.ffslim_dgim_lru import LRUCache
from cgfg_lfu import LFUCache
from cache.ffslim_dgim_iacma import IACMACache
from cache.ffslim_dgim_iacma_with_heap import IACMACacheWithHeap
from cache.ffslim_lru import FGFGLRUCache
from fgfg_lfu import FGFGLFUCache
from cache.ffslim_dgim_iacma_with_memory import IACMACacheWithMemory
from cache.ffslim_coordl import CoorDL
from cache.ffslim_quiver import Quiver
import argparse
import os


def find_index(x,pre_sums):
    # 计算全局累积长度中第一个 >=x 的索引
    left, right = 0, len(pre_sums)-1
    while left <= right:
        mid = (left + right) // 2
        if pre_sums[mid][0] >= x:
            right = mid - 1
        else:
            left = mid + 1
    # if x == 0:
    #     print(f"left={left},right={right},pre_sums[left]={pre_sums[left]}")
    # 验证是否在对应数组范围内
    if left < len(pre_sums) and pre_sums[left][1] <= x <= pre_sums[left][2]:
        return (left,x-pre_sums[left][1])
    return -1

def warm_up(cache,num,reader,pre_sums):
    for i in range(num):
        result = cache.get(i)
        # 缓存命中
        if result != -1:
            continue
        if args.dataset in ["flickr30k","flickr30k_entities","mscoco"]:
            # 哈希映射
            X = 5
            datasets = reader.read([i//X])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                cache.put(i-i%X,samples1)
        elif args.dataset in ["cc3m"]:
            datasets = reader.read([i])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                cache.put(i,samples1)
        else:
            # 完全平衡二叉搜索树
            temp = find_index(i,pre_sums)
            datasets = reader.read([temp[0]])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                cache.put(i-temp[1],samples1)

# 每个epoch每个media随机采样一个text
def process_samples(samples,batch_size,file_path,num,cache_algorithm):
    # 构建batch阶段
    n = len(samples)//batch_size
    m = len(samples)%batch_size
    batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    if m > 0:
        batch_samples_list.append(samples[n*batch_size:])
    with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
        f.write(f"n={n},m={m},size={n*batch_size+m}\n")


    # 读取batch阶段
    reader = FileReader(file_path)
    with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
        f.write(f"reader.n={reader.n}\n")


    # 构建缓存
    if num != 0:
        if cache_algorithm == "fgfglru":
            cache = FGFGLRUCache(num)
        elif cache_algorithm == "fgfglfu":
            cache = FGFGLFUCache(num)
        elif cache_algorithm == "lru":
            cache = LRUCache(num)
        elif cache_algorithm == "lfu":
            cache = LFUCache(num)
        elif cache_algorithm == "iacma_with_heap":
            cache = IACMACacheWithHeap(num)
        elif cache_algorithm == "iacma":
            cache = IACMACache(num)
        elif cache_algorithm == "iacma_with_memory":
            cache = IACMACacheWithMemory(num)
        elif cache_algorithm == "quiver":
            cache = Quiver(num)
        elif cache_algorithm == "coordl":
            cache = CoorDL(num)
            # # 预热
            # warm_up_start = time.time()
            # warm_up(cache,num,reader,pre_sums)
            # with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output_{args.num}_{current_pid}.log","a+") as f:
            #     f.write(f"FFSlim warm_up耗时 at {args.num},{args.cache_algorithm},memory_size:{args.memory}:{time.time() - warm_up_start:.4f}秒\n") 
        hits = 0
        count = 0

    for batch in batch_samples_list:
        for i in batch:
            if num != 0:
                count += 1
                result = cache.get(i)
                # 缓存命中
                if result != -1:
                    hits += 1
                    continue
            # 缓存未命中
            datasets = reader.read([i])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                result = [samples1["image"],samples1["captions"][0]]

    if num != 0:
        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
            f.write(f"cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")
                

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('dataset', type=str, help='输入数据集名称')
parser.add_argument('num', type=int, help='输入缓存大小')
parser.add_argument('cache_algorithm', type=str, help='输入缓存算法')
parser.add_argument('memory', type=int, help='输入限制内存容量大小')


# 解析参数
args = parser.parse_args()
with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"dataset:{args.dataset},num:{args.num},cache_algorithm:{args.cache_algorithm},memory_size:{args.memory}\n")

# flickr30k
if args.dataset == "flickr30k":
    filename = "/mnt/datasets/flickr30k_FFSlim/mscoco_caption.ffr"
# flickr30k_entities
if args.dataset == "flickr30k_entities":
    filename = "/mnt/datasets/flickr30k_entities_FFSlim/mscoco_caption.ffr"
# mscoco
if args.dataset == "mscoco":
    filename = "/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
# vqa2
if args.dataset == "vqa2":
    filename = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
# gqa
if args.dataset == "gqa":
    filename = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"
# cc3m
if args.dataset == "cc3m":
    filename = "/mnt/datasets/cc3m_FFSlim/mscoco_caption.ffr"


samples = {"flickr30k":31012,"flickr30k_entities":31781,"mscoco":117957,"vqa2":82772,"gqa":72138,"cc3m":256877}

# 调用处理函数
sample_list = [i+1 for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理

batch_size = 64
random.seed(123)
random.shuffle(sample_list)

import time
start = time.time()
process_samples(sample_list,batch_size,filename,args.num,args.cache_algorithm)
with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"FFSlim random read耗时 at S1:{time.time() - start:.4f}秒\n") 
