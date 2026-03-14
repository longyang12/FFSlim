# 写入阶段
from ffrecord import FileReader
import pickle
import random
from ffrecord_lru import LRUCache

def process_samples(samples,batch_size,file_path,num,cache_algorithm):
    # 构建batch阶段
    n = len(samples)//batch_size
    m = len(samples)%batch_size
    batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    if m > 0:
        batch_samples_list.append(samples[n*batch_size:]) 

    # 构建缓存
    if num != 0:
        if cache_algorithm == "lru":
            cache = LRUCache(num)
        hits = 0
        count = 0

    # 读取batch阶段
    reader = FileReader(file_path)
    # print(f"reader.n={reader.n}")
    # print(f"size={4*reader.n}")

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
                result = pickle.loads(sample)
                if num != 0:
                    cache.put(i,result)

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
    filename = "/mnt/datasets/flickr30k_FFRecord/mscoco_caption.ffr"
# flickr30k_entities
if args.dataset == "flickr30k_entities":
    filename = "/mnt/datasets/flickr30k_entities_FFRecord/mscoco_caption.ffr"
# mscoco
if args.dataset == "mscoco":
    filename = "/mnt/datasets/mscoco_FFRecord/mscoco_caption.ffr"
# vqa2
if args.dataset == "vqa2":
    filename = "/mnt/datasets/vqa2_FFRecord/mscoco_caption.ffr"
# gqa
if args.dataset == "gqa":
    filename = "/mnt/datasets/gqa_FFRecord/mscoco_caption.ffr"
# cc3m
if args.dataset == "cc3m":
    filename = "/mnt/datasets/cc3m_FFRecord/mscoco_caption.ffr"
# MSRVTT
if args.dataset == "MSRVTT":
    filename = "/mnt/datasets/MSRVTT_FFRecord/mscoco_caption.ffr"
# Clotho
if args.dataset == "Clotho":
    filename = "/mnt/datasets/Clotho_FFRecord/mscoco_caption.ffr"

samples = {"flickr30k":155060,"flickr30k_entities":158905,"mscoco":589785,"vqa2":443757,"gqa":943000,"cc3m":256877,"MSRVTT":200000-10,"Clotho":19690-5}


# 调用处理函数
sample_list = [i for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理
batch_size = 64
random.seed(123)
random.shuffle(sample_list)

import time
start = time.time()
process_samples(sample_list,batch_size,filename, args.num, args.cache_algorithm)
with open("/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"FFRecord random read耗时 at {args.num},{args.cache_algorithm},memory_size:{args.memory}:{time.time() - start:.4f}秒\n") 
