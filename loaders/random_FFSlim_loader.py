# 写入阶段
from ffrecord import FileReader
import pickle
import random
import sys
import os
import argparse
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from cache.ffslim_lru import LRUCache
from cache.ffslim_quiver import Quiver
from cache.ffslim_coordl import CoorDL
from cache.ffslim_dgim_lru import DGIMLRUCache
from cache.ffslim_dgim_quiver import DGIMQuiver
from cache.ffslim_dgim_coordl import DGIMCoorDL
from cache.ffslim_dgim_iacma import DGIMIACMACache
from cache.ffslim_dgim_iacma_with_memory import DGIMIACMACacheWithMemory
from cache.ffslim_dgim_iacma_admission import DGIMIACMAAdmissionCache
from cache.ffslim_dgim_iacma_with_memory_admission import DGIMIACMAAdmissionCacheWithMemory

# def find_index(x,pre_sums):
#     # 计算全局累积长度中第一个 >=x 的索引
#     left, right = 0, len(pre_sums)-1
#     while left <= right:
#         mid = (left + right) // 2
#         if pre_sums[mid][0] > x:
#             right = mid - 1
#         else:
#             left = mid + 1
#     # if x == 0:
#     #     print(f"left={left},right={right},pre_sums[left]={pre_sums[left]}")
#     # 验证是否在对应数组范围内
#     if left < len(pre_sums) and pre_sums[left][1] <= x <= pre_sums[left][2]:
#         return (left,x-pre_sums[left][1])
#     return -1

def find_index(pre_sums, x):
    # 提取所有的“最大编号”创建一个虚拟数组（或者直接针对元组查找）
    # 我们查找第一个“最大编号 >= x”的元素
    # 由于元组比较是逐个字段进行的，我们可以利用这一点
    
    # 方式 A：手动二分（性能最强，无额外空间）
    low = 0
    high = len(pre_sums) - 1
    while low <= high:
        mid = (low + high) // 2
        node_min, node_max = pre_sums[mid]

        # print(f"x={x},low={low},high={high},mid={mid},pre_sums[mid]={pre_sums[mid]}")

        if node_min <= x <= node_max:
            return (mid,x-node_min)
        elif x < node_min:
            high = mid - 1
        else:
            low = mid + 1

    # print(f"x={x},low={low},high={high},mid={mid},pre_sums[mid]={pre_sums[mid]}")
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
        elif args.dataset in ["MSRVTT"]:
            # 哈希映射
            X = 20
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
            temp = find_index(pre_sums,i)
            datasets = reader.read([temp[0]])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                cache.put(i-temp[1],samples1)

def process_samples(samples,batch_size,file_path,num,cache_algorithm):
    process_begin = time.perf_counter()
    # 构建batch阶段
    n = len(samples)//batch_size
    m = len(samples)%batch_size
    batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    if m > 0:
        batch_samples_list.append(samples[n*batch_size:])
    with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
        f.write(f"n={n},m={m},size={n*batch_size+m}\n")

    X = 0
    pre_sums = 0
    hash_index = dict()

    # 加载索引阶段
    # 加载哈希索引
    if args.index_structure == "hash_index":
        with open(f'/mnt/datasets/{args.dataset}_FFSlim/{args.dataset}-hash.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        hash_index = loaded_data["hash_index"]

    # 加载自适应索引
    if args.index_structure == "adaptive_index":
        if args.dataset in ["cc3m"]:
            X = 1
        elif args.dataset in ["flickr30k","flickr30k_entities","mscoco","mscoco_hdd", "Clotho","Clotho_E"]:
            X = 5
        elif args.dataset in ["MSRVTT"]:
            X = 20
        elif args.dataset in ["vqa2","gqa"]:
            with open(f'/mnt/datasets/{args.dataset}_FFSlim/{args.dataset}-avl.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            pre_sums = loaded_data["pre_sums"] # (start id,end id) ...



    # 读取batch阶段
    reader = FileReader(file_path)
    with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
        f.write(f"reader.n={reader.n}\n")

    # 构建缓存
    if num != 0:
        if cache_algorithm == "lru":
            cache = LRUCache(num)
        elif cache_algorithm == "quiver":
            cache = Quiver(num)
        elif cache_algorithm == "coordl":
            cache = CoorDL(num)
        elif cache_algorithm == "dgim_lru":
            cache = DGIMLRUCache(num)
        elif cache_algorithm == "dgim_quiver":
            cache = DGIMQuiver(num)
        elif cache_algorithm == "dgim_coordl":
            cache = DGIMCoorDL(num)
        elif cache_algorithm == "dgim_iacma":
            cache = DGIMIACMACache(num)
        elif cache_algorithm == "dgim_iacma_with_memory":
            cache = DGIMIACMACacheWithMemory(num)
        elif cache_algorithm == "dgim_iacma_admission":
            cache = DGIMIACMAAdmissionCache(num)
        elif cache_algorithm == "dgim_iacma_with_memory_admission":
            cache = DGIMIACMAAdmissionCacheWithMemory(num)   
        hits = 0
        count = 0
        cache_get_calls = 0
        cache_put_calls = 0
        cache_get_overhead = 0.0
        cache_put_overhead = 0.0

    index_time = []
    id = 0
    for batch in batch_samples_list:
        # print(f"batch:{batch}")
        for i in batch:
            if num != 0:
                count += 1
                cache_get_start = time.perf_counter()
                result = cache.get(i,id)
                cache_get_overhead += time.perf_counter() - cache_get_start
                cache_get_calls += 1
                # 缓存命中
                if result != -1:
                    hits += 1
                    continue
            # 缓存未命中
            #根据哈希表检索
            if args.index_structure=="hash_index":
                start = time.time()
                temp = hash_index[i] 
                index_time.append(time.time() - start)
                datasets = reader.read([temp[0]])
                for sample in datasets:
                    samples1 = pickle.loads(sample)
                    result = [samples1["image"],samples1["captions"][temp[1]]]
                    if num != 0:
                            cache_put_start = time.perf_counter()
                            cache.put(i-temp[1],samples1,id)
                            cache_put_overhead += time.perf_counter() - cache_put_start
                            cache_put_calls += 1
            # 根据自适应索引检索
            elif args.index_structure=="adaptive_index":
                if args.dataset in ["cc3m","flickr30k","flickr30k_entities","mscoco","mscoco_hdd" ,"MSRVTT","Clotho","Clotho_E"]:
                    start = time.time()
                    temp = [i//X,i%X] # 从0开始编号
                    index_time.append(time.time() - start)
                    
                    try:  
                        datasets = reader.read([temp[0]])
                    except:
                        print(temp)

                    for sample in datasets:
                        samples1 = pickle.loads(sample)
                        result = [samples1["image"],samples1["captions"][temp[1]]]
                        if num != 0:
                            cache_put_start = time.perf_counter()
                            cache.put(i-i%X,samples1,id)
                            cache_put_overhead += time.perf_counter() - cache_put_start
                            cache_put_calls += 1
                else:
                    # 完全平衡二叉搜索树
                    start = time.time()
                    temp = find_index(pre_sums,i) 
                    index_time.append(time.time() - start)
                    # print(f"i={i},temp:{temp}")
                    # print(f"first_index={temp[0]},second_index={temp[1]}")
                    datasets = reader.read([temp[0]])
                    for sample in datasets:
                        samples1 = pickle.loads(sample)
                        result = [samples1["image"],samples1["captions"][temp[1]]]
                        # if i == 1:
                        #     print(f'image:{type(samples1["image"])},caption:{type(samples1["captions"][0])}')
                        if num != 0:
                            cache_put_start = time.perf_counter()
                            cache.put(i-temp[1],samples1,id)
                            cache_put_overhead += time.perf_counter() - cache_put_start
                            cache_put_calls += 1
            id += 1

    with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
        f.write(f"index_time={sum(index_time):.4f}\n")
    if num != 0:
        process_total = time.perf_counter() - process_begin
        cache_overhead_total = cache_get_overhead + cache_put_overhead
        avg_get_us = (cache_get_overhead / cache_get_calls * 1e6) if cache_get_calls else 0.0
        avg_put_us = (cache_put_overhead / cache_put_calls * 1e6) if cache_put_calls else 0.0
        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
            f.write(f"cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")
            f.write(
                "cache mgmt overhead: "
                f"get_calls={cache_get_calls},put_calls={cache_put_calls},"
                f"get_total={cache_get_overhead:.6f}s,put_total={cache_put_overhead:.6f}s,\n"
                f"avg_get={avg_get_us:.2f}us,avg_put={avg_put_us:.2f}us,"
                f"cache_total={cache_overhead_total:.6f}s,"
                f"process_total={process_total:.6f}s,"
                f"cache_overhead_ratio={cache_overhead_total/process_total:.6f}\n"
            )
            
        # if args.cache_algorithm in ["dgim_iacma","dgim_iacma_with_memory","dgim_iacma_admission","dgim_iacma_with_memory_admission"]:
        #     # 输出对象重要性
        #     with open(f"/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/{args.dataset}_FFSlim_{args.num}_{args.cache_algorithm}_object_importance.log","w") as f:
        #         f.write(f"dataset:{args.dataset}_FFSlim,num:{args.num},cache_algorithm:{args.cache_algorithm},cache hits:{hits},requests:{count},cache hit rate:{hits/count}\n")
        #         for key, score_shifts in cache.get_object_importance().items():
        #             f.write(f"key:{key},score_shifts:{score_shifts}\n")

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('dataset', type=str, help='输入数据集名称')
parser.add_argument('num', type=int, help='输入缓存大小')
parser.add_argument('cache_algorithm', type=str, help='输入缓存算法')
parser.add_argument('index_structure', type=str, help='索引结构')
parser.add_argument('memory', type=int, help='输入限制内存容量大小')



# 解析参数
args = parser.parse_args()
with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"dataset:{args.dataset}_FFSlim,num:{args.num},cache_algorithm:{args.cache_algorithm},index_structure:{args.index_structure},memory_size:{args.memory}\n")


# cc3m
if args.dataset == "cc3m":
    filename = "/mnt/datasets/cc3m_FFSlim/mscoco_caption.ffr"
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
# MSRVTT
if args.dataset == "MSRVTT":
    filename = "/mnt/datasets/MSRVTT_FFSlim/mscoco_caption.ffr"
# Clotho
if args.dataset == "Clotho":
    filename = "/mnt/datasets/Clotho_FFSlim/mscoco_caption.ffr"
# Clotho_E
if args.dataset == "Clotho_E":
    filename = "/mnt/datasets/Clotho_E_FFSlim/mscoco_caption.ffr"

# mscoco_hdd
if args.dataset == "mscoco_hdd":
    filename = "/mnt/datasets_hdd/mscoco_FFSlim/mscoco_caption.ffr"

samples = {"cc3m":256878,"flickr30k":155065,"flickr30k_entities":158910,"mscoco":589790,"mscoco_hdd":589790,"vqa2":443757,"gqa":943000,"MSRVTT":200000,"Clotho":19690,"Clotho_E":98450}

# 调用处理函数
sample_list = [i for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理

batch_size = 64
random.seed(123)
random.shuffle(sample_list)

start = time.time()
process_samples(sample_list,batch_size,filename,args.num,args.cache_algorithm)
with open(f"/home/llm/experiment/dataset_Analysis_Modeling/output.log","a+") as f:
    f.write(f"FFSlim random read耗时 at {args.num},{args.cache_algorithm},memory_size:{args.memory}:{time.time() - start:.4f}秒\n") 


# test
# sudo python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py flickr30k  12405  dgim_iacma adaptive_index 0
