# 写入阶段
from ffrecord import FileReader
import pickle


def process_samples(samples,batch_size,file_path):
    # 构建batch阶段
    start = time.time()
    n = len(samples)//batch_size
    m = len(samples)%batch_size
    batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    if m > 0:
        batch_samples_list.append(samples[n*batch_size:])
    print(f"FFRecord map and read耗时:{time.time() - start:.4f}秒") 

    # 读取batch阶段
    reader = FileReader(file_path)
    print(f"reader.n={reader.n}")
    print(f"size={4*reader.n}")
    
    for batch in batch_samples_list:
        for i in batch:
            datasets = reader.read([i])
            for sample in datasets:
                result = pickle.loads(sample)


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

samples = {"flickr30k":155070,"flickr30k_entities":158915,"mscoco":589860,"vqa2":443757,"gqa":943000}
sample_list = [i for i in range(samples[args.dataset])]  
batch_size = 64

import time
start = time.time()
process_samples(sample_list,batch_size,filename)
print(f"FFRecord seq read耗时:{time.time() - start:.4f}秒") 
