# 写入阶段
from ffrecord import FileReader
import pickle


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

def process_samples(samples,batch_size,file_path):
    # 构建batch阶段
    n = len(samples)//batch_size
    m = len(samples)%batch_size
    batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    if m > 0:
        batch_samples_list.append(samples[n*batch_size:])
    print(f"n={n},m={m},size={n*batch_size+m}")

    # 加载完全平衡二叉搜索树
    if args.dataset == "vqa2":
        with open('/mnt/datasets/vqa2_FFSlim/vqa2.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        pre_sums = loaded_data["pre_sums"]
    if args.dataset == "gqa":
        with open('/mnt/datasets/gqa_FFSlim/gqa.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        pre_sums = loaded_data["pre_sums"]

    # 读取batch阶段
    reader = FileReader(file_path)
    print(f"reader.n={reader.n}")

    if args.dataset in ["flickr30k","flickr30k_entities","mscoco"]:
        print(f"size={4*reader.n}")
    else:
        print(f"size={4*reader.n+12*len(pre_sums)}")

    index_time = []    

    for batch in batch_samples_list:
        # print(f"batch:{batch}")
        if args.dataset in ["flickr30k","flickr30k_entities","mscoco"]:
            # 哈希映射
            X = 5
            first_index = [i//X for i in batch]
            second_index = [i%X for i in batch]
            for index,i in enumerate(first_index):
                # print(f"first_index={i},second_index={second_index[index]}")
                datasets = reader.read([i])
                for sample in datasets:
                    samples1 = pickle.loads(sample)
                    result = [samples1["image"],samples1["captions"][second_index[index]]]
        else:
            # 完全平衡二叉搜索树
            for i in batch:
                start = time.time()
                temp = find_index(i,pre_sums)
                index_time.append(time.time() - start)
                # print(f"i={i},temp:{temp}")
                # print(f"first_index={temp[0]},second_index={temp[1]}")
                datasets = reader.read([temp[0]])
                for sample in datasets:
                    samples1 = pickle.loads(sample)
                    result = [samples1["image"],samples1["captions"][temp[1]]]
    print(f"index_time={sum(index_time):.4f}")

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
    filename = "/mnt/datasets/flickr30k_FFSlim/flickr30k_caption.ffr"
# flickr30k_entities
if args.dataset == "flickr30k_entities":
    filename = "/mnt/datasets/flickr30k_entities_FFSlim/flickr30k_entities_caption.ffr"
# mscoco
if args.dataset == "mscoco":
    filename = "/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
# vqa2
if args.dataset == "vqa2":
    filename = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
# gqa
if args.dataset == "gqa":
    filename = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"

samples = {"flickr30k":155060,"flickr30k_entities":158905,"mscoco":589785,"vqa2":443757,"gqa":943000}
# 调用处理函数
sample_list = [i+1 for i in range(samples[args.dataset])]  # 1-3号文件不存在用于演示错误处理
batch_size = 64

import time
start = time.time()
process_samples(sample_list,batch_size,filename)
print(f"FFSlim seq read耗时:{time.time() - start:.4f}秒") 
