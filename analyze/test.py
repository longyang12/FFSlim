from ffrecord import FileReader
import pickle

file_path = "/mnt/datasets/flickr30k_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/flickr30k_entities_FFSlim/flickr30k_entities_caption.ffr"
# file_path = "/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/cc3m_FFSlim/mscoco_caption.ffr"


reader = FileReader(file_path)
print(f"reader.n={reader.n}")
datasets = reader.read([0])
for sample in datasets:
    samples1 = pickle.loads(sample)
    print(samples1)

datasets = reader.read([reader.n-1])
for sample in datasets:
    samples1 = pickle.loads(sample)
    print(samples1)


# nums = []

# sum = 0
# for i in range(reader.n):
#     datasets = reader.read([i])
#     for sample in datasets:
#         samples1 = pickle.loads(sample)
#         n = len(samples1["captions"])
#         nums.append(n)
#         if n != 1:
#             # print(f"i={i},n={n},captions={samples1["captions"]}")
#             sum += (n-1)
# print(sum)
