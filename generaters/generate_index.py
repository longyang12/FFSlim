from ffrecord import FileReader
import pickle
import random

file_path = "/mnt/datasets/flickr30k_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/flickr30k_entities_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"

reader = FileReader(file_path)
samples = [i for i in range(reader.n)]
batch_size = 64
n = len(samples)//batch_size
m = len(samples)%batch_size
batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(0,n)]
if m > 0:
    batch_samples_list.append(samples[n*batch_size:])
print(f"n={n},m={m}")

nums = []
pre_sums = []
index = 0
for batch in batch_samples_list:
    # print(f"batch={batch}")
    datasets = reader.read(batch)
    for sample in datasets:
        obj = pickle.loads(sample)
        captions = obj["captions"]
        n = len(captions)
        nums.append(n)
        if len(pre_sums) == 0:
            pre_sums.append((n,1,n))
        else:
            pre_sums.append((pre_sums[-1][0]+n,pre_sums[-1][0]+1,pre_sums[-1][0]+n))
        index += 1
    # print(f"index={index},len(nums)={len(nums)},sum(nums)={sum(nums)},len(pre_sums)={len(pre_sums)}")

    # print(len(nums))
    # print(len(pre_sums))
    # print(nums)
    # print(pre_sums)
    # break
print(f"reader.n={reader.n}, index={index}")

data = {"nums":nums}
with open('/mnt/datasets/gqa_FFSlim/gqa_nums.pkl', 'wb') as f:
    pickle.dump(data, f)
data = {"pre_sums":pre_sums}
with open('/mnt/datasets/gqa_FFSlim/gqa.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"len(nums)={len(nums)},sum(nums)={sum(nums)},len(pre_sums)={len(pre_sums)}")




with open('/mnt/datasets/gqa_FFSlim/gqa_nums.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
nums = loaded_data["nums"]
with open('/mnt/datasets/gqa_FFSlim/gqa.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
pre_sums = loaded_data["pre_sums"]

print(f"len(nums)={len(nums)},sum(nums)={sum(nums)},len(pre_sums)={len(pre_sums)}")


