from ffrecord import FileReader
import pickle
import random


file_path = "/mnt/datasets/flickr30k_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
# file_path = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"

# reader = FileReader(file_path)
# samples = [i for i in range(reader.n)]
# batch_size = 64
# n = len(samples)//batch_size
# m = len(samples)%batch_size
# batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(0,n)]
# if m > 0:
#     batch_samples_list.append(samples[n*batch_size:])
# print(f"n={n},m={m}")


# hash_index = dict()
# key_index = 1
# Media2Text_index = 0
# for batch in batch_samples_list:
#     datasets = reader.read(batch)
#     for sample in datasets:
#         obj = pickle.loads(sample)
#         captions = obj["captions"]
#         n = len(captions)
#         for i in range(n):
#             hash_index[key_index] = (Media2Text_index,i)
#             key_index += 1
#         Media2Text_index += 1

# data = {"hash_index":hash_index}
# with open('/mnt/datasets/flickr30k_FFSlim/flickr30k-hash.pkl', 'wb') as f:
#     pickle.dump(data, f)


with open('/mnt/datasets/flickr30k_FFSlim/flickr30k-hash.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
hash_index = loaded_data["hash_index"]

print(f"hash_index size:{len(hash_index)}, hash_index:{hash_index}")


