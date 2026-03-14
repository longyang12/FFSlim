from ffrecord import FileReader
import pickle
import matplotlib.pyplot as plt


file_path = "/mnt/datasets/flickr30k_FFSlim/flickr30k_caption.ffr"
file_path = "/mnt/datasets/flickr30k_entities_FFSlim/flickr30k_entities_caption.ffr"
file_path = "/mnt/datasets/mscoco_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/vqa2_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/gqa_FFSlim/mscoco_caption.ffr"
file_path = "/mnt/datasets/cc3m_FFSlim/mscoco_caption.ffr"

reader = FileReader(file_path)
print(f"reader.n={reader.n}")
nums = []

sum = 0
for i in range(reader.n):
    datasets = reader.read([i])
    for sample in datasets:
        samples1 = pickle.loads(sample)
        n = len(samples1["captions"])
        nums.append(n)
        if n != 1:
            # print(f"i={i},n={n},captions={samples1["captions"]}")
            sum += (n-1)
print(sum)

# for num in nums:
#     with open("nums.txt","a+") as f:
#         f.write(f"{num} ")


# # 生成随机数据
# x = [i for i in range(len(nums))]
# y = nums

# # 绘制散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c='blue', s=10, alpha=0.6, edgecolors='w')
# plt.title('Basic Scatter Plot')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.grid(True)
# plt.show()
# plt.savefig("nums.png")