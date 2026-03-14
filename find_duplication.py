import os
from PIL import Image
import imagehash
from collections import defaultdict

def find_duplicate_images(folder_path):
    hash_dict = defaultdict(list)

    # 遍历目录及子目录
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img_hash = imagehash.phash(img)
                    hash_dict[str(img_hash)].append(file_path)
            except Exception as e:
                print(f"跳过无法打开的文件: {file_path}，错误: {e}")

    # 输出重复图片
    with open("dupinfo.out","a+") as f:
        f.write(f"\n重复图片文件列表:")
    duplicates_found = False
    for hash_value, paths in hash_dict.items():
        if len(paths) > 1:
            duplicates_found = True
            with open("dupinfo.out","a+") as f:
                f.write(f"\n哈希值 {hash_value} 的重复图片:")
            for p in paths:
                with open("dupinfo.out","a+") as f:
                    f.write(f"  - {p}")
    
    if not duplicates_found:
        print("未发现重复图片。")

# 示例用法
if __name__ == "__main__":
    folder = "/home/llm/experiment/dataset_Analysis_Modeling/datasets/cc3m/cc3m-images"  # 替换为你的路径
    find_duplicate_images(folder)
