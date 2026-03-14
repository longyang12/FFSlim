import os

def get_total_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_total_size(entry.path)
    return total


# datasets = ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m"]
datasets = ["Clotho_E"]

for dataset in datasets:
    file_path = "/mnt/datasets/" + dataset + "_Files"
    print(f"_Files数据之和:{get_total_size(file_path)} 字节")

    file_path = "/mnt/datasets/" + dataset + "_TDP"
    print(f"_TDP数据之和:{get_total_size(file_path)} 字节")

    file_path = "/mnt/datasets/" + dataset + "_FFRecord"
    print(f"_FFRecord数据之和:{get_total_size(file_path)} 字节")

    file_path = "/mnt/datasets/" + dataset + "_FFSlim"
    print(f"_FFSlim数据之和:{get_total_size(file_path)} 字节")