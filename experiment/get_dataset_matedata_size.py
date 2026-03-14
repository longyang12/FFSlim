import os

def count_dirs_and_files(path):
    dir_count, file_count = 0, 0
    for root, dirs, files in os.walk(path):  # '.' 表示当前目录
        dir_count += len(dirs)   # 当前层子目录数量
        file_count += len(files) # 当前层文件数量
    return dir_count, file_count

# datasets = ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m"]
datasets = ["Clotho_E"]

for dataset in datasets:
    file_path = "/mnt/datasets/" + dataset + "_Files"
    dir_num, file_num = count_dirs_and_files(file_path)
    metadata_size = (dir_num+file_num)*4*1024
    print(f"_Files目录数量:{dir_num}，文件数量：{file_num},元数据大小：{metadata_size}")

    file_path = "/mnt/datasets/" + dataset + "_TDP"
    dir_num, file_num = count_dirs_and_files(file_path)
    metadata_size = (dir_num+file_num)*4*1024
    print(f"_TDP目录数量:{dir_num}，文件数量：{file_num},元数据大小：{metadata_size}")

    file_path = "/mnt/datasets/" + dataset + "_FFRecord"
    dir_num, file_num = count_dirs_and_files(file_path)
    metadata_size = (dir_num+file_num)*4*1024
    print(f"_FFRecord目录数量:{dir_num}，文件数量：{file_num},元数据大小：{metadata_size}")

    file_path = "/mnt/datasets/" + dataset + "_FFSlim"
    dir_num, file_num = count_dirs_and_files(file_path)
    metadata_size = (dir_num+file_num)*4*1024
    print(f"_FFSlim目录数量:{dir_num}，文件数量：{file_num},元数据大小：{metadata_size}")
